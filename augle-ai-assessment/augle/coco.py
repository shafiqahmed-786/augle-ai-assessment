"""COCO JSON helpers: I/O, box utilities and a strict schema validator.

Coordinate conventions (used everywhere in this repository)
-----------------------------------------------------------
* Stored boxes follow COCO: ``[x, y, width, height]`` in absolute pixels,
  origin at the top-left corner of the image, x to the right, y downwards.
* Continuous pixel-edge convention: pixel ``(col i, row j)`` covers
  ``[i, i+1) x [j, j+1)``. A mask whose set pixels span columns ``xmin..xmax``
  therefore has ``x = xmin`` and ``width = xmax - xmin + 1``.
* Internally the models and the augmentation code use ``[x1, y1, x2, y2]`` (xyxy).
* Polygon segmentations are flat ``[x1, y1, x2, y2, ...]`` lists in the same continuous
  coordinates, i.e. they trace the *outer edge* of the mask pixels. A pixel belongs to the mask when
  its centre ``(i + 0.5, j + 0.5)`` lies inside the polygon (the rule used by pycocotools).
  Vertices therefore lie inside the annotation's own box ``[x, y, x + w, y + h]``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from augle.imaging import ImageReadError, read_image_bgr

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Box helpers
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(box: Sequence[float]) -> Box:
    """COCO ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``."""
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


def xyxy_to_xywh(box: Sequence[float]) -> Box:
    """``[x1, y1, x2, y2]`` -> COCO ``[x, y, w, h]``."""
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def clip_xyxy(box: Sequence[float], width: float, height: float) -> Box:
    """Clip an xyxy box to ``[0, width] x [0, height]``."""
    x1, y1, x2, y2 = box
    return (
        float(min(max(x1, 0.0), width)),
        float(min(max(y1, 0.0), height)),
        float(min(max(x2, 0.0), width)),
        float(min(max(y2, 0.0), height)),
    )


def box_area_xyxy(box: Sequence[float]) -> float:
    """Area of an xyxy box (0 for degenerate boxes)."""
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy arrays -> ``(N, M)``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices ordered by descending score."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = list(np.argsort(-scores, kind="stable"))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if not order:
            break
        ious = iou_matrix_xyxy(boxes[i : i + 1], boxes[order])[0]
        order = [j for j, v in zip(order, ious) if v <= iou_threshold]
    return keep


def polygons_to_mask(polygons: Sequence[Sequence[float]], height: int, width: int) -> np.ndarray:
    """Rasterise continuous-coordinate polygons to a boolean ``(H, W)`` mask.

    Uses ``pycocotools`` (the COCO reference implementation) so that masks built here are
    identical to what any COCO tool derives from the same JSON. ``cv2.fillPoly`` was measured and
    rejected: it is boundary-inclusive and over-covers by ~1 px per edge.
    """
    from pycocotools import mask as mask_utils

    polys = [[float(v) for v in p] for p in polygons if len(p) >= 6]
    if not polys:
        return np.zeros((height, width), dtype=bool)
    rle = mask_utils.merge(mask_utils.frPyObjects(polys, height, width))
    return mask_utils.decode(rle).astype(bool)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_coco(path: str | Path) -> dict[str, Any]:
    """Load a COCO JSON file (raises ``FileNotFoundError`` / ``ValueError`` with clear messages)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top-level JSON value must be an object")
    return data


def save_json_atomic(data: Any, path: str | Path, *, indent: int | None = 2) -> None:
    """Write JSON atomically (temp file + rename) so a crash never leaves a half-written file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600 files; outputs should be normally readable
        # On Windows, os.replace() raises PermissionError if the destination is open/locked.
        # Remove it first so the rename always succeeds on both Linux and Windows.
        if os.name == "nt" and p.exists():
            p.unlink()
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    """Outcome of :func:`validate_coco`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no errors were found (warnings are allowed)."""
        return not self.errors

    def summary(self, limit: int = 20) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"COCO validation: {'PASSED' if self.ok else 'FAILED'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        ]
        lines += [f"  stats: {json.dumps(self.stats, sort_keys=True)}"]
        for label, items in (("ERROR", self.errors), ("WARNING", self.warnings)):
            for msg in items[:limit]:
                lines.append(f"  {label}: {msg}")
            if len(items) > limit:
                lines.append(f"  ... {len(items) - limit} more {label.lower()}s")
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        """Raise ``ValueError`` carrying the summary when validation failed."""
        if not self.ok:
            raise ValueError(self.summary())


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_coco(
    coco: dict[str, Any],
    image_root: str | Path | None = None,
    *,
    check_files: bool = True,
    bounds_tol_px: float = 1.0,
    area_rel_tol: float = 0.02,
) -> ValidationReport:
    """Validate a COCO detection/segmentation dict.

    Checked: top-level structure, unique/consistent ids, image fields, file existence and
    true pixel dimensions, category fields, annotation fields, ``image_id``/``category_id``
    references, finite positive boxes inside the image, ``area`` consistency, ``iscrowd``,
    polygon/RLE segmentation validity and polygon-vs-box consistency, license references.
    """
    rep = ValidationReport()
    err, warn = rep.errors.append, rep.warnings.append

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            err(f"missing required top-level key '{key}'")
        elif not isinstance(coco[key], list):
            err(f"top-level '{key}' must be a list")
    if rep.errors:
        return rep

    images, annotations, categories = coco["images"], coco["annotations"], coco["categories"]
    licenses = coco.get("licenses", [])
    license_ids = {lic.get("id") for lic in licenses if isinstance(lic, dict)}
    root = Path(image_root) if image_root is not None else None
    if check_files and root is None:
        warn("image_root not given: image files and true dimensions were not checked")
        check_files = False

    # --- categories ------------------------------------------------------- #
    cat_ids: set[int] = set()
    cat_names: set[str] = set()
    for i, cat in enumerate(categories):
        if not isinstance(cat, dict) or not _is_int(cat.get("id")) or not isinstance(cat.get("name"), str):
            err(f"categories[{i}] needs integer 'id' and string 'name'")
            continue
        if not cat["name"].strip():
            err(f"categories[{i}] has an empty name")
        if cat["id"] in cat_ids:
            err(f"duplicate category id {cat['id']}")
        if cat["name"] in cat_names:
            err(f"duplicate category name '{cat['name']}'")
        cat_ids.add(cat["id"])
        cat_names.add(cat["name"])
    if not categories:
        err("'categories' is empty")

    # --- images ------------------------------------------------------------ #
    image_dims: dict[int, tuple[int, int]] = {}
    file_names: set[str] = set()
    for i, im in enumerate(images):
        if not isinstance(im, dict):
            err(f"images[{i}] is not an object")
            continue
        iid = im.get("id")
        if not _is_int(iid):
            err(f"images[{i}] has non-integer id {iid!r}")
            continue
        if iid in image_dims:
            err(f"duplicate image id {iid}")
        w, h, fn = im.get("width"), im.get("height"), im.get("file_name")
        if not (_is_int(w) and _is_int(h) and w > 0 and h > 0):
            err(f"image {iid}: width/height must be positive integers (got {w!r}, {h!r})")
            continue
        if not isinstance(fn, str) or not fn:
            err(f"image {iid}: missing file_name")
            continue
        if fn in file_names:
            warn(f"image {iid}: file_name '{fn}' is used by more than one image entry")
        file_names.add(fn)
        image_dims[iid] = (w, h)
        if "license" in im and licenses and im["license"] not in license_ids:
            err(f"image {iid}: license id {im['license']!r} not present in 'licenses'")
        if check_files and root is not None:
            path = root / fn
            try:
                pixels = read_image_bgr(path)
            except ImageReadError as exc:
                err(f"image {iid}: {exc}")
                continue
            if pixels.shape[0] != h or pixels.shape[1] != w:
                err(f"image {iid}: JSON says {w}x{h} but file is {pixels.shape[1]}x{pixels.shape[0]} ({fn})")

    # --- annotations ------------------------------------------------------- #
    ann_ids: set[int] = set()
    n_seg = 0
    per_image: dict[int, int] = {}
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            err(f"annotations[{i}] is not an object")
            continue
        aid = ann.get("id")
        if not _is_int(aid):
            err(f"annotations[{i}] has non-integer id {aid!r}")
            continue
        if aid in ann_ids:
            err(f"duplicate annotation id {aid}")
        if aid < 1:
            warn(f"annotation id {aid} < 1 (COCO ids conventionally start at 1)")
        ann_ids.add(aid)
        iid, cid = ann.get("image_id"), ann.get("category_id")
        if iid not in image_dims:
            err(f"annotation {aid}: image_id {iid!r} does not match any image")
            continue
        if cid not in cat_ids:
            err(f"annotation {aid}: category_id {cid!r} does not match any category")
        per_image[iid] = per_image.get(iid, 0) + 1
        iw, ih = image_dims[iid]

        bbox = ann.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(_is_num(v) for v in bbox)):
            err(f"annotation {aid}: bbox must be 4 finite numbers [x, y, w, h] (got {bbox!r})")
            continue
        x, y, bw, bh = (float(v) for v in bbox)
        if bw <= 0 or bh <= 0:
            err(f"annotation {aid}: degenerate bbox {bbox} (width/height must be > 0)")
            continue
        if x < -bounds_tol_px or y < -bounds_tol_px or x + bw > iw + bounds_tol_px or y + bh > ih + bounds_tol_px:
            err(f"annotation {aid}: bbox {bbox} outside image {iw}x{ih}")

        area = ann.get("area")
        seg = ann.get("segmentation", [])
        if not _is_num(area) or area <= 0:
            err(f"annotation {aid}: area must be a positive finite number (got {area!r})")
        elif seg:
            if area > bw * bh * (1 + area_rel_tol) + 1.0:
                err(f"annotation {aid}: mask area {area} exceeds bbox area {bw * bh:.1f}")
        elif abs(area - bw * bh) > area_rel_tol * bw * bh + 1e-6:
            err(f"annotation {aid}: area {area} != bbox w*h {bw * bh:.1f} (no segmentation present)")

        if ann.get("iscrowd", 0) not in (0, 1):
            err(f"annotation {aid}: iscrowd must be 0 or 1")

        if isinstance(seg, list) and seg:
            n_seg += 1
            for p_i, poly in enumerate(seg):
                if not (
                    isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0 and all(_is_num(v) for v in poly)
                ):
                    err(f"annotation {aid}: polygon {p_i} must be an even-length list of >= 6 finite numbers")
                    continue
                xs, ys = poly[0::2], poly[1::2]
                tol = bounds_tol_px + 1.0
                if min(xs) < x - tol or max(xs) > x + bw + tol or min(ys) < y - tol or max(ys) > y + bh + tol:
                    err(f"annotation {aid}: polygon {p_i} extends outside its bbox {bbox}")
                if (
                    min(xs) < -bounds_tol_px
                    or max(xs) > iw + bounds_tol_px
                    or min(ys) < -bounds_tol_px
                    or max(ys) > ih + bounds_tol_px
                ):
                    err(f"annotation {aid}: polygon {p_i} outside image {iw}x{ih}")
        elif isinstance(seg, dict):
            n_seg += 1
            if "counts" not in seg or seg.get("size") != [ih, iw]:
                err(f"annotation {aid}: RLE needs 'counts' and size == [{ih}, {iw}] (got {seg.get('size')!r})")
        elif seg not in ([], None):
            err(f"annotation {aid}: unsupported segmentation type {type(seg).__name__}")

    empty_images = sum(1 for iid in image_dims if iid not in per_image)
    if empty_images:
        warn(f"{empty_images} image(s) have no annotations")
    if not annotations:
        err("'annotations' is empty")
    rep.stats = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "annotations_with_segmentation": n_seg,
        "images_without_annotations": empty_images,
        "files_checked": bool(check_files),
    }
    return rep
