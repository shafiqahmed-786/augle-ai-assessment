"""Zero-shot auto-labelling: Grounding DINO boxes, optional SAM masks, COCO assembly.

The model wrappers (:class:`GroundingDinoDetector`, :class:`SamSegmenter`) import
``transformers`` lazily and are the only code that needs model weights. Everything else
(post-processing, polygon extraction, COCO building, caching) is pure and unit-tested with
stub detectors that are clearly labelled as test doubles.

COCO conventions: see ``augle/coco.py``. When SAM masks exist, ``bbox`` is the tight pixel box
of the mask (``w = xmax - xmin + 1``), ``area`` is the mask pixel count and ``segmentation``
holds polygons tracing the outer pixel edges (continuous coordinates). Without masks, ``bbox`` is the detector box,
``area = w * h`` and ``segmentation`` is ``[]`` - no mask is ever invented from a box.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from augle.coco import (
    box_area_xyxy,
    clip_xyxy,
    nms_xyxy,
    polygons_to_mask,
    save_json_atomic,
    validate_coco,
    xyxy_to_xywh,
)
from augle.config import AnnotationConfig
from augle.imaging import ImageReadError, read_image_bgr

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected instance in absolute xyxy pixel coordinates of the source image."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    category: str
    polygons: list[list[float]] = field(default_factory=list)
    mask_area: float = 0.0
    mask_bbox_xywh: tuple[float, float, float, float] | None = None


class Detector(Protocol):
    """Open-vocabulary detector returning boxes for each requested category prompt."""

    def detect(self, image_bgr: np.ndarray, prompts: dict[str, list[str]]) -> list[Detection]:
        """Detect instances of every category in ``prompts`` (category -> phrases)."""


class Segmenter(Protocol):
    """Promptable segmenter: one boolean mask per input box."""

    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        """Return ``len(boxes)`` boolean ``(H, W)`` masks."""


def resolve_device(requested: str) -> str:
    """Map ``auto`` to ``cuda`` when available, else ``cpu``."""
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Foundation-model wrappers (need downloaded weights)
# --------------------------------------------------------------------------- #
class GroundingDinoDetector:
    """Grounding DINO through Hugging Face ``transformers`` (one text prompt per category)."""

    def __init__(self, model_id: str, box_threshold: float, text_threshold: float, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = resolve_device(device)
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        LOGGER.info("loading Grounding DINO %s on %s", model_id, self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def _detect_phrase(self, rgb: np.ndarray, phrase: str) -> list[tuple[tuple[float, float, float, float], float]]:
        from PIL import Image

        h, w = rgb.shape[:2]
        text = phrase.lower().strip().rstrip(".") + "."  # Grounding DINO expects lower-case, dot-terminated text
        inputs = self.processor(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        return [(tuple(float(v) for v in b), float(s)) for b, s in zip(boxes, scores)]

    def detect(self, image_bgr: np.ndarray, prompts: dict[str, list[str]]) -> list[Detection]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        found: list[Detection] = []
        for category, phrases in prompts.items():
            for phrase in phrases:
                for box, score in self._detect_phrase(rgb, phrase):
                    found.append(Detection(box_xyxy=box, score=score, category=category))
        return found


class SamSegmenter:
    """Segment Anything (``transformers`` ``SamModel``) prompted with detector boxes."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        import torch
        from transformers import SamModel, SamProcessor

        self.torch = torch
        self.device = resolve_device(device)
        LOGGER.info("loading SAM %s on %s", model_id, self.device)
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()

    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
        from PIL import Image

        if len(boxes_xyxy) == 0:
            return []
        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(rgb, input_boxes=[np.asarray(boxes_xyxy, dtype=float).tolist()], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]  # (num_boxes, num_masks, H, W) bool
        return [masks[i, 0].numpy().astype(bool) for i in range(masks.shape[0])]


# --------------------------------------------------------------------------- #
# Pure post-processing
# --------------------------------------------------------------------------- #
def filter_detections(
    dets: list[Detection], width: int, height: int, cfg: AnnotationConfig
) -> tuple[list[Detection], dict[str, int]]:
    """Clip, validate, size-filter and NMS detections per category.

    Returns the kept detections and a ``reason -> count`` dict for everything discarded.
    """
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    image_area = float(width * height)
    by_cat: dict[str, list[Detection]] = {}
    for det in dets:
        box = det.box_xyxy
        if not np.all(np.isfinite(box)) or not np.isfinite(det.score):
            drop("non_finite")
            continue
        clipped = clip_xyxy(box, width, height)
        area = box_area_xyxy(clipped)
        if clipped[2] - clipped[0] < 1 or clipped[3] - clipped[1] < 1:
            drop("degenerate_box")
            continue
        frac = area / image_area
        if frac < cfg.min_box_area_frac:
            drop("box_too_small")
            continue
        if frac > cfg.max_box_area_frac:
            drop("box_covers_image")
            continue
        det.box_xyxy = clipped
        by_cat.setdefault(det.category, []).append(det)
    kept: list[Detection] = []
    for cat_dets in by_cat.values():
        boxes = np.array([d.box_xyxy for d in cat_dets])
        scores = np.array([d.score for d in cat_dets])
        order = nms_xyxy(boxes, scores, cfg.nms_iou)
        dropped["suppressed_by_nms"] = dropped.get("suppressed_by_nms", 0) + len(cat_dets) - len(order)
        capped = order[: cfg.max_detections_per_category]
        dropped["over_max_per_category"] = dropped.get("over_max_per_category", 0) + len(order) - len(capped)
        kept.extend(cat_dets[i] for i in capped)
    return kept, {k: v for k, v in dropped.items() if v}


def mask_to_polygons(
    mask: np.ndarray, min_area_px: float = 16.0, epsilon_px: float = 0.3, upsample: int = 4
) -> list[list[float]]:
    """Convert a binary mask to COCO polygons that trace the outer pixel edges (external contours).

    The mask is cropped to its bounding box, zero-padded and up-sampled ``upsample`` times with
    nearest-neighbour before ``cv2.findContours``; contour points (pixel centres of the up-sampled
    grid) are mapped back to continuous source coordinates, which places them within
    ``0.5 / upsample`` pixels of the true pixel edges. Contours are then simplified with
    ``approxPolyDP`` (``epsilon_px`` pixels). Contours with area below ``min_area_px`` or fewer
    than 3 vertices are discarded.
    """
    mask = np.asarray(mask).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = np.pad(mask[y0:y1, x0:x1], 1)
    big = cv2.resize(crop, None, fx=upsample, fy=upsample, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: list[list[float]] = []
    for contour in contours:
        pts = (contour.reshape(-1, 2).astype(np.float64) + 0.5) / upsample - 1.0 + np.array([x0, y0], dtype=np.float64)
        pts32 = pts.astype(np.float32).reshape(-1, 1, 2)
        if cv2.contourArea(pts32) < min_area_px:
            continue
        approx = cv2.approxPolyDP(pts32, epsilon_px, True) if epsilon_px > 0 else pts32
        if len(approx) < 3:
            continue
        polygons.append([float(v) for v in approx.reshape(-1, 2).flatten()])
    return polygons


def attach_mask(det: Detection, mask: np.ndarray, cfg: AnnotationConfig) -> bool:
    """Store polygons / area / tight bbox from ``mask`` on ``det``. Returns False if unusable.

    ``area`` and ``bbox`` are computed from the rasterisation of the *stored* polygons (the same
    rasteriser the Dataset uses), so annotation fields always agree with each other.
    """
    polygons = mask_to_polygons(mask, cfg.min_polygon_area_px, cfg.polygon_epsilon_px)
    if not polygons:
        return False
    raster = polygons_to_mask(polygons, *mask.shape[:2])
    ys, xs = np.nonzero(raster)
    if len(xs) == 0:
        return False
    det.polygons = polygons
    det.mask_area = float(len(xs))
    det.mask_bbox_xywh = (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    )
    return True


# --------------------------------------------------------------------------- #
# COCO assembly
# --------------------------------------------------------------------------- #
def license_table(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    """Build the COCO ``licenses`` list and a ``(name, url) -> id`` lookup from manifest records."""
    lookup: dict[tuple[str, str], int] = {}
    table: list[dict[str, Any]] = []
    for rec in records:
        key = (rec.get("license_short", "") or "unknown", rec.get("license_url", ""))
        if key not in lookup:
            lookup[key] = len(table) + 1
            table.append({"id": lookup[key], "name": key[0], "url": key[1]})
    return table, lookup


def build_coco(
    records: list[dict[str, Any]],
    detections: dict[str, list[Detection]],
    sizes: dict[str, tuple[int, int]],
    category_names: list[str],
    *,
    description: str = "Industrial robot arm dataset auto-labelled with Grounding DINO (+SAM)",
) -> dict[str, Any]:
    """Assemble a COCO dict. ``records`` are manifest dicts; ``detections`` is keyed by ``image_id``.

    Only images that have at least one detection are expected in ``detections``; callers decide
    whether to drop the rest. Image ids and annotation ids are 1-based consecutive integers.
    """
    licenses, lookup = license_table(records)
    cat_ids = {name: i + 1 for i, name in enumerate(category_names)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_ann = 1
    for rec in records:
        dets = detections.get(rec["image_id"])
        if dets is None:
            continue
        width, height = sizes[rec["image_id"]]
        coco_image_id = len(images) + 1
        key = (rec.get("license_short", "") or "unknown", rec.get("license_url", ""))
        images.append(
            {
                "id": coco_image_id,
                "file_name": rec["file_name"],
                "width": width,
                "height": height,
                "license": lookup[key],
                "coco_url": rec.get("original_url", ""),
                "date_captured": "",
                "source_page": rec.get("page_url", ""),
                "author": rec.get("author", ""),
                "attribution": rec.get("credit", ""),
                "source_title": rec.get("title", ""),
            }
        )
        for det in dets:
            if det.polygons and det.mask_bbox_xywh is not None:
                bbox, area, seg = list(det.mask_bbox_xywh), det.mask_area, det.polygons
            else:
                bbox = list(xyxy_to_xywh(det.box_xyxy))
                area, seg = bbox[2] * bbox[3], []
            annotations.append(
                {
                    "id": next_ann,
                    "image_id": coco_image_id,
                    "category_id": cat_ids[det.category],
                    "bbox": [round(float(v), 2) for v in bbox],
                    "area": round(float(area), 2),
                    "iscrowd": 0,
                    "segmentation": [[round(v, 2) for v in p] for p in seg],
                    "score": round(float(det.score), 4),
                    "detector_box_xyxy": [round(float(v), 2) for v in det.box_xyxy],
                }
            )
            next_ann += 1
    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "auto-generated (zero-shot); labels are model pseudo-labels, not human verified",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cat_ids[n], "name": n, "supercategory": "robot"} for n in category_names],
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class AnnotationSummary:
    """Counts describing what the annotation run produced and why images were dropped."""

    images_in_manifest: int = 0
    images_processed: int = 0
    images_with_detections: int = 0
    images_without_detections: int = 0
    images_failed: int = 0
    annotations: int = 0
    annotations_with_masks: int = 0
    detections_dropped: dict[str, int] = field(default_factory=dict)
    per_category: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)


def config_fingerprint(cfg: AnnotationConfig) -> str:
    """Hash of every setting that changes model output, used to invalidate stale cache files."""
    relevant = {k: v for k, v in asdict(cfg).items() if k not in {"cache_dir", "output_json", "device"}}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _det_to_json(d: Detection) -> dict[str, Any]:
    return asdict(d)


def _det_from_json(d: dict[str, Any]) -> Detection:
    return Detection(
        box_xyxy=tuple(d["box_xyxy"]),
        score=d["score"],
        category=d["category"],
        polygons=d.get("polygons", []),
        mask_area=d.get("mask_area", 0.0),
        mask_bbox_xywh=tuple(d["mask_bbox_xywh"]) if d.get("mask_bbox_xywh") else None,
    )


def annotate_collection(
    manifest_path: str | Path,
    image_root: str | Path,
    detector: Detector,
    segmenter: Segmenter | None,
    cfg: AnnotationConfig,
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], AnnotationSummary]:
    """Run detection (+ optional segmentation) on every manifest image and build the COCO dict.

    Per-image results are cached as JSON under ``cfg.cache_dir`` so an interrupted run resumes
    without repeating model inference. Model failures are recorded per image and never abort
    the run; nothing is fabricated for failed images.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = manifest["accepted"][: limit or None]
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = AnnotationSummary(images_in_manifest=len(records))
    fingerprint = config_fingerprint(cfg)
    detections: dict[str, list[Detection]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    kept_records: list[dict[str, Any]] = []
    for rec in records:
        cache_file = cache_dir / f"{rec['image_id']}.json"
        try:
            image = read_image_bgr(Path(image_root) / rec["file_name"])
            h, w = image.shape[:2]
            cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else None
            if cached is not None and cached.get("fingerprint") == fingerprint:
                dets = [_det_from_json(d) for d in cached["detections"]]
                dropped = cached.get("dropped", {})
            else:
                raw = detector.detect(image, cfg.categories)
                dets, dropped = filter_detections(raw, w, h, cfg)
                if segmenter is not None and dets:
                    masks = segmenter.segment(image, np.array([d.box_xyxy for d in dets]))
                    usable: list[Detection] = []
                    for det, mask in zip(dets, masks):
                        if attach_mask(det, mask, cfg):
                            usable.append(det)
                        else:
                            dropped["empty_mask"] = dropped.get("empty_mask", 0) + 1
                    dets = usable
                save_json_atomic(
                    {"fingerprint": fingerprint, "detections": [_det_to_json(d) for d in dets], "dropped": dropped},
                    cache_file,
                    indent=None,
                )
        except (ImageReadError, RuntimeError, ValueError, OSError) as exc:
            LOGGER.error("annotation failed for %s: %s", rec["image_id"], exc)
            summary.images_failed += 1
            summary.failures.append({"image_id": rec["image_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        summary.images_processed += 1
        for reason, n in dropped.items():
            summary.detections_dropped[reason] = summary.detections_dropped.get(reason, 0) + n
        sizes[rec["image_id"]] = (w, h)
        if dets:
            summary.images_with_detections += 1
            detections[rec["image_id"]] = dets
            kept_records.append(rec)
        else:
            summary.images_without_detections += 1
            if not cfg.drop_images_without_detections:
                detections[rec["image_id"]] = []
                kept_records.append(rec)
    names = list(cfg.categories)
    coco = build_coco(kept_records, detections, sizes, names)
    summary.annotations = len(coco["annotations"])
    summary.annotations_with_masks = sum(1 for a in coco["annotations"] if a["segmentation"])
    for ann in coco["annotations"]:
        cname = names[ann["category_id"] - 1]
        summary.per_category[cname] = summary.per_category.get(cname, 0) + 1
    return coco, summary


def write_and_validate(coco: dict[str, Any], out_json: str | Path, image_root: str | Path) -> None:
    """Persist the COCO file, then validate it (including real image dimensions); raise on errors."""
    save_json_atomic(coco, out_json)
    report = validate_coco(coco, image_root, check_files=True)
    LOGGER.info("\n%s", report.summary())
    report.raise_if_invalid()
