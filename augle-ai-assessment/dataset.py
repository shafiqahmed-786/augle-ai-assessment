"""Custom PyTorch ``Dataset`` over COCO JSON plus the complete OpenCV preprocessing pipeline.

Everything geometric lives here and is covered by ``tests/test_geometry.py``:

* **letterbox resize** - aspect-preserving resize + constant padding to a square canvas;
* **flips** - horizontal / vertical;
* **affine augmentation** - rotation, isotropic scale, shear and translation composed into one
  3x3 matrix;
* **normalisation** - per-channel mean/std, HWC uint8 -> CHW float32.

Conventions (see also ``augle/coco.py``)
----------------------------------------
Boxes are ``[x1, y1, x2, y2]`` floats in continuous pixel-edge coordinates (pixel ``i`` covers
``[i, i+1)``). Affine matrices are expressed in that continuous system (``A``); OpenCV indexes
by pixel *centres*, so the matrix handed to ``cv2.warpAffine`` is ``T(-0.5) @ A @ T(+0.5)``.
Using the same ``A`` for boxes and (converted) for pixels keeps boxes, images and masks aligned
to sub-pixel accuracy. Boxes under an affine map are obtained by transforming the four corners
and taking the axis-aligned envelope, then clipping to the canvas; when masks are available the
envelope is optionally replaced by the tight box of the warped mask.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augle.coco import load_coco, polygons_to_mask, validate_coco, xywh_to_xyxy
from augle.config import DataConfig
from augle.imaging import read_image_bgr

LOGGER = logging.getLogger(__name__)
PAD_VALUE = 114


# --------------------------------------------------------------------------- #
# Sample container
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One image with its instance annotations.

    Attributes:
        image: ``(H, W, 3)`` ``uint8`` RGB.
        boxes: ``(N, 4)`` ``float32`` xyxy in pixels of ``image``.
        labels: ``(N,)`` ``int64`` contiguous class indices.
        masks: ``(N, H, W)`` ``uint8`` in ``{0, 1}`` or ``None`` when unavailable.
    """

    image: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if len(self.boxes) != len(self.labels):
            raise ValueError("boxes and labels must have the same length")
        if self.masks is not None:
            self.masks = np.asarray(self.masks, dtype=np.uint8)
            if self.masks.shape != (len(self.boxes), *self.image.shape[:2]):
                raise ValueError(
                    f"masks shape {self.masks.shape} != (N={len(self.boxes)}, H, W)={self.image.shape[:2]}"
                )

    def select(self, keep: np.ndarray) -> Sample:
        """Return a copy restricted to the instances selected by ``keep`` (bool mask or indices)."""
        return Sample(
            self.image,
            self.boxes[keep],
            self.labels[keep],
            None if self.masks is None else self.masks[keep],
        )


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip ``(N, 4)`` xyxy boxes to the canvas ``[0, width] x [0, height]``."""
    out = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
    return out


def box_areas(boxes: np.ndarray) -> np.ndarray:
    """Areas of ``(N, 4)`` xyxy boxes (zero for degenerate boxes)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def transform_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 / 3x3 affine map to xyxy boxes via their four corners.

    The result is the axis-aligned envelope of the transformed quadrilateral, so it stays
    correct under rotation and shear (not just scale + translation).
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(b) == 0:
        return b.astype(np.float32)
    m = np.asarray(matrix, dtype=np.float64)[:2]
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1)  # (N, 4, 2)
    moved = corners @ m[:, :2].T + m[:, 2]
    out = np.concatenate([moved.min(axis=1), moved.max(axis=1)], axis=1)
    return out.astype(np.float32)


def mask_tight_boxes(masks: np.ndarray) -> np.ndarray:
    """Tight xyxy boxes of binary masks ``(N, H, W)``; empty masks give ``[0, 0, 0, 0]``."""
    out = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            out[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return out


def build_affine(
    width: int,
    height: int,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    shear_x_deg: float = 0.0,
    shear_y_deg: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    """Compose ``A = T(c + t) @ R(angle) S(scale) @ Shear @ T(-c)`` as a 3x3 matrix.

    ``c`` is the continuous centre of the canvas ``(width / 2, height / 2)``. Positive
    ``angle_deg`` rotates counter-clockwise as displayed (image y axis points down).
    """
    cx, cy = width / 2.0, height / 2.0
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    rot_scale = np.array([[scale * cos_a, scale * sin_a, 0.0], [-scale * sin_a, scale * cos_a, 0.0], [0.0, 0.0, 1.0]])
    shear = np.array(
        [[1.0, np.tan(np.deg2rad(shear_x_deg)), 0.0], [np.tan(np.deg2rad(shear_y_deg)), 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    back = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]])
    return back @ rot_scale @ shear @ to_origin


def to_opencv_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a continuous-coordinate affine to the pixel-centre convention of OpenCV (2x3)."""
    m = np.eye(3)
    m[:2] = np.asarray(matrix, dtype=np.float64)[:2]
    half = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    unhalf = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
    return (unhalf @ m @ half)[:2]


# --------------------------------------------------------------------------- #
# Sample-level transforms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LetterboxInfo:
    """Parameters of a letterbox so that coordinates can be mapped back to the source image."""

    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int


def letterbox(sample: Sample, size: int, pad_value: int = PAD_VALUE) -> tuple[Sample, LetterboxInfo]:
    """Resize preserving aspect ratio and pad to a ``size x size`` canvas.

    Boxes use the exact per-axis scale ``new_dim / old_dim`` actually applied by OpenCV, plus the
    integer padding offsets. Masks are resized with nearest-neighbour so they stay binary.
    """
    h, w = sample.image.shape[:2]
    ratio = size / float(max(h, w))
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(sample.image, (new_w, new_h), interpolation=interp)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    sx, sy = new_w / float(w), new_h / float(h)
    boxes = sample.boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx + pad_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy + pad_y
    boxes = clip_boxes(boxes, size, size)

    masks = None
    if sample.masks is not None:
        masks = np.zeros((len(sample.masks), size, size), dtype=np.uint8)
        for i, m in enumerate(sample.masks):
            small = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            masks[i, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = small
    return Sample(canvas, boxes, sample.labels, masks), LetterboxInfo(sx, sy, pad_x, pad_y)


def flip_horizontal(sample: Sample) -> Sample:
    """Mirror left-right: ``x1' = W - x2``, ``x2' = W - x1``."""
    h, w = sample.image.shape[:2]
    boxes = sample.boxes.copy()
    boxes[:, 0], boxes[:, 2] = w - sample.boxes[:, 2], w - sample.boxes[:, 0]
    masks = None if sample.masks is None else np.ascontiguousarray(sample.masks[:, :, ::-1])
    return Sample(np.ascontiguousarray(sample.image[:, ::-1]), boxes, sample.labels, masks)


def flip_vertical(sample: Sample) -> Sample:
    """Mirror top-bottom: ``y1' = H - y2``, ``y2' = H - y1``."""
    h, w = sample.image.shape[:2]
    boxes = sample.boxes.copy()
    boxes[:, 1], boxes[:, 3] = h - sample.boxes[:, 3], h - sample.boxes[:, 1]
    masks = None if sample.masks is None else np.ascontiguousarray(sample.masks[:, ::-1, :])
    return Sample(np.ascontiguousarray(sample.image[::-1]), boxes, sample.labels, masks)


def apply_affine(
    sample: Sample,
    matrix: np.ndarray,
    out_size: tuple[int, int] | None = None,
    *,
    min_box_side: float = 2.0,
    min_visibility: float = 0.3,
    refine_with_masks: bool = True,
    pad_value: int = PAD_VALUE,
) -> Sample:
    """Warp image, masks and boxes with the same continuous-coordinate affine ``matrix``.

    Instances are dropped when less than ``min_visibility`` of their (transformed) box area stays
    inside the canvas or when the clipped box is thinner than ``min_box_side`` pixels.
    ``out_size`` is ``(width, height)`` and defaults to the input size.
    """
    h, w = sample.image.shape[:2]
    out_w, out_h = out_size if out_size is not None else (w, h)
    cv_m = to_opencv_matrix(matrix)
    image = cv2.warpAffine(
        sample.image,
        cv_m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(pad_value,) * 3,
    )
    moved = transform_boxes(sample.boxes, matrix)
    full_area = box_areas(moved)
    clipped = clip_boxes(moved, out_w, out_h)
    visible = box_areas(clipped) / np.maximum(full_area, 1e-9)
    side_ok = ((clipped[:, 2] - clipped[:, 0]) >= min_box_side) & ((clipped[:, 3] - clipped[:, 1]) >= min_box_side)
    keep = (visible >= min_visibility) & side_ok

    masks = None
    if sample.masks is not None:
        masks = np.zeros((len(sample.masks), out_h, out_w), dtype=np.uint8)
        for i, m in enumerate(sample.masks):
            masks[i] = cv2.warpAffine(
                m,
                cv_m,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    boxes = clipped
    if masks is not None and refine_with_masks:
        tight = mask_tight_boxes(masks)
        has_pixels = masks.reshape(len(masks), -1).any(axis=1) if len(masks) else np.zeros(0, bool)
        keep &= has_pixels
        boxes = np.where(has_pixels[:, None], tight, clipped)
        side_ok = ((boxes[:, 2] - boxes[:, 0]) >= min_box_side) & ((boxes[:, 3] - boxes[:, 1]) >= min_box_side)
        keep &= side_ok
    out = Sample(image, boxes, sample.labels, masks)
    return out.select(keep)


def photometric_jitter(image: np.ndarray, brightness: float, contrast: float, rng: np.random.Generator) -> np.ndarray:
    """Random brightness (multiplicative) and contrast (about mid-grey) jitter; geometry untouched."""
    b = 1.0 + rng.uniform(-brightness, brightness)
    c = 1.0 + rng.uniform(-contrast, contrast)
    out = (image.astype(np.float32) - 127.5) * c + 127.5 * b
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_image(image_rgb: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """``uint8`` HWC RGB -> ``float32`` CHW, scaled to [0, 1] then standardised per channel."""
    arr = image_rgb.astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def denormalize_to_uint8(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`normalize_image` for visualisation: CHW float tensor -> HWC uint8 RGB."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)


class DetectionTransform:
    """Train/eval pipeline: letterbox -> (flips -> random affine -> photometric) -> arrays."""

    def __init__(self, size: int, train: bool, cfg: DataConfig) -> None:
        self.size, self.train, self.cfg = size, train, cfg

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        """Apply the pipeline; all randomness is drawn from ``rng`` for reproducibility."""
        cfg = self.cfg
        sample, _ = letterbox(sample, self.size)
        if not self.train:
            return sample
        if rng.random() < cfg.hflip_p:
            sample = flip_horizontal(sample)
        if rng.random() < cfg.vflip_p:
            sample = flip_vertical(sample)
        matrix = build_affine(
            self.size,
            self.size,
            angle_deg=rng.uniform(-cfg.rotation_deg, cfg.rotation_deg),
            scale=rng.uniform(*cfg.scale_range),
            shear_x_deg=rng.uniform(-cfg.shear_deg, cfg.shear_deg),
            shear_y_deg=rng.uniform(-cfg.shear_deg, cfg.shear_deg),
            tx=rng.uniform(-cfg.translate_frac, cfg.translate_frac) * self.size,
            ty=rng.uniform(-cfg.translate_frac, cfg.translate_frac) * self.size,
        )
        sample = apply_affine(
            sample,
            matrix,
            (self.size, self.size),
            min_box_side=cfg.min_box_side_px,
            min_visibility=cfg.min_visibility,
            refine_with_masks=cfg.refine_boxes_with_masks,
        )
        return replace(sample, image=photometric_jitter(sample.image, cfg.brightness, cfg.contrast, rng))


# --------------------------------------------------------------------------- #
# Mask rasterisation
# --------------------------------------------------------------------------- #
def segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray | None:
    """Rasterise a COCO polygon list or RLE (compressed or uncompressed) to ``uint8``; ``None`` if absent."""
    if isinstance(segmentation, list) and segmentation:
        return polygons_to_mask(segmentation, height, width).astype(np.uint8)
    if isinstance(segmentation, dict):
        from pycocotools import mask as mask_utils

        rle = segmentation
        if isinstance(rle.get("counts"), list):  # uncompressed RLE -> compressed
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(np.uint8)
    return None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CocoDetectionDataset(Dataset):
    """COCO-JSON backed detection dataset (written from scratch, no ``torchvision`` dataset).

    Each item is a dict with ``image`` (3, S, S) float32, ``boxes`` (N, 4) float32 xyxy in the
    letterboxed canvas, ``labels`` (N,) int64 contiguous class ids, optional ``masks`` (N, S, S)
    uint8, plus ``image_id``, ``file_name`` and ``orig_size``.

    Augmentation randomness is derived from ``(seed, epoch, index)`` so results are identical
    regardless of ``num_workers``; call :meth:`set_epoch` once per epoch to vary augmentations.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        *,
        image_ids: Sequence[int] | None = None,
        train: bool = False,
        cfg: DataConfig | None = None,
        seed: int = 0,
        validate: bool = True,
        skip_empty: bool = False,
    ) -> None:
        self.cfg = cfg or DataConfig()
        self.image_root = Path(image_root)
        self.train, self.seed, self.epoch = train, seed, 0
        coco = load_coco(annotation_file)
        if validate:
            validate_coco(coco, check_files=False).raise_if_invalid()
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        self.category_ids = [c["id"] for c in categories]
        self.class_names = [c["name"] for c in categories]
        self._cat_to_label = {cid: i for i, cid in enumerate(self.category_ids)}
        by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)
        images = {im["id"]: im for im in coco["images"]}
        wanted = sorted(images) if image_ids is None else list(image_ids)
        missing = [i for i in wanted if i not in images]
        if missing:
            raise KeyError(f"image ids not present in {annotation_file}: {missing[:5]}")
        if skip_empty:
            wanted = [i for i in wanted if by_image.get(i)]
        self.records = [(images[i], by_image.get(i, [])) for i in wanted]
        self.transform = DetectionTransform(self.cfg.image_size, train, self.cfg)

    @property
    def num_classes(self) -> int:
        """Number of categories in the COCO file."""
        return len(self.category_ids)

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation stream (call at the start of each training epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def load_sample(self, index: int) -> Sample:
        """Load the raw (un-augmented) sample at ``index`` with boxes clipped to the image."""
        info, anns = self.records[index]
        bgr = read_image_bgr(self.image_root / info["file_name"])
        h, w = bgr.shape[:2]
        if (h, w) != (info["height"], info["width"]):
            raise ValueError(f"{info['file_name']}: file is {w}x{h}, JSON says {info['width']}x{info['height']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes, labels, masks = [], [], []
        for ann in anns:
            x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
            x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, w), min(y2, h)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                LOGGER.warning("dropping degenerate box in %s (ann %s)", info["file_name"], ann["id"])
                continue
            boxes.append((x1, y1, x2, y2))
            labels.append(self._cat_to_label[ann["category_id"]])
            masks.append(segmentation_to_mask(ann.get("segmentation"), h, w))
        use_masks = self.cfg.load_masks and boxes and all(m is not None for m in masks)
        return Sample(
            rgb,
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.stack(masks) if use_masks else None,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        info, _ = self.records[index]
        raw = self.load_sample(index)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        out = self.transform(raw, rng)
        image = normalize_image(out.image, self.cfg.mean, self.cfg.std)
        return {
            "image": torch.from_numpy(image),
            "boxes": torch.from_numpy(out.boxes.astype(np.float32)),
            "labels": torch.from_numpy(out.labels),
            "masks": None if out.masks is None else torch.from_numpy(out.masks),
            "image_id": int(info["id"]),
            "file_name": info["file_name"],
            "orig_size": (int(info["height"]), int(info["width"])),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack images; keep variable-length targets as lists."""
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "masks": [b["masks"] for b in batch],
        "image_ids": torch.tensor([b["image_id"] for b in batch], dtype=torch.long),
        "file_names": [b["file_name"] for b in batch],
    }


def make_splits(image_ids: Sequence[int], fractions: Sequence[float], seed: int) -> dict[str, list[int]]:
    """Deterministic, disjoint train/val/test split of image ids.

    Each split gets at least one image when ``len(image_ids) >= 3``.
    """
    ids = sorted(int(i) for i in image_ids)
    n = len(ids)
    if n < 3:
        raise ValueError(f"need at least 3 images to split, got {n}")
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6 or min(fractions) <= 0:
        raise ValueError(f"fractions must be 3 positive numbers summing to 1, got {tuple(fractions)}")
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(fractions[1] * n)))
    n_test = max(1, int(round(fractions[2] * n)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("split fractions leave no training images")
    shuffled = [ids[i] for i in perm]
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def build_dataloader(
    dataset: CocoDetectionDataset, batch_size: int, *, shuffle: bool, seed: int, num_workers: int = 0
) -> DataLoader:
    """DataLoader with a seeded generator so shuffling is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )


# --------------------------------------------------------------------------- #
# Visualisation helper (used for experiment-tracking images)
# --------------------------------------------------------------------------- #
_PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]


def draw_annotations(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    scores: Sequence[float] | None = None,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of an RGB uint8 image with boxes (and optional masks) drawn on it."""
    canvas = image_rgb.copy()
    if masks is not None:
        for i, m in enumerate(masks):
            color = np.array(_PALETTE[int(labels[i]) % len(_PALETTE)], dtype=np.float32)
            sel = m.astype(bool)
            canvas[sel] = (0.6 * canvas[sel] + 0.4 * color).astype(np.uint8)
    for i, box in enumerate(np.asarray(boxes).reshape(-1, 4)):
        color = _PALETTE[int(labels[i]) % len(_PALETTE)]
        p1, p2 = (int(round(box[0])), int(round(box[1]))), (int(round(box[2])), int(round(box[3])))
        cv2.rectangle(canvas, p1, p2, color, 2)
        text = class_names[int(labels[i])] if int(labels[i]) < len(class_names) else str(int(labels[i]))
        if scores is not None:
            text += f" {float(scores[i]):.2f}"
        cv2.putText(canvas, text, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas
