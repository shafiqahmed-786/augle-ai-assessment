"""SYNTHETIC robot-arm-like scenes with exact masks, for tests and smoke runs only.

These images are procedurally drawn with OpenCV. They are *not* real photographs and any
numbers measured on them (training loss, AP, auditor output) are smoke-test results, never
results of the real assessment dataset. Ground truth is exact by construction (masks are drawn,
polygons/boxes are derived from them), which makes them ideal for geometry and pipeline tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from augle.annotation import Detection, attach_mask, build_coco
from augle.coco import save_json_atomic
from augle.config import AnnotationConfig
from augle.imaging import encode_jpeg

CATEGORIES = ["robot_arm", "gripper"]


def render_scene(rng: np.random.Generator, height: int, width: int) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Draw 1-2 articulated arms (+ a gripper each) and return the BGR image and exact masks."""
    top, bottom = rng.integers(60, 200, size=3), rng.integers(60, 200, size=3)
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    image = ((1 - ramp) * top + ramp * bottom) * np.ones((height, width, 3), np.float32)
    image += rng.normal(0, 4, size=image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    instances: list[tuple[str, np.ndarray]] = []
    short = min(height, width)
    for _ in range(int(rng.integers(1, 3))):
        arm_mask = np.zeros((height, width), np.uint8)
        grip_mask = np.zeros((height, width), np.uint8)
        point = np.array([rng.uniform(0.25, 0.75) * width, rng.uniform(0.6, 0.9) * height])
        angle = rng.uniform(-np.pi * 0.9, -np.pi * 0.1)
        thickness = max(3, int(short * rng.uniform(0.04, 0.07)))
        for _seg in range(int(rng.integers(2, 4))):
            length = short * rng.uniform(0.15, 0.28)
            nxt = point + length * np.array([np.cos(angle), np.sin(angle)])
            nxt = np.clip(nxt, 2, [width - 3, height - 3])
            cv2.line(arm_mask, tuple(int(v) for v in point), tuple(int(v) for v in nxt), 1, thickness)
            cv2.circle(arm_mask, tuple(int(v) for v in nxt), thickness, 1, -1)
            point, angle = nxt, angle + rng.uniform(-0.9, 0.9)
        for side in (-1, 1):  # two gripper prongs at the end effector
            tip = np.clip(
                point + short * 0.07 * np.array([np.cos(angle + side * 0.5), np.sin(angle + side * 0.5)]),
                2,
                [width - 3, height - 3],
            )
            cv2.line(grip_mask, tuple(int(v) for v in point), tuple(int(v) for v in tip), 1, max(2, thickness // 2))
        arm_mask[grip_mask > 0] = 0
        color = tuple(int(c) for c in rng.integers(150, 255, size=3))
        image[arm_mask > 0] = color
        image[grip_mask > 0] = (30, 30, 30)
        for name, mask in (("robot_arm", arm_mask), ("gripper", grip_mask)):
            if mask.sum() >= 40:
                instances.append((name, mask.astype(bool)))
    return image, instances


def build_synthetic_dataset(
    root: str | Path, n_images: int = 24, seed: int = 0, size: tuple[int, int] = (160, 224)
) -> Path:
    """Write ``root/images/*.jpg`` and ``root/instances.json`` (COCO, with polygon masks).

    ``size`` is ``(height, width)``; images alternate between landscape and portrait so that
    letterboxing is exercised. Returns the annotation file path.
    """
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    acfg = AnnotationConfig(min_polygon_area_px=8.0)
    records: list[dict[str, Any]] = []
    detections: dict[str, list[Detection]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for i in range(n_images):
        h, w = size if i % 2 == 0 else (size[1], size[0])
        image, instances = render_scene(rng, h, w)
        dets: list[Detection] = []
        for name, mask in instances:
            ys, xs = np.nonzero(mask)
            det = Detection((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)), 1.0, name)
            if attach_mask(det, mask, acfg):
                dets.append(det)
        if not dets:
            continue
        image_id = f"synthetic{i:04d}"
        file_name = f"images/{image_id}.jpg"
        (root / file_name).write_bytes(encode_jpeg(image, 95))
        records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "license_short": "CC0 (synthetic)",
                "license_url": "",
                "original_url": "",
                "page_url": "",
                "author": "procedural generator",
                "credit": "",
                "title": f"synthetic scene {i}",
            }
        )
        detections[image_id], sizes[image_id] = dets, (w, h)
    coco = build_coco(
        records, detections, sizes, CATEGORIES, description="SYNTHETIC smoke-test dataset (not real data)"
    )
    coco["info"]["contributor"] = "augle.synthetic - procedurally generated, for tests/smoke runs only"
    out = root / "instances.json"
    save_json_atomic(coco, out)
    return out
