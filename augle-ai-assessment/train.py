#!/usr/bin/env python3
"""Phase 2: custom CNN object detector trained from scratch with a manual PyTorch loop.

Task definition
---------------
**Single-stage, anchor-free object detection** (CenterNet formulation) on the COCO dataset
produced in Phase 1. For every image the network predicts, at stride 4:

* ``heatmap`` ``(C, H/4, W/4)``  - per-class object-centre likelihood (logits),
* ``wh``      ``(2, H/4, W/4)``  - box width/height in feature-grid units,
* ``offset``  ``(2, H/4, W/4)``  - sub-cell centre refinement.

Target assignment: each ground-truth box writes a Gaussian (radius from the CenterNet IoU rule)
into the heat-map channel of its class at its centre cell; its size and offset are regressed only
at that centre cell. Loss: penalty-reduced focal loss on the heat-map (handles the huge
background/foreground imbalance and empty images) + L1 on ``wh`` and ``offset``.
Decoding: 3x3 max-pool peak extraction (acts as NMS) + top-K.

Labels are zero-shot pseudo-labels from Phase 1, so every metric reported here measures
agreement with Grounding DINO(+SAM) output, not with human ground truth. Masks are carried through
augmentation (and used to tighten augmented boxes) but are not a training target.

Usage::

    python train.py --coco data/annotations/instances.json --image-root data/raw \
        --output-dir artifacts/run1 --epochs 60 --tracker tensorboard
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from augle.coco import iou_matrix_xyxy, load_coco, save_json_atomic
from augle.config import Config, DataConfig, TrainConfig, config_to_dict, load_config
from augle.utils import environment_info, file_sha256, load_dotenv, setup_logging
from dataset import (
    CocoDetectionDataset,
    build_dataloader,
    denormalize_to_uint8,
    draw_annotations,
    make_splits,
)

LOGGER = logging.getLogger("train")
STRIDE = 4
MAX_OBJECTS = 64
METRIC_MODES = {"val_loss": "min", "val_ap50": "max", "val_ap": "max"}


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU+CUDA) and request deterministic kernels."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    """``auto`` -> CUDA when available else CPU; explicit strings are validated."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu or auto")
    return device


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Sequential):
    """Conv2d (no bias) -> BatchNorm -> SiLU. Padding keeps spatial size for odd kernels."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, dilation: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, padding=dilation * (k // 2), dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Two 3x3 convs with an identity (or strided 1x1 projection) skip connection."""

    def __init__(self, cin: int, cout: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, stride, dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.shortcut: nn.Module = (
            nn.Identity()
            if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class AugleNet(nn.Module):
    """Custom residual encoder + top-down (FPN-style) neck + three CenterNet heads.

    Encoder (input ``S x S``, ``S`` divisible by 32): 5x5/2 stem, then four stages of residual
    blocks with channels ``base * (1.5, 3, 5, 8)``, depths ``(1, 2, 3, 2)``, each stage opening
    with a stride-2 block (strides 4, 8, 16, 32). The last stage uses dilation 2 to enlarge the
    receptive field cheaply, which helps with long, thin structures such as robot arms.
    The neck merges all four stages into one stride-4 map; heads predict heat-map / wh / offset.
    """

    def __init__(
        self, num_classes: int, base_width: int = 32, depths: tuple[int, ...] = (1, 2, 3, 2), neck_width: int = 64
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError("depths must have 4 entries")
        self.num_classes = num_classes
        widths = [int(base_width * m) for m in (1.5, 3, 5, 8)]
        self.stem = ConvBNAct(3, base_width, k=5, stride=2)
        stages: list[nn.Module] = []
        cin = base_width
        for i, (cout, depth) in enumerate(zip(widths, depths)):
            dilation = 2 if i == 3 else 1
            blocks = [ResidualBlock(cin, cout, stride=2, dilation=1)]
            blocks += [ResidualBlock(cout, cout, 1, dilation) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.lateral = nn.ModuleList([nn.Conv2d(c, neck_width, 1) for c in widths])
        self.smooth = ConvBNAct(neck_width, neck_width, 3)
        self.head_hm = self._head(neck_width, num_classes)
        self.head_wh = self._head(neck_width, 2)
        self.head_off = self._head(neck_width, 2)
        self._init_weights()

    @staticmethod
    def _head(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(ConvBNAct(cin, cin, 3), nn.Conv2d(cin, cout, 1))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for head in (self.head_hm, self.head_wh, self.head_off):  # small output weights -> tame initial logits
            nn.init.normal_(head[-1].weight, std=0.01)
        nn.init.constant_(self.head_hm[-1].bias, -2.19)  # prior p=0.1 -> stable focal-loss start

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(f"input size must be divisible by 32, got {tuple(x.shape[-2:])}")
        feats: list[torch.Tensor] = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        top = self.lateral[3](feats[3])
        for i in (2, 1, 0):  # top-down pathway with nearest-neighbour upsampling
            top = self.lateral[i](feats[i]) + F.interpolate(top, size=feats[i].shape[-2:], mode="nearest")
        fused = self.smooth(top)
        return {"heatmap": self.head_hm(fused), "wh": self.head_wh(fused), "offset": self.head_off(fused)}


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Targets, loss, decoding
# --------------------------------------------------------------------------- #
def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    """CenterNet rule: largest radius such that a box shifted by it still has IoU >= ``min_overlap``."""
    height, width = det_size
    b1, c1 = height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1**2 - 4 * c1, 0.0))) / 2
    b2, c2 = 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2**2 - 16 * c2, 0.0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Splat a 2-D Gaussian (peak 1.0) into ``heatmap`` with element-wise max (in place)."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    gauss = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma * sigma))
    x, y = center
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    region = heatmap[y - top : y + bottom, x - left : x + right]
    patch = gauss[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, patch, out=region)


def encode_targets(
    boxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    num_classes: int,
    out_h: int,
    out_w: int,
    stride: int = STRIDE,
    max_objects: int = MAX_OBJECTS,
) -> dict[str, torch.Tensor]:
    """Build CenterNet training targets for a batch (CPU tensors).

    Returns ``heatmap (B,C,H,W)``, ``wh (B,K,2)`` in grid units, ``offset (B,K,2)``,
    ``index (B,K)`` (flat ``y*W+x`` of the centre cell) and ``mask (B,K)``.
    """
    bsz = len(boxes)
    heat = np.zeros((bsz, num_classes, out_h, out_w), dtype=np.float32)
    wh = np.zeros((bsz, max_objects, 2), dtype=np.float32)
    off = np.zeros((bsz, max_objects, 2), dtype=np.float32)
    ind = np.zeros((bsz, max_objects), dtype=np.int64)
    msk = np.zeros((bsz, max_objects), dtype=np.float32)
    for b in range(bsz):
        k = 0
        for box, lab in zip(boxes[b].numpy(), labels[b].numpy()):
            if k >= max_objects:
                break
            bw, bh = (box[2] - box[0]) / stride, (box[3] - box[1]) / stride
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (box[0] + box[2]) / 2 / stride, (box[1] + box[3]) / 2 / stride
            ci = (min(max(int(cx), 0), out_w - 1), min(max(int(cy), 0), out_h - 1))
            radius = max(0, int(gaussian_radius((math.ceil(bh), math.ceil(bw)))))
            draw_gaussian(heat[b, int(lab)], ci, radius)
            wh[b, k] = (bw, bh)
            off[b, k] = (cx - ci[0], cy - ci[1])
            ind[b, k] = ci[1] * out_w + ci[0]
            msk[b, k] = 1.0
            k += 1
    return {
        "heatmap": torch.from_numpy(heat),
        "wh": torch.from_numpy(wh),
        "offset": torch.from_numpy(off),
        "index": torch.from_numpy(ind),
        "mask": torch.from_numpy(msk),
    }


def focal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CenterNet), computed from logits for stability."""
    prob = torch.sigmoid(logits)
    pos = gt.eq(1).float()
    neg_weight = torch.pow(1 - gt, 4)
    pos_loss = F.logsigmoid(logits) * torch.pow(1 - prob, 2) * pos
    neg_loss = F.logsigmoid(-logits) * torch.pow(prob, 2) * neg_weight * (1 - pos)
    num_pos = pos.sum()
    total = pos_loss.sum() + neg_loss.sum()
    return -total / num_pos.clamp(min=1.0)


def gather_at_index(feat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``feat (B,C,H,W)``, ``index (B,K)`` flat cell ids -> ``(B,K,C)``."""
    bsz, ch = feat.shape[:2]
    flat = feat.view(bsz, ch, -1).permute(0, 2, 1)
    return flat.gather(1, index.unsqueeze(2).expand(-1, -1, ch))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 between predictions gathered at centre cells and targets, averaged over valid objects."""
    gathered = gather_at_index(pred, index)
    m = mask.unsqueeze(2)
    return (torch.abs(gathered - target) * m).sum() / (m.sum() * gathered.shape[2] + 1e-4)


class CenterNetLoss(nn.Module):
    """``total = focal(heatmap) + w_wh * L1(wh) + w_off * L1(offset)``."""

    def __init__(self, w_wh: float = 0.1, w_off: float = 1.0) -> None:
        super().__init__()
        self.w_wh, self.w_off = w_wh, w_off

    def forward(self, out: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hm = focal_loss(out["heatmap"], tgt["heatmap"])
        wh = masked_l1(out["wh"], tgt["wh"], tgt["index"], tgt["mask"])
        off = masked_l1(out["offset"], tgt["offset"], tgt["index"], tgt["mask"])
        return {
            "total": hm + self.w_wh * wh + self.w_off * off,
            "hm": hm.detach(),
            "wh": wh.detach(),
            "off": off.detach(),
        }


@torch.no_grad()
def decode_predictions(
    out: dict[str, torch.Tensor],
    image_size: int,
    score_threshold: float = 0.05,
    max_detections: int = 50,
    stride: int = STRIDE,
) -> list[dict[str, np.ndarray]]:
    """Heat-map peaks -> per-image dicts with ``boxes`` (xyxy px), ``scores`` and ``labels``."""
    heat = torch.sigmoid(out["heatmap"])
    peaks = (F.max_pool2d(heat, 3, 1, 1) == heat).float() * heat  # 3x3 max-pool = NMS
    bsz, ncls, h, w = peaks.shape
    k = min(max_detections, ncls * h * w)
    scores, inds = torch.topk(peaks.view(bsz, -1), k)
    cls = torch.div(inds, h * w, rounding_mode="floor")
    pix = inds % (h * w)
    ys, xs = torch.div(pix, w, rounding_mode="floor").float(), (pix % w).float()
    wh = gather_at_index(out["wh"], pix).clamp(min=0)
    off = gather_at_index(out["offset"], pix)
    cx, cy = (xs + off[..., 0]) * stride, (ys + off[..., 1]) * stride
    bw, bh = wh[..., 0] * stride, wh[..., 1] * stride
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1).clamp(0, image_size)
    results = []
    for b in range(bsz):
        keep = scores[b] >= score_threshold
        results.append(
            {
                "boxes": boxes[b][keep].cpu().numpy(),
                "scores": scores[b][keep].cpu().numpy(),
                "labels": cls[b][keep].cpu().numpy(),
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Metrics (COCO-style AP, implemented from scratch and cross-checked against pycocotools in tests)
# --------------------------------------------------------------------------- #
def average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP for one class at one IoU threshold (COCO definition)."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = is_tp[order].astype(np.float64)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tp_c / n_gt
    precision = tp_c / (tp_c + fp_c)
    for i in range(len(precision) - 1, 0, -1):  # monotone precision envelope
        precision[i - 1] = max(precision[i - 1], precision[i])
    thresholds = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, thresholds, side="left")
    q = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(q.mean())


def evaluate_detections(
    preds: list[dict[str, np.ndarray]],
    gts: list[dict[str, np.ndarray]],
    num_classes: int,
    iou_thresholds: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Mean AP over classes (those with ground truth) at IoU 0.5 and averaged over 0.5:0.95.

    ``per_class_ap50`` is keyed by class name when ``class_names`` is given, else by class index.
    """
    thresholds = np.arange(0.5, 0.951, 0.05) if iou_thresholds is None else np.asarray(iou_thresholds)
    per_class_ap: dict[int, list[float]] = {}
    for c in range(num_classes):
        n_gt = int(sum((g["labels"] == c).sum() for g in gts))
        aps = []
        for thr in thresholds:
            scores_all: list[float] = []
            tp_all: list[int] = []
            for pred, gt in zip(preds, gts):
                p_sel, g_sel = pred["labels"] == c, gt["labels"] == c
                p_boxes, p_scores = pred["boxes"][p_sel], pred["scores"][p_sel]
                g_boxes = gt["boxes"][g_sel]
                order = np.argsort(-p_scores, kind="stable")
                p_boxes, p_scores = p_boxes[order], p_scores[order]
                ious = iou_matrix_xyxy(p_boxes, g_boxes)
                taken = np.zeros(len(g_boxes), dtype=bool)
                for i in range(len(p_boxes)):
                    best, best_j = thr, -1
                    for j in range(len(g_boxes)):
                        if not taken[j] and ious[i, j] >= best:
                            best, best_j = ious[i, j], j
                    if best_j >= 0:
                        taken[best_j] = True
                    scores_all.append(float(p_scores[i]))
                    tp_all.append(int(best_j >= 0))
            aps.append(average_precision(np.array(scores_all), np.array(tp_all), n_gt))
        per_class_ap[c] = aps
    valid = [c for c, aps in per_class_ap.items() if not math.isnan(aps[0])]
    ap50 = float(np.mean([per_class_ap[c][0] for c in valid])) if valid else 0.0
    ap = float(np.mean([np.mean(per_class_ap[c]) for c in valid])) if valid else 0.0
    label = (lambda c: class_names[c]) if class_names is not None else (lambda c: c)
    return {
        "ap50": ap50,
        "ap": ap,
        "per_class_ap50": {
            label(c): (None if math.isnan(per_class_ap[c][0]) else float(per_class_ap[c][0])) for c in per_class_ap
        },
    }


# --------------------------------------------------------------------------- #
# Early stopping, optimiser, scheduler
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Stop when the monitored value has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best: float | None = None
        self.best_epoch = 0
        self.num_bad = 0
        self.should_stop = False

    def step(self, value: float, epoch: int) -> bool:
        """Record an epoch value; returns True if this epoch is a new best."""
        improved = False
        if math.isfinite(value):
            if self.best is None:
                improved = True
            elif self.mode == "min":
                improved = value < self.best - self.min_delta
            else:
                improved = value > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.num_bad = float(value), epoch, 0
        else:
            self.num_bad += 1
            self.should_stop = self.num_bad >= self.patience
        return improved

    def state_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "num_bad": self.num_bad,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best, self.best_epoch = state["best"], state["best_epoch"]
        self.num_bad, self.should_stop = state["num_bad"], state["should_stop"]


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> Optimizer:
    """AdamW with weight decay applied to conv/linear weights only (not BatchNorm / bias)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)
    return AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def build_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int, eta_min_frac: float = 0.01) -> LRScheduler:
    """Linear warm-up followed by cosine decay; call ``step()`` once per epoch."""
    base_lr = optimizer.param_groups[0]["lr"]
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=base_lr * eta_min_frac)
    if warmup_epochs <= 0:
        return cosine
    warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    return SequentialLR(optimizer, [warm, cosine], milestones=[warmup_epochs])


# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #
class Tracker(Protocol):
    """Minimal tracking interface implemented by TensorBoard / W&B / no-op backends."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None: ...
    def log_images(self, tag: str, step: int, images: list[np.ndarray]) -> None: ...
    def log_config(self, config: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class NullTracker:
    """Tracking disabled."""

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        pass

    def log_images(self, tag: str, step: int, images: list[np.ndarray]) -> None:
        pass

    def log_config(self, config: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardTracker:
    """TensorBoard via ``torch.utils.tensorboard``."""

    def __init__(self, log_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:  # broken tensorflow stub or missing tensorboard
            from tensorboardX import SummaryWriter  # type: ignore[no-redef]

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        for key, value in values.items():
            if value is not None and math.isfinite(value):
                self.writer.add_scalar(key, value, step)

    def log_images(self, tag: str, step: int, images: list[np.ndarray]) -> None:
        for i, img in enumerate(images):
            self.writer.add_image(f"{tag}/{i}", img, step, dataformats="HWC")

    def log_config(self, config: dict[str, Any]) -> None:
        self.writer.add_text("config", "```json\n" + json.dumps(config, indent=2, default=str) + "\n```", 0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class WandbTracker:
    """Weights & Biases (``WANDB_MODE=offline`` works without network)."""

    def __init__(self, run_name: str, config: dict[str, Any]) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "augle-ai-assessment"), name=run_name, config=config
        )

    def log_scalars(self, step: int, values: dict[str, float]) -> None:
        self.wandb.log({k: v for k, v in values.items() if v is not None and math.isfinite(v)}, step=step)

    def log_images(self, tag: str, step: int, images: list[np.ndarray]) -> None:
        self.wandb.log({tag: [self.wandb.Image(img) for img in images]}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self.run.config.update(config, allow_val_change=True)

    def close(self) -> None:
        self.run.finish()


def make_tracker(kind: str, run_dir: Path, run_name: str, config: dict[str, Any]) -> Tracker:
    """Create a tracker by name (``tensorboard`` | ``wandb`` | ``none``)."""
    if kind == "none":
        return NullTracker()
    if kind == "tensorboard":
        tracker: Tracker = TensorBoardTracker(run_dir / "tensorboard")
    elif kind == "wandb":
        tracker = WandbTracker(run_name, config)
    else:
        raise ValueError(f"unknown tracker '{kind}' (use tensorboard | wandb | none)")
    tracker.log_config(config)
    return tracker


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def _targets_to_device(tgt: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in tgt.items()}


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    loss_fn: CenterNetLoss,
    optimizer: Optimizer,
    device: torch.device,
    num_classes: int,
    grad_clip: float,
) -> dict[str, float]:
    """One pass over the training loader: forward, loss, backward, clip, step (manual loop)."""
    model.train()
    sums = {"loss": 0.0, "hm": 0.0, "wh": 0.0, "off": 0.0, "grad_norm": 0.0}
    n_samples, n_batches = 0, 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        out_h, out_w = images.shape[-2] // STRIDE, images.shape[-1] // STRIDE
        targets = _targets_to_device(encode_targets(batch["boxes"], batch["labels"], num_classes, out_h, out_w), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = loss_fn(outputs, targets)
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss {loss.item()} (check learning rate / data)")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bsz = images.shape[0]
        n_samples += bsz
        n_batches += 1
        sums["loss"] += loss.item() * bsz
        for key in ("hm", "wh", "off"):
            sums[key] += losses[key].item() * bsz
        sums["grad_norm"] += float(grad_norm)
    if n_samples == 0:
        raise ValueError("training loader produced no samples")
    return {
        "loss": sums["loss"] / n_samples,
        "hm": sums["hm"] / n_samples,
        "wh": sums["wh"] / n_samples,
        "off": sums["off"] / n_samples,
        "grad_norm": sums["grad_norm"] / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    loss_fn: CenterNetLoss,
    device: torch.device,
    num_classes: int,
    image_size: int,
    score_threshold: float,
    max_detections: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation loss + COCO-style AP. Never updates weights."""
    model.eval()
    sums = {"loss": 0.0, "hm": 0.0, "wh": 0.0, "off": 0.0}
    n_samples = 0
    preds: list[dict[str, np.ndarray]] = []
    gts: list[dict[str, np.ndarray]] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        out_h, out_w = images.shape[-2] // STRIDE, images.shape[-1] // STRIDE
        targets = _targets_to_device(encode_targets(batch["boxes"], batch["labels"], num_classes, out_h, out_w), device)
        outputs = model(images)
        losses = loss_fn(outputs, targets)
        bsz = images.shape[0]
        n_samples += bsz
        sums["loss"] += losses["total"].item() * bsz
        for key in ("hm", "wh", "off"):
            sums[key] += losses[key].item() * bsz
        preds.extend(decode_predictions(outputs, image_size, score_threshold, max_detections))
        gts.extend({"boxes": b.numpy(), "labels": lab.numpy()} for b, lab in zip(batch["boxes"], batch["labels"]))
    if n_samples == 0:
        raise ValueError("evaluation loader produced no samples")
    result = {k: v / n_samples for k, v in sums.items()}
    result.update(evaluate_detections(preds, gts, num_classes, class_names=class_names))
    return result


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    early_stopping: EarlyStopping,
    meta: dict[str, Any],
) -> None:
    """Atomically save a checkpoint containing only tensors and plain Python types."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "early_stopping": early_stopping.state_dict(),
        "meta": meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint with ``weights_only=True`` (no arbitrary pickle execution)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# --------------------------------------------------------------------------- #
# Image logging
# --------------------------------------------------------------------------- #
def make_montage(images: list[np.ndarray]) -> np.ndarray:
    """Horizontally concatenate equally sized RGB images."""
    return np.concatenate(images, axis=1)


@torch.no_grad()
def collect_epoch_images(
    train_ds: CocoDetectionDataset,
    val_ds: CocoDetectionDataset,
    model: nn.Module,
    device: torch.device,
    indices: tuple[list[int], list[int]],
    cfg: TrainConfig,
    data_cfg: DataConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Augmented training samples (with GT boxes/masks) and validation predictions for logging."""
    aug: list[np.ndarray] = []
    for idx in indices[0]:
        item = train_ds[idx]
        img = denormalize_to_uint8(item["image"], data_cfg.mean, data_cfg.std)
        masks = None if item["masks"] is None else item["masks"].numpy()
        aug.append(
            draw_annotations(img, item["boxes"].numpy(), item["labels"].numpy(), train_ds.class_names, masks=masks)
        )
    model.eval()
    pred_imgs: list[np.ndarray] = []
    items = [val_ds[i] for i in indices[1]]
    if items:
        batch = torch.stack([it["image"] for it in items]).to(device)
        dets = decode_predictions(model(batch), data_cfg.image_size, 0.3, 20)
        for it, det in zip(items, dets):
            img = denormalize_to_uint8(it["image"], data_cfg.mean, data_cfg.std)
            pred_imgs.append(draw_annotations(img, det["boxes"], det["labels"], val_ds.class_names, det["scores"]))
    return aug, pred_imgs


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def json_safe(obj: Any) -> Any:
    """Make nested structures JSON-serialisable (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Paths and headline numbers of a finished run."""

    run_dir: Path
    metrics_path: Path
    best_epoch: int
    best_value: float | None
    stopped_early: bool
    epochs_run: int


def fit(
    cfg: Config, coco_path: str | Path, image_root: str | Path, run_dir: str | Path, resume: str | Path | None = None
) -> RunResult:
    """Train, validate, checkpoint and track; write ``metrics.json`` for the auditor."""
    tcfg, dcfg = cfg.train, cfg.data
    if tcfg.monitor not in METRIC_MODES:
        raise ValueError(f"monitor must be one of {sorted(METRIC_MODES)}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    seed_everything(tcfg.seed)
    device = select_device(tcfg.device)
    LOGGER.info("device=%s seed=%d", device, tcfg.seed)

    coco = load_coco(coco_path)
    splits = make_splits([im["id"] for im in coco["images"]], dcfg.split_fractions, dcfg.split_seed)
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    save_json_atomic(
        {k: {"image_ids": v, "file_names": [id_to_file[i] for i in v]} for k, v in splits.items()},
        run_dir / "splits.json",
    )
    LOGGER.info("split sizes: %s", {k: len(v) for k, v in splits.items()})

    common = {"cfg": dcfg, "seed": tcfg.seed}
    train_ds = CocoDetectionDataset(coco_path, image_root, image_ids=splits["train"], train=True, **common)
    val_ds = CocoDetectionDataset(coco_path, image_root, image_ids=splits["val"], train=False, **common)
    test_ds = CocoDetectionDataset(coco_path, image_root, image_ids=splits["test"], train=False, **common)
    train_loader = build_dataloader(
        train_ds, tcfg.batch_size, shuffle=True, seed=tcfg.seed, num_workers=tcfg.num_workers
    )
    val_loader = build_dataloader(val_ds, tcfg.batch_size, shuffle=False, seed=tcfg.seed, num_workers=tcfg.num_workers)
    test_loader = build_dataloader(
        test_ds, tcfg.batch_size, shuffle=False, seed=tcfg.seed, num_workers=tcfg.num_workers
    )

    num_classes = train_ds.num_classes
    model = AugleNet(num_classes, tcfg.base_width).to(device)
    n_params = count_parameters(model)
    LOGGER.info("AugleNet: %d trainable parameters, %d classes %s", n_params, num_classes, train_ds.class_names)
    loss_fn = CenterNetLoss()
    optimizer = build_optimizer(model, tcfg.lr, tcfg.weight_decay)
    scheduler = build_scheduler(optimizer, tcfg.epochs, tcfg.warmup_epochs)
    stopper = EarlyStopping(tcfg.patience, tcfg.min_delta, METRIC_MODES[tcfg.monitor])

    config_dict = config_to_dict(cfg)
    run_name = run_dir.name
    tracker = make_tracker(tcfg.tracker, run_dir, run_name, config_dict)
    history: list[dict[str, Any]] = []
    history_path = run_dir / "history.jsonl"
    start_epoch = 1
    if resume:
        ckpt = load_checkpoint(Path(resume), model, optimizer, scheduler, device)
        stopper.load_state_dict(ckpt["early_stopping"])
        start_epoch = int(ckpt["epoch"]) + 1
        if history_path.is_file():
            history = [
                json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            history = [h for h in history if h["epoch"] < start_epoch]
        LOGGER.info("resumed from %s at epoch %d", resume, start_epoch)
    history_path.write_text("".join(json.dumps(json_safe(h)) + "\n" for h in history), encoding="utf-8")

    log_rng = np.random.default_rng(tcfg.seed)
    log_train_idx = sorted(
        log_rng.choice(len(train_ds), size=min(tcfg.num_log_images, len(train_ds)), replace=False).tolist()
    )
    log_val_idx = list(range(min(tcfg.num_log_images, len(val_ds))))
    meta = {
        "class_names": train_ds.class_names,
        "num_classes": num_classes,
        "image_size": dcfg.image_size,
        "base_width": tcfg.base_width,
        "monitor": tcfg.monitor,
        "config": config_dict,
    }
    mean_std_note = {"mean": list(dcfg.mean), "std": list(dcfg.std)}
    stopped_early, epoch = False, start_epoch - 1
    try:
        for epoch in range(start_epoch, tcfg.epochs + 1):
            t0 = time.time()
            train_ds.set_epoch(epoch)
            lr = optimizer.param_groups[0]["lr"]
            tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device, num_classes, tcfg.grad_clip)
            va = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                num_classes,
                dcfg.image_size,
                tcfg.score_threshold,
                tcfg.max_detections,
                train_ds.class_names,
            )
            scheduler.step()
            record = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": tr["loss"],
                "train_hm_loss": tr["hm"],
                "train_wh_loss": tr["wh"],
                "train_off_loss": tr["off"],
                "grad_norm": tr["grad_norm"],
                "val_loss": va["loss"],
                "val_hm_loss": va["hm"],
                "val_wh_loss": va["wh"],
                "val_off_loss": va["off"],
                "val_ap50": va["ap50"],
                "val_ap": va["ap"],
                "val_ap50_per_class": va["per_class_ap50"],
                "epoch_time_s": time.time() - t0,
            }
            improved = stopper.step(record[tcfg.monitor], epoch)
            record["is_best"] = bool(improved)
            history.append(record)
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(json_safe(record)) + "\n")
            scalars = {k: v for k, v in record.items() if isinstance(v, (int, float)) and k != "epoch"}
            scalars.update({f"val_ap50_{name}": v for name, v in record["val_ap50_per_class"].items() if v is not None})
            tracker.log_scalars(epoch, scalars)
            aug_imgs, pred_imgs = collect_epoch_images(
                train_ds, val_ds, model, device, (log_train_idx, log_val_idx), tcfg, dcfg
            )
            if aug_imgs:
                tracker.log_images("augmented_train_samples", epoch, aug_imgs)
                cv2.imwrite(
                    str(run_dir / "samples" / f"epoch_{epoch:03d}_augmented.jpg"),
                    cv2.cvtColor(make_montage(aug_imgs), cv2.COLOR_RGB2BGR),
                )
            if pred_imgs:
                tracker.log_images("val_predictions", epoch, pred_imgs)
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, stopper, meta)
            if improved:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, stopper, meta)
            LOGGER.info(
                "epoch %3d/%d lr=%.2e train_loss=%.4f val_loss=%.4f val_AP50=%.3f val_AP=%.3f %s",
                epoch,
                tcfg.epochs,
                lr,
                tr["loss"],
                va["loss"],
                va["ap50"],
                va["ap"],
                "* best" if improved else "",
            )
            if stopper.should_stop:
                stopped_early = True
                LOGGER.info("early stopping: no improvement in %s for %d epochs", tcfg.monitor, tcfg.patience)
                break
    finally:
        tracker.close()

    test_result: dict[str, Any] | None = None
    best_path = run_dir / "best.pt"
    if best_path.is_file():
        load_checkpoint(best_path, model, device=device)
        test_result = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            num_classes,
            dcfg.image_size,
            tcfg.score_threshold,
            tcfg.max_detections,
            train_ds.class_names,
        )
        LOGGER.info(
            "test (best checkpoint, epoch %d): loss=%.4f AP50=%.3f AP=%.3f",
            stopper.best_epoch,
            test_result["loss"],
            test_result["ap50"],
            test_result["ap"],
        )

    metrics = {
        "schema_version": 1,
        "run_id": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "single-stage anchor-free object detection (CenterNet-style: heatmap + wh + offset heads)",
        "metric_notes": (
            "AP is COCO-style (101-point) in the letterboxed network-input frame. Labels are zero-shot "
            "pseudo-labels, so AP measures agreement with Grounding DINO/SAM output, not human ground truth."
        ),
        "config": config_dict,
        "normalisation": mean_std_note,
        "environment": {**environment_info(), "device": str(device)},
        "model": {"name": "AugleNet", "trainable_parameters": n_params, "classes": train_ds.class_names},
        "dataset": {
            "coco_path": str(coco_path),
            "coco_sha256": file_sha256(coco_path),
            "num_images": len(coco["images"]),
            "num_annotations": len(coco["annotations"]),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "split_seed": dcfg.split_seed,
        },
        "history": history,
        "best": {
            "epoch": stopper.best_epoch,
            "monitor": tcfg.monitor,
            "mode": METRIC_MODES[tcfg.monitor],
            "value": stopper.best,
            "checkpoint": "best.pt" if best_path.is_file() else None,
        },
        "early_stopping": {
            "enabled": True,
            "patience": tcfg.patience,
            "min_delta": tcfg.min_delta,
            "triggered": stopped_early,
            "stopped_epoch": epoch if stopped_early else None,
        },
        "epochs_requested": tcfg.epochs,
        "epochs_completed": len(history),
        "test": test_result,
    }
    metrics_path = run_dir / "metrics.json"
    save_json_atomic(json_safe(metrics), metrics_path)
    LOGGER.info("wrote %s", metrics_path)
    return RunResult(run_dir, metrics_path, stopper.best_epoch, stopper.best, stopped_early, len(history))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """CLI definition (flags override the YAML config)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--coco", default="data/annotations/instances.json")
    p.add_argument("--image-root", default="data/raw")
    p.add_argument("--output-dir")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--tracker", choices=["tensorboard", "wandb", "none"])
    p.add_argument("--num-workers", type=int)
    p.add_argument("--resume", help="path to last.pt to continue training")
    p.add_argument("--log-level", default="INFO")
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Copy of ``cfg`` with CLI overrides applied."""
    cfg = copy.deepcopy(cfg)
    for name in ("epochs", "batch_size", "lr", "seed", "device", "tracker", "num_workers", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.train, name, value)
    if args.image_size is not None:
        cfg.data.image_size = args.image_size
    return cfg


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = apply_overrides(load_config(args.config), args)
    run_dir = Path(cfg.train.output_dir)
    setup_logging(args.log_level, run_dir / "train.log")
    result = fit(cfg, args.coco, args.image_root, run_dir, args.resume)
    LOGGER.info(
        "done: best epoch %d (%s=%s), early stopped=%s, metrics: %s",
        result.best_epoch,
        cfg.train.monitor,
        result.best_value,
        result.stopped_early,
        result.metrics_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
