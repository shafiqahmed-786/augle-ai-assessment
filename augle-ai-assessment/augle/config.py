"""Typed configuration loaded from YAML (``configs/default.yaml``).

Unknown keys are rejected so typos fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class CollectionConfig:
    """Settings for the Playwright image collector."""

    source: str = "wikimedia_commons"
    base_url: str = "https://commons.wikimedia.org"
    queries: list[str] = field(
        default_factory=lambda: [
            "industrial robot arm",
            "KUKA industrial robot",
            "FANUC robot",
            "ABB industrial robot",
            "robotic arm welding",
            "articulated robot manipulator",
            "SCARA robot",
            "robot gripper end effector",
        ]
    )
    target_images: int = 220
    results_per_page: int = 50
    max_pages_per_query: int = 6
    min_long_side: int = 1024
    min_short_side: int = 600
    max_file_bytes: int = 15_000_000
    max_stored_side: int = 2048
    jpeg_quality: int = 95
    allowed_mime: list[str] = field(default_factory=lambda: ["image/jpeg", "image/png", "image/webp"])
    allowed_license_prefixes: list[str] = field(
        default_factory=lambda: ["CC BY ", "CC BY-SA", "CC0", "Public domain", "PD"]
    )
    denied_license_substrings: list[str] = field(default_factory=lambda: ["-NC", "-ND"])
    dedupe_hamming: int = 4
    request_delay_s: float = 1.0
    page_timeout_ms: int = 30_000
    max_retries: int = 3
    headless: bool = True
    out_dir: str = "data/raw"


@dataclass
class AnnotationConfig:
    """Settings for zero-shot annotation (Grounding DINO boxes, optional SAM masks)."""

    detector_model: str = "IDEA-Research/grounding-dino-tiny"
    categories: dict[str, list[str]] = field(
        default_factory=lambda: {
            "robot_arm": ["industrial robot arm", "robotic arm"],
            "gripper": ["robot gripper", "end effector"],
        }
    )
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    nms_iou: float = 0.5
    min_box_area_frac: float = 0.002
    max_box_area_frac: float = 0.98
    max_detections_per_category: int = 8
    use_sam: bool = True
    sam_model: str = "facebook/sam-vit-base"
    min_polygon_area_px: float = 16.0
    polygon_epsilon_px: float = 0.3
    drop_images_without_detections: bool = True
    device: str = "auto"
    cache_dir: str = "data/annotations/cache"
    output_json: str = "data/annotations/instances.json"


@dataclass
class DataConfig:
    """Dataset / preprocessing settings."""

    image_size: int = 320
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 42
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    rotation_deg: float = 10.0
    scale_range: tuple[float, float] = (0.8, 1.2)
    translate_frac: float = 0.1
    shear_deg: float = 5.0
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    brightness: float = 0.2
    contrast: float = 0.2
    min_box_side_px: float = 2.0
    min_visibility: float = 0.3
    refine_boxes_with_masks: bool = True
    load_masks: bool = True


@dataclass
class TrainConfig:
    """Training-loop settings."""

    epochs: int = 60
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 10.0
    patience: int = 12
    min_delta: float = 1e-3
    monitor: str = "val_loss"
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    base_width: int = 32
    num_log_images: int = 4
    score_threshold: float = 0.05
    max_detections: int = 50
    output_dir: str = "artifacts/run"
    tracker: str = "tensorboard"


@dataclass
class AuditConfig:
    """RAG auditor settings."""

    index_dir: str = "artifacts/audit_index"
    store: str = "numpy"
    embedder: str = "hashing"
    hashing_dim: int = 1024
    top_k: int = 6
    max_tool_iterations: int = 5
    small_abs_area_px: float = 1024.0
    small_norm_area: float = 0.001
    min_side_px: float = 8.0
    max_examples: int = 10


@dataclass
class Config:
    """Top-level configuration."""

    collection: CollectionConfig = field(default_factory=CollectionConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


def _coerce(value: Any, annotation: Any) -> Any:
    """Convert YAML lists to tuples where the dataclass field is declared as a tuple."""
    if isinstance(value, list) and "tuple" in str(annotation):
        return tuple(value)
    return value


def dataclass_from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Build ``cls`` from a dict, rejecting unknown keys."""
    data = dict(data or {})
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}; valid keys: {sorted(fields)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        sub_type = fields[name].type
        if isinstance(sub_type, type) and dataclasses.is_dataclass(sub_type):
            kwargs[name] = dataclass_from_dict(sub_type, value)
        else:
            kwargs[name] = _coerce(value, sub_type)
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    """Load a YAML config; ``None`` returns pure defaults."""
    if path is None:
        return Config()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    return Config(
        collection=dataclass_from_dict(CollectionConfig, raw.get("collection")),
        annotation=dataclass_from_dict(AnnotationConfig, raw.get("annotation")),
        data=dataclass_from_dict(DataConfig, raw.get("data")),
        train=dataclass_from_dict(TrainConfig, raw.get("train")),
        audit=dataclass_from_dict(AuditConfig, raw.get("audit")),
    )


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a config dataclass to plain JSON-serialisable types."""
    return _plain(dataclasses.asdict(cfg))


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj
