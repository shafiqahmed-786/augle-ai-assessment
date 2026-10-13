"""Deterministic audit tools and a schema-validated registry for LLM tool calling.

Every number the auditor reports comes from these functions, which read the real COCO JSON and
the real ``metrics.json`` written by ``train.py``. The LLM never computes statistics itself.

Security notes
--------------
* Tools take *analysis parameters only*. File locations are bound in :class:`AuditContext`, so a
  model (or text injected into retrieved documents) cannot make a tool read arbitrary paths.
* Arguments are validated against a JSON schema (``additionalProperties: false``) before the
  function runs; failures are returned to the model as structured errors, never raised.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from augle.coco import load_coco


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class AuditContext:
    """Files and default thresholds the tools operate on."""

    coco_path: Path
    metrics_path: Path | None = None
    small_abs_area_px: float = 1024.0
    small_norm_area: float = 0.001
    min_side_px: float = 8.0
    max_examples: int = 10
    _coco_cache: tuple[float, dict[str, Any]] | None = field(default=None, repr=False)

    def coco(self) -> dict[str, Any]:
        """Load the COCO JSON (cached until the file's modification time changes)."""
        mtime = Path(self.coco_path).stat().st_mtime if Path(self.coco_path).exists() else -1.0
        if self._coco_cache is None or self._coco_cache[0] != mtime:
            self._coco_cache = (mtime, load_coco(self.coco_path))
        return self._coco_cache[1]


def _pct(values: np.ndarray, qs: tuple[float, ...] = (0, 5, 25, 50, 75, 95, 100)) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {f"p{int(q)}": float(np.percentile(values, q)) for q in qs} | {"mean": float(values.mean())}


def _category_names(coco: dict[str, Any]) -> dict[int, str]:
    return {c["id"]: c["name"] for c in coco.get("categories", [])}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def get_dataset_statistics(ctx: AuditContext, args: dict[str, Any]) -> dict[str, Any]:
    """Counts of images, annotations, categories and annotations per category (+ size/licence summaries)."""
    coco = ctx.coco()
    names = _category_names(coco)
    images, anns = coco.get("images", []), coco.get("annotations", [])
    per_cat = {name: 0 for name in names.values()}
    per_image: dict[int, int] = {im["id"]: 0 for im in images}
    for a in anns:
        per_cat[names.get(a["category_id"], f"unknown_{a['category_id']}")] = (
            per_cat.get(names.get(a["category_id"], f"unknown_{a['category_id']}"), 0) + 1
        )
        if a["image_id"] in per_image:
            per_image[a["image_id"]] += 1
    counts = np.array(list(per_image.values()), dtype=float)
    widths = np.array([im["width"] for im in images], dtype=float)
    heights = np.array([im["height"] for im in images], dtype=float)
    licenses = {lic["id"]: lic.get("name", "unknown") for lic in coco.get("licenses", [])}
    by_license: dict[str, int] = {}
    for im in images:
        key = licenses.get(im.get("license"), "unspecified")
        by_license[key] = by_license.get(key, 0) + 1
    return {
        "num_images": len(images),
        "num_annotations": len(anns),
        "num_categories": len(names),
        "annotations_per_category": per_cat,
        "images_without_annotations": int((counts == 0).sum()) if len(counts) else 0,
        "annotations_per_image": (
            {
                "mean": float(counts.mean()),
                "median": float(np.median(counts)),
                "min": int(counts.min()),
                "max": int(counts.max()),
            }
            if len(counts)
            else {}
        ),
        "annotations_with_segmentation": int(sum(1 for a in anns if a.get("segmentation"))),
        "image_size_px": (
            {
                "width_min": int(widths.min()),
                "width_max": int(widths.max()),
                "height_min": int(heights.min()),
                "height_max": int(heights.max()),
            }
            if len(images)
            else {}
        ),
        "images_per_license": by_license,
    }


def analyze_bbox_areas(ctx: AuditContext, args: dict[str, Any]) -> dict[str, Any]:
    """Bounding-box areas, normalised areas, suspiciously small boxes and invalid geometry.

    A box is *suspiciously small* when ANY criterion holds: area < ``abs_area_threshold_px`` px^2,
    area / image area < ``norm_area_threshold``, or min(width, height) < ``min_side_px``.
    """
    abs_thr = float(args.get("abs_area_threshold_px", ctx.small_abs_area_px))
    norm_thr = float(args.get("norm_area_threshold", ctx.small_norm_area))
    side_thr = float(args.get("min_side_px", ctx.min_side_px))
    max_examples = int(args.get("max_examples", ctx.max_examples))
    coco = ctx.coco()
    names = _category_names(coco)
    img = {im["id"]: im for im in coco.get("images", [])}
    invalid: dict[str, list[int]] = {
        "non_positive_size": [],
        "non_finite": [],
        "out_of_bounds": [],
        "unknown_image": [],
    }
    rows: list[dict[str, Any]] = []
    for a in coco.get("annotations", []):
        bbox = a.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, int | float) and math.isfinite(v) for v in bbox)
        ):
            invalid["non_finite"].append(a["id"])
            continue
        x, y, w, h = (float(v) for v in bbox)
        im = img.get(a["image_id"])
        if im is None:
            invalid["unknown_image"].append(a["id"])
            continue
        if w <= 0 or h <= 0:
            invalid["non_positive_size"].append(a["id"])
            continue
        if x < -1 or y < -1 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
            invalid["out_of_bounds"].append(a["id"])
        area = w * h
        norm = area / float(im["width"] * im["height"])
        flags = []
        if area < abs_thr:
            flags.append("area_px_below_threshold")
        if norm < norm_thr:
            flags.append("normalized_area_below_threshold")
        if min(w, h) < side_thr:
            flags.append("min_side_below_threshold")
        rows.append(
            {
                "annotation_id": a["id"],
                "image_id": a["image_id"],
                "file_name": im["file_name"],
                "category": names.get(a["category_id"], "unknown"),
                "bbox_xywh": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area_px": round(area, 2),
                "normalized_area": norm,
                "min_side_px": round(min(w, h), 2),
                "flags": flags,
            }
        )
    areas = np.array([r["area_px"] for r in rows], dtype=float)
    norms = np.array([r["normalized_area"] for r in rows], dtype=float)
    small = [r for r in rows if r["flags"]]
    per_cat: dict[str, dict[str, Any]] = {}
    for name in names.values():
        sel = [r for r in rows if r["category"] == name]
        per_cat[name] = {
            "annotations": len(sel),
            "suspiciously_small": sum(1 for r in sel if r["flags"]),
            "median_normalized_area": float(np.median([r["normalized_area"] for r in sel])) if sel else None,
        }
    examples = sorted(small, key=lambda r: r["normalized_area"])[:max_examples]
    for e in examples:
        e["normalized_area"] = round(e["normalized_area"], 6)
    return {
        "thresholds_used": {"abs_area_threshold_px": abs_thr, "norm_area_threshold": norm_thr, "min_side_px": side_thr},
        "annotations_analyzed": len(rows),
        "suspiciously_small_count": len(small),
        "suspiciously_small_fraction": (len(small) / len(rows)) if rows else None,
        "count_by_criterion": {
            "area_px_below_threshold": sum("area_px_below_threshold" in r["flags"] for r in rows),
            "normalized_area_below_threshold": sum("normalized_area_below_threshold" in r["flags"] for r in rows),
            "min_side_below_threshold": sum("min_side_below_threshold" in r["flags"] for r in rows),
        },
        "invalid_geometry": {k: {"count": len(v), "annotation_ids": v[:10]} for k, v in invalid.items()},
        "area_px_distribution": _pct(areas),
        "normalized_area_distribution": _pct(norms),
        "per_category": per_cat,
        "smallest_examples": examples,
        "examples_shown": len(examples),
    }


def analyze_class_imbalance(ctx: AuditContext, args: dict[str, Any]) -> dict[str, Any]:
    """Class frequencies and imbalance measures (ratio max/min, normalised entropy, per-class ratios)."""
    coco = ctx.coco()
    names = _category_names(coco)
    counts = {name: 0 for name in names.values()}
    images_with: dict[str, set[int]] = {name: set() for name in names.values()}
    for a in coco.get("annotations", []):
        name = names.get(a["category_id"], f"unknown_{a['category_id']}")
        counts[name] = counts.get(name, 0) + 1
        images_with.setdefault(name, set()).add(a["image_id"])
    total = sum(counts.values())
    if total == 0:
        return {"error": "dataset has no annotations", "num_annotations": 0}
    freqs = {k: v / total for k, v in counts.items()}
    majority = max(counts, key=lambda k: counts[k])
    minority = min(counts, key=lambda k: counts[k])
    ratio = counts[majority] / counts[minority] if counts[minority] > 0 else None
    probs = np.array([f for f in freqs.values() if f > 0])
    entropy = float(-(probs * np.log(probs)).sum() / math.log(len(counts))) if len(counts) > 1 else 1.0
    label = None
    if ratio is not None:
        label = "balanced" if ratio < 1.5 else "mild" if ratio < 3 else "moderate" if ratio < 10 else "severe"
    return {
        "num_annotations": total,
        "class_counts": counts,
        "class_frequencies": {k: round(v, 4) for k, v in freqs.items()},
        "images_containing_class": {k: len(v) for k, v in images_with.items()},
        "majority_class": majority,
        "minority_class": minority,
        "imbalance_ratio_max_over_min": ratio,
        "per_class_ratio_to_majority": {k: round(counts[majority] / v, 3) if v else None for k, v in counts.items()},
        "normalized_entropy": entropy,
        "classes_with_zero_annotations": [k for k, v in counts.items() if v == 0],
        "heuristic_label": label,
        "heuristic_label_rule": "ratio <1.5 balanced; <3 mild; <10 moderate; >=10 severe (rule of thumb, not a "
        "statistical test)",
    }


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.array(values, dtype=float), 1)[0])


def get_training_metrics(ctx: AuditContext, args: dict[str, Any]) -> dict[str, Any]:
    """Recorded training/validation metrics, best checkpoint and computed trends from ``metrics.json``."""
    path = ctx.metrics_path
    if path is None or not Path(path).is_file():
        return {"available": False, "error": f"training metrics file not found: {path}"}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        history = data["history"]
    except (json.JSONDecodeError, KeyError) as exc:
        return {"available": False, "error": f"metrics file unreadable: {type(exc).__name__}: {exc}"}
    if not history:
        return {"available": False, "error": "metrics file contains no epochs"}
    best = data.get("best", {})
    monitor = best.get("monitor", "val_loss")
    mode = best.get("mode", "min")
    best_rec = next((h for h in history if h["epoch"] == best.get("epoch")), None)
    if best_rec is None:  # recompute from history rather than trusting a missing pointer
        pick = min if mode == "min" else max
        best_rec = pick(
            history,
            key=lambda h: h.get(monitor) if h.get(monitor) is not None else math.inf if mode == "min" else -math.inf,
        )
    last = history[-1]
    metric = args.get("metric")
    keys = [metric] if metric else ["train_loss", "val_loss", "val_ap50", "val_ap"]
    trends: dict[str, Any] = {}
    window = min(5, len(history))
    for key in keys:
        series = [h[key] for h in history if h.get(key) is not None]
        if not series:
            trends[key] = {"error": f"metric '{key}' not recorded"}
            continue
        trends[key] = {
            "first": series[0],
            "last": series[-1],
            "min": min(series),
            "max": max(series),
            "epoch_of_min": history[
                int(np.argmin([h.get(key, math.inf) if h.get(key) is not None else math.inf for h in history]))
            ]["epoch"],
            f"slope_per_epoch_last_{window}": _slope(series[-window:]),
            "relative_change_first_to_last": ((series[-1] - series[0]) / abs(series[0])) if series[0] else None,
        }
    flags: dict[str, Any] = {}
    if len(history) >= 3 and best_rec.get("val_loss") is not None and last.get("val_loss") is not None:
        tr_best, tr_last = best_rec.get("train_loss"), last.get("train_loss")
        flags["val_loss_last_vs_best_ratio"] = last["val_loss"] / best_rec["val_loss"] if best_rec["val_loss"] else None
        flags["train_loss_decreased_since_best"] = (
            (tr_last < tr_best) if tr_best is not None and tr_last is not None else None
        )
        flags["possible_overfitting_signal"] = bool(
            best_rec["epoch"] < last["epoch"]
            and flags["val_loss_last_vs_best_ratio"]
            and flags["val_loss_last_vs_best_ratio"] > 1.05
            and flags["train_loss_decreased_since_best"]
        )
        flags["train_val_loss_gap_at_last_epoch"] = (
            last["val_loss"] - last["train_loss"] if last.get("train_loss") is not None else None
        )
        flags["flag_rule"] = (
            "possible_overfitting_signal = best epoch is before the last epoch AND last val_loss > 1.05x best "
            "AND train_loss fell since the best epoch (heuristic, not proof)"
        )
    else:
        flags["note"] = "fewer than 3 epochs recorded: trend and overfitting heuristics are not meaningful"
    return {
        "available": True,
        "task": data.get("task"),
        "epochs_completed": data.get("epochs_completed", len(history)),
        "epochs_requested": data.get("epochs_requested"),
        "early_stopping": data.get("early_stopping"),
        "best_checkpoint": {
            "epoch": best_rec["epoch"],
            "selected_by": monitor,
            "mode": mode,
            "metrics_at_best_epoch": {k: v for k, v in best_rec.items() if k not in ("is_best",)},
            "checkpoint_file": best.get("checkpoint"),
        },
        "last_epoch": {k: v for k, v in last.items() if k != "is_best"},
        "trends": trends,
        "heuristic_flags": flags,
        "test_metrics_best_checkpoint": data.get("test"),
        "split_sizes": data.get("dataset", {}).get("split_sizes"),
        "model_parameters": data.get("model", {}).get("trainable_parameters"),
        "metric_notes": data.get("metric_notes"),
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """A callable tool with a JSON-schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[AuditContext, dict[str, Any]], dict[str, Any]]


NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

DEFAULT_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "get_dataset_statistics",
        "Exact counts from the COCO file: number of images, annotations, categories, annotations per category, "
        "images without annotations, per-image annotation counts, image size range and licence breakdown.",
        NO_ARGS,
        get_dataset_statistics,
    ),
    ToolSpec(
        "analyze_bbox_areas",
        "Computes bounding-box areas (pixels and normalised by image area) from the COCO JSON, counts suspiciously "
        "small boxes "
        "under configurable thresholds, reports invalid geometry (non-positive, non-finite, out-of-bounds) and "
        "returns the "
        "smallest example annotations with ids and file names. Omit thresholds to use the configured defaults.",
        {
            "type": "object",
            "properties": {
                "abs_area_threshold_px": {
                    "type": "number",
                    "minimum": 0,
                    "description": "box area below this many square pixels is suspicious",
                },
                "norm_area_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "box area / image area below this fraction is suspicious",
                },
                "min_side_px": {
                    "type": "number",
                    "minimum": 0,
                    "description": "box width or height below this many pixels is suspicious",
                },
                "max_examples": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "number of example annotations to return",
                },
            },
            "additionalProperties": False,
        },
        analyze_bbox_areas,
    ),
    ToolSpec(
        "analyze_class_imbalance",
        "Class frequencies, majority/minority class, imbalance ratio (max/min), normalised entropy and images per "
        "class "
        "computed from the COCO annotations.",
        NO_ARGS,
        analyze_class_imbalance,
    ),
    ToolSpec(
        "get_training_metrics",
        "Reads the recorded training run (metrics.json): per-epoch losses/AP, the best checkpoint, last epoch, trends "
        "(slopes, extrema), heuristic overfitting flags, early-stopping info and held-out test metrics if present.",
        {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["train_loss", "val_loss", "val_ap50", "val_ap"],
                    "description": "restrict trends to one metric (optional)",
                }
            },
            "additionalProperties": False,
        },
        get_training_metrics,
    ),
)


@dataclass
class ToolResult:
    """Outcome of one tool execution (always serialisable; never raises)."""

    name: str
    arguments: Any
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0

    def as_message_content(self) -> str:
        """JSON text handed back to the LLM."""
        payload = {"ok": True, "result": self.result} if self.ok else {"ok": False, "error": self.error}
        return json.dumps(payload, default=str, allow_nan=False) if self.ok or self.error else "{}"


class ToolRegistry:
    """Validates tool-call arguments against JSON schemas and executes tools safely."""

    MAX_RESULT_CHARS = 20_000

    def __init__(self, ctx: AuditContext, specs: tuple[ToolSpec, ...] = DEFAULT_TOOLS) -> None:
        self.ctx = ctx
        self._specs = {s.name: s for s in specs}
        self._validators = {s.name: Draft202012Validator(s.parameters) for s in specs}

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def schemas(self) -> list[dict[str, Any]]:
        """Provider-neutral tool definitions: ``name``, ``description``, ``parameters`` (JSON schema)."""
        return [
            {"name": s.name, "description": s.description, "parameters": s.parameters} for s in self._specs.values()
        ]

    def execute(self, name: str, arguments: Any) -> ToolResult:
        """Validate and run a tool. Unknown tools, bad arguments and exceptions become error results."""
        start = time.perf_counter()

        def fail(msg: str) -> ToolResult:
            return ToolResult(name, arguments, False, error=msg, duration_ms=(time.perf_counter() - start) * 1000)

        spec = self._specs.get(name)
        if spec is None:
            return fail(f"unknown tool '{name}'. Available tools: {self.names}")
        if not isinstance(arguments, dict):
            return fail(f"arguments must be a JSON object, got {type(arguments).__name__}")
        errors = sorted(self._validators[name].iter_errors(arguments), key=lambda e: list(e.path))
        if errors:
            return fail(
                "invalid arguments: "
                + "; ".join(f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5])
            )
        try:
            result = _sanitize(spec.func(self.ctx, arguments))
        except FileNotFoundError as exc:
            return fail(f"data file missing: {exc}")
        except Exception as exc:  # tools must never crash the agent loop
            return fail(f"{type(exc).__name__}: {exc}")
        if len(json.dumps(result, default=str)) > self.MAX_RESULT_CHARS:
            return fail("tool result too large; narrow the request (e.g. lower max_examples)")
        return ToolResult(name, arguments, True, result=result, duration_ms=(time.perf_counter() - start) * 1000)


def _sanitize(obj: Any) -> Any:
    """Make results strictly JSON-serialisable (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.floating | float):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
