"""Deterministic audit tools and the schema-validated registry (no LLM involved)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from augle.audit_tools import (
    AuditContext,
    ToolRegistry,
    ToolSpec,
    analyze_bbox_areas,
    analyze_class_imbalance,
    get_dataset_statistics,
    get_training_metrics,
)


def write_coco(path: Path, boxes: list[tuple[int, int, list]], *, img=(1000, 500), cats=("a", "b", "c")) -> Path:
    """boxes: (image_id, category_id, bbox). Images 1..max are created with size ``img`` (w, h)."""
    n_images = max([b[0] for b in boxes] + [1])
    coco = {
        "images": [
            {"id": i, "file_name": f"im{i}.jpg", "width": img[0], "height": img[1]} for i in range(1, n_images + 1)
        ],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(cats)],
        "annotations": [
            {"id": k + 1, "image_id": i, "category_id": c, "bbox": b, "area": 1.0, "iscrowd": 0}
            for k, (i, c, b) in enumerate(boxes)
        ],
    }
    path.write_text(json.dumps(coco))
    return path


def ctx_for(path: Path, metrics: Path | None = None, **kw) -> AuditContext:
    return AuditContext(path, metrics, **kw)


# ------------------------------------------------------------- statistics ----
def test_dataset_statistics_match_independent_count(coco_path):
    raw = json.loads(coco_path.read_text())
    res = get_dataset_statistics(ctx_for(coco_path), {})
    assert (
        res["num_images"] == len(raw["images"])
        and res["num_annotations"] == len(raw["annotations"])
        and res["num_categories"] == 2
    )
    names = {c["id"]: c["name"] for c in raw["categories"]}
    expected = {n: sum(1 for a in raw["annotations"] if names[a["category_id"]] == n) for n in names.values()}
    assert res["annotations_per_category"] == expected
    assert sum(res["annotations_per_category"].values()) == res["num_annotations"]
    assert res["annotations_with_segmentation"] == res["num_annotations"]


def test_statistics_with_empty_images_and_unknown_category(tmp_path):
    p = write_coco(
        tmp_path / "c.json", [(1, 1, [10, 10, 50, 50]), (3, 9, [10, 10, 50, 50])]
    )  # image 2 empty, category 9 unknown
    res = get_dataset_statistics(ctx_for(p), {})
    assert res["num_images"] == 3 and res["images_without_annotations"] == 1
    assert res["annotations_per_category"]["a"] == 1 and res["annotations_per_category"]["unknown_9"] == 1
    assert res["annotations_per_image"]["max"] == 1 and res["annotations_per_image"]["min"] == 0


# ------------------------------------------------------------- bbox areas ----
def test_bbox_area_analysis_exact_counts_and_examples(tmp_path):
    boxes = [
        (1, 1, [10, 10, 3, 3]),  # tiny: area 9, norm 1.8e-5
        (1, 1, [100, 100, 100, 4]),  # thin: area 400 (norm 8e-4), min side 4
        (1, 2, [200, 100, 200, 200]),  # normal: area 40000
        (1, 2, [0, 0, 1000, 500]),  # full image, normal
        (2, 3, [900, 400, 100, 100]),  # normal, touches the corner exactly
        (2, 1, [10, 10, 0, 20]),  # zero width  -> invalid
        (2, 1, [10, 10, 20, -5]),  # negative height -> invalid
        (2, 1, [950, 450, 100, 100]),  # out of bounds
        (2, 1, [float("nan"), 0, 10, 10]),  # non finite  -> invalid (JSON writes NaN)
    ]
    p = write_coco(tmp_path / "c.json", boxes)
    res = analyze_bbox_areas(
        ctx_for(p), {"abs_area_threshold_px": 1000, "norm_area_threshold": 0.001, "min_side_px": 8, "max_examples": 5}
    )
    assert res["annotations_analyzed"] == 6  # 3 invalid shapes excluded from the area analysis
    assert res["suspiciously_small_count"] == 2  # tiny + thin
    assert res["count_by_criterion"] == {
        "area_px_below_threshold": 2,
        "normalized_area_below_threshold": 2,
        "min_side_below_threshold": 2,
    }  # thin box: 400/500000 = 8e-4 < 1e-3
    inv = res["invalid_geometry"]
    assert (
        inv["non_positive_size"]["count"] == 2
        and inv["non_finite"]["count"] == 1
        and inv["out_of_bounds"]["count"] == 1
    )
    ex = res["smallest_examples"]
    assert [e["annotation_id"] for e in ex] == [1, 2]  # sorted ascending by normalised area
    assert ex[0]["area_px"] == 9.0 and ex[0]["bbox_xywh"] == [10.0, 10.0, 3.0, 3.0] and ex[0]["file_name"] == "im1.jpg"
    assert ex[0]["normalized_area"] == pytest.approx(9 / 500000, rel=1e-3)
    assert res["thresholds_used"] == {"abs_area_threshold_px": 1000.0, "norm_area_threshold": 0.001, "min_side_px": 8.0}
    assert res["per_category"]["a"]["suspiciously_small"] == 2 and res["per_category"]["b"]["suspiciously_small"] == 0
    assert res["area_px_distribution"]["p0"] == 9.0 and res["area_px_distribution"]["p100"] == 500000.0


def test_thresholds_are_configurable_and_defaults_come_from_context(tmp_path):
    p = write_coco(tmp_path / "c.json", [(1, 1, [0, 0, 50, 50]), (1, 1, [0, 0, 100, 100]), (1, 1, [0, 0, 300, 300])])
    ctx = ctx_for(p, small_abs_area_px=5000, small_norm_area=0.0, min_side_px=0)
    assert analyze_bbox_areas(ctx, {})["suspiciously_small_count"] == 1  # defaults: 2500 px^2 < 5000
    assert analyze_bbox_areas(ctx, {"abs_area_threshold_px": 20000})["suspiciously_small_count"] == 2
    assert analyze_bbox_areas(ctx, {"abs_area_threshold_px": 0})["suspiciously_small_count"] == 0
    assert analyze_bbox_areas(ctx, {"abs_area_threshold_px": 1e9, "max_examples": 2})["examples_shown"] == 2


def test_bbox_analysis_on_dataset_with_no_annotations(tmp_path):
    p = write_coco(tmp_path / "c.json", [])
    res = analyze_bbox_areas(ctx_for(p), {})
    assert (
        res["annotations_analyzed"] == 0
        and res["suspiciously_small_fraction"] is None
        and res["smallest_examples"] == []
    )


# --------------------------------------------------------------- imbalance ----
def test_class_imbalance_numbers(tmp_path):
    boxes = [(1, 1, [0, 0, 10, 10])] * 90 + [(1, 2, [0, 0, 10, 10])] * 9 + [(2, 3, [0, 0, 10, 10])]
    res = analyze_class_imbalance(ctx_for(write_coco(tmp_path / "c.json", boxes)), {})
    assert res["class_counts"] == {"a": 90, "b": 9, "c": 1}
    assert res["imbalance_ratio_max_over_min"] == 90.0 and res["heuristic_label"] == "severe"
    assert res["majority_class"] == "a" and res["minority_class"] == "c"
    p = np.array([0.9, 0.09, 0.01])
    assert res["normalized_entropy"] == pytest.approx(float(-(p * np.log(p)).sum() / math.log(3)))
    assert res["per_class_ratio_to_majority"] == {"a": 1.0, "b": 10.0, "c": 90.0}
    assert res["images_containing_class"] == {"a": 1, "b": 1, "c": 1}


def test_imbalance_zero_class_balanced_and_empty(tmp_path):
    res = analyze_class_imbalance(
        ctx_for(write_coco(tmp_path / "z.json", [(1, 1, [0, 0, 5, 5])] * 4 + [(1, 2, [0, 0, 5, 5])] * 4)), {}
    )
    assert (
        res["classes_with_zero_annotations"] == ["c"]
        and res["imbalance_ratio_max_over_min"] is None
        and res["heuristic_label"] is None
    )
    bal = analyze_class_imbalance(
        ctx_for(
            write_coco(
                tmp_path / "b.json",
                [(1, 1, [0, 0, 5, 5])] * 5 + [(1, 2, [0, 0, 5, 5])] * 5 + [(1, 3, [0, 0, 5, 5])] * 5,
            )
        ),
        {},
    )
    assert bal["heuristic_label"] == "balanced" and bal["normalized_entropy"] == pytest.approx(1.0)
    assert "error" in analyze_class_imbalance(ctx_for(write_coco(tmp_path / "e.json", [])), {})


# ------------------------------------------------------- training metrics ----
def write_metrics(path: Path, val: list[float], train: list[float], **extra) -> Path:
    hist = [
        {
            "epoch": i + 1,
            "lr": 1e-3,
            "train_loss": t,
            "val_loss": v,
            "val_ap50": 0.1 * (i + 1),
            "val_ap": 0.05 * (i + 1),
        }
        for i, (v, t) in enumerate(zip(val, train))
    ]
    best_i = int(np.argmin(val))
    for i, h in enumerate(hist):  # like train.py: every epoch that improves on the running best is flagged
        h["is_best"] = i == 0 or val[i] < min(val[:i])
    data = {
        "task": "detection",
        "epochs_completed": len(hist),
        "epochs_requested": 50,
        "history": hist,
        "best": {
            "epoch": best_i + 1,
            "monitor": "val_loss",
            "mode": "min",
            "value": val[best_i],
            "checkpoint": "best.pt",
        },
        "early_stopping": {"triggered": False},
        "dataset": {"split_sizes": {"train": 7, "val": 2, "test": 2}},
        "test": {"ap50": 0.4},
    } | extra
    path.write_text(json.dumps(data))
    return path


def test_training_metrics_best_checkpoint_trends_and_overfitting_flag(tmp_path, coco_path):
    m = write_metrics(
        tmp_path / "m.json", val=[2.0, 1.5, 1.2, 1.3, 1.5, 1.8, 2.2], train=[2.1, 1.6, 1.2, 1.0, 0.8, 0.6, 0.5]
    )
    res = get_training_metrics(ctx_for(coco_path, m), {})
    assert (
        res["available"]
        and res["best_checkpoint"]["epoch"] == 3
        and res["best_checkpoint"]["selected_by"] == "val_loss"
    )
    assert res["best_checkpoint"]["metrics_at_best_epoch"]["val_loss"] == 1.2
    assert res["trends"]["val_loss"]["epoch_of_min"] == 3 and res["trends"]["val_loss"]["slope_per_epoch_last_5"] > 0
    assert res["trends"]["train_loss"]["slope_per_epoch_last_5"] < 0
    flags = res["heuristic_flags"]
    assert flags["possible_overfitting_signal"] is True and flags["val_loss_last_vs_best_ratio"] == pytest.approx(
        2.2 / 1.2
    )
    assert res["test_metrics_best_checkpoint"] == {"ap50": 0.4} and res["split_sizes"] == {
        "train": 7,
        "val": 2,
        "test": 2,
    }


def test_training_metrics_healthy_run_has_no_overfitting_flag(tmp_path, coco_path):
    m = write_metrics(tmp_path / "m.json", val=[2.0, 1.6, 1.3, 1.1, 1.0], train=[2.1, 1.7, 1.4, 1.2, 1.0])
    res = get_training_metrics(ctx_for(coco_path, m), {"metric": "val_loss"})
    assert res["heuristic_flags"]["possible_overfitting_signal"] is False and list(res["trends"]) == ["val_loss"]
    assert res["trends"]["val_loss"]["relative_change_first_to_last"] == pytest.approx(-0.5)


def test_training_metrics_short_missing_and_corrupt_files(tmp_path, coco_path):
    short = get_training_metrics(ctx_for(coco_path, write_metrics(tmp_path / "s.json", [1.0, 0.9], [1.0, 0.8])), {})
    assert "note" in short["heuristic_flags"]
    missing = get_training_metrics(ctx_for(coco_path, tmp_path / "nope.json"), {})
    assert missing["available"] is False and "not found" in missing["error"]
    assert get_training_metrics(ctx_for(coco_path, None), {})["available"] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert get_training_metrics(ctx_for(coco_path, bad), {})["available"] is False
    (tmp_path / "noeps.json").write_text(json.dumps({"history": []}))
    assert "no epochs" in get_training_metrics(ctx_for(coco_path, tmp_path / "noeps.json"), {})["error"]


def test_training_metrics_recomputes_best_when_pointer_missing(tmp_path, coco_path):
    p = write_metrics(tmp_path / "m.json", [2.0, 1.0, 1.5], [2.0, 1.5, 1.2])
    data = json.loads(p.read_text())
    data["best"]["epoch"] = 99
    p.write_text(json.dumps(data))
    assert get_training_metrics(ctx_for(coco_path, p), {})["best_checkpoint"]["epoch"] == 2


# ---------------------------------------------------------------- registry ----
@pytest.fixture()
def registry(coco_path, tmp_path):
    return ToolRegistry(ctx_for(coco_path, write_metrics(tmp_path / "m.json", [2.0, 1.0, 1.5], [2.0, 1.5, 1.2])))


def test_registry_executes_all_four_required_tools(registry):
    assert registry.names == [
        "get_dataset_statistics",
        "analyze_bbox_areas",
        "analyze_class_imbalance",
        "get_training_metrics",
    ]
    for name in registry.names:
        r = registry.execute(name, {})
        assert r.ok and r.result and r.duration_ms >= 0
        json.loads(r.as_message_content())  # strictly valid JSON for the LLM
    schemas = registry.schemas()
    assert all(
        {"name", "description", "parameters"} == set(s) and s["parameters"]["additionalProperties"] is False
        for s in schemas
    )


@pytest.mark.parametrize(
    "name,args,needle",
    [
        ("nope", {}, "unknown tool"),
        ("analyze_bbox_areas", {"abs_area_threshold_px": "huge"}, "not of type 'number'"),
        ("analyze_bbox_areas", {"abs_area_threshold_px": -5}, "minimum"),
        ("analyze_bbox_areas", {"norm_area_threshold": 2}, "maximum"),
        ("analyze_bbox_areas", {"max_examples": 500}, "maximum"),
        ("analyze_bbox_areas", {"max_examples": 2.5}, "integer"),
        ("analyze_bbox_areas", {"path": "/etc/passwd"}, "Additional properties"),
        ("get_dataset_statistics", {"coco_path": "/tmp/other.json"}, "Additional properties"),
        ("get_training_metrics", {"metric": "accuracy"}, "is not one of"),
        ("analyze_class_imbalance", "not a dict", "must be a JSON object"),
        ("analyze_class_imbalance", None, "must be a JSON object"),
    ],
)
def test_registry_rejects_invalid_calls_without_raising(registry, name, args, needle):
    r = registry.execute(name, args)
    assert not r.ok and needle in r.error
    assert json.loads(r.as_message_content())["ok"] is False


def test_registry_converts_tool_exceptions_and_missing_files(coco_path, tmp_path):
    def boom(ctx, args):
        raise RuntimeError("kaboom")

    def nan_result(ctx, args):
        return {"x": float("nan"), "y": np.float32(2.5), "z": np.int64(3)}

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    reg = ToolRegistry(
        ctx_for(coco_path),
        (
            ToolSpec("boom", "d", schema, boom),
            ToolSpec("nan", "d", schema, nan_result),
            ToolSpec("big", "d", schema, lambda c, a: {"s": "x" * 50_000}),
        ),
    )
    assert "RuntimeError: kaboom" in reg.execute("boom", {}).error
    assert reg.execute("nan", {}).result == {"x": None, "y": 2.5, "z": 3}
    assert "too large" in reg.execute("big", {}).error
    gone = ToolRegistry(ctx_for(tmp_path / "missing.json"))
    r = gone.execute("get_dataset_statistics", {})
    assert not r.ok and "missing" in r.error


def test_coco_cache_refreshes_when_file_changes(tmp_path):
    p = write_coco(tmp_path / "c.json", [(1, 1, [0, 0, 5, 5])])
    ctx = ctx_for(p)
    assert get_dataset_statistics(ctx, {})["num_annotations"] == 1
    import os
    import time

    write_coco(p, [(1, 1, [0, 0, 5, 5])] * 3)
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert get_dataset_statistics(ctx, {})["num_annotations"] == 3
