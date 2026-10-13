"""Zero-shot annotation pipeline: post-processing, polygons, COCO assembly, caching, failure handling.

``StubDetector`` / ``StubSegmenter`` are TEST DOUBLES that replace the foundation models so the
surrounding logic can be verified offline. They produce no real predictions and nothing here is
reported as a model result. The ``*_interface`` tests run the real ``transformers`` classes with tiny
randomly initialised weights, which verifies API usage only (never detection quality).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from augle.annotation import (
    Detection,
    GroundingDinoDetector,
    SamSegmenter,
    annotate_collection,
    attach_mask,
    build_coco,
    config_fingerprint,
    filter_detections,
    mask_to_polygons,
    write_and_validate,
)
from augle.coco import load_coco, polygons_to_mask, validate_coco
from augle.config import AnnotationConfig
from augle.imaging import encode_jpeg


def det(box, score=0.9, cat="robot_arm"):
    return Detection(tuple(float(v) for v in box), score, cat)


# ------------------------------------------------------------ filtering ----
def test_filter_detections_rules():
    cfg = AnnotationConfig(min_box_area_frac=0.01, max_box_area_frac=0.9, nms_iou=0.5, max_detections_per_category=2)
    dets = [
        det((10, 10, 60, 60), 0.9),  # kept (best of overlapping pair)
        det((12, 12, 62, 62), 0.8),  # suppressed by NMS
        det((100, 100, 140, 140), 0.7),  # kept
        det((150, 150, 190, 190), 0.6),  # cut by max_detections_per_category
        det((0, 0, 5, 5), 0.9),  # too small (25 / 40000 < 1%)
        det((0, 0, 199, 199), 0.9),  # covers image
        det((20, 20, 20, 80), 0.9),  # zero width
        det((float("nan"), 0, 10, 10), 0.9),  # non finite
        det((-30, -30, 50, 50), 0.5, cat="gripper"),  # clipped to (0,0,50,50) and kept
    ]
    kept, dropped = filter_detections(dets, 200, 200, cfg)
    assert sorted((d.category, d.box_xyxy) for d in kept) == sorted(
        [
            ("robot_arm", (10.0, 10.0, 60.0, 60.0)),
            ("robot_arm", (100.0, 100.0, 140.0, 140.0)),
            ("gripper", (0.0, 0.0, 50.0, 50.0)),
        ]
    )
    assert dropped == {
        "box_too_small": 1,
        "box_covers_image": 1,
        "degenerate_box": 1,
        "non_finite": 1,
        "suppressed_by_nms": 1,
        "over_max_per_category": 1,
    }
    for d in kept:
        assert 0 <= d.box_xyxy[0] < d.box_xyxy[2] <= 200


def test_nms_is_applied_per_category():
    cfg = AnnotationConfig()
    kept, _ = filter_detections(
        [det((10, 10, 60, 60), 0.9, "robot_arm"), det((10, 10, 60, 60), 0.8, "gripper")], 200, 200, cfg
    )
    assert len(kept) == 2


# -------------------------------------------------------------- polygons ----
def blob(h=120, w=160) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (80, 60), (40, 18), 25, 0, 360, 1, -1)
    cv2.rectangle(m, (20, 20), (36, 50), 1, -1)
    return m


def test_mask_polygon_roundtrip_is_faithful():
    m = blob().astype(bool)
    polys = mask_to_polygons(m)
    assert len(polys) == 2 and all(len(p) >= 6 and len(p) % 2 == 0 for p in polys)
    back = polygons_to_mask(polys, *m.shape)
    assert (back & m).sum() / (back | m).sum() > 0.99


def test_mask_polygons_edge_cases():
    assert mask_to_polygons(np.zeros((20, 20), bool)) == []
    speck = np.zeros((20, 20), bool)
    speck[5, 5] = True
    assert mask_to_polygons(speck, min_area_px=16.0) == []  # sub-threshold noise dropped
    full = np.ones((30, 40), bool)
    poly = mask_to_polygons(full)[0]
    xs, ys = poly[0::2], poly[1::2]
    assert min(xs) >= 0 and max(xs) <= 40 and min(ys) >= 0 and max(ys) <= 30
    assert polygons_to_mask([poly], 30, 40).all()  # touching the image border works


def test_attach_mask_makes_bbox_area_and_polygons_consistent():
    d = det((0, 0, 1, 1))
    assert attach_mask(d, blob().astype(bool), AnnotationConfig())
    raster = polygons_to_mask(d.polygons, 120, 160)
    ys, xs = np.nonzero(raster)
    x, y, w, h = d.mask_bbox_xywh
    assert (x, y, w, h) == (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    assert d.mask_area == raster.sum()
    assert not attach_mask(det((0, 0, 1, 1)), np.zeros((10, 10), bool), AnnotationConfig())


# ------------------------------------------------------------------ COCO ----
def _write_image(root: Path, name: str, h: int, w: int) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / name).write_bytes(encode_jpeg(np.full((h, w, 3), 90, np.uint8), 90))


def _record(i: int, license_short="CC BY-SA 4.0") -> dict:
    return {
        "image_id": f"img{i}",
        "file_name": f"images/img{i}.jpg",
        "license_short": license_short,
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "original_url": f"https://example.org/{i}.jpg",
        "page_url": f"https://example.org/wiki/{i}",
        "author": "A. Author",
        "credit": "own work",
        "title": f"File:{i}.jpg",
    }


def test_build_coco_ids_licenses_and_bbox_only_semantics(tmp_path):
    for i in (1, 2, 3):
        _write_image(tmp_path, f"img{i}.jpg", 100, 150)
    masked = det((10, 20, 60, 70))
    assert attach_mask(
        masked, np.pad(np.ones((40, 40), np.uint8), ((25, 35), (15, 95))).astype(bool), AnnotationConfig()
    )
    dets = {"img1": [masked, det((70, 10, 140, 90), cat="gripper")], "img3": [det((5, 5, 50, 50))]}
    sizes = {k: (150, 100) for k in ("img1", "img2", "img3")}
    records = [_record(1), _record(2), _record(3, "CC0 1.0")]
    coco = build_coco(records, dets, sizes, ["robot_arm", "gripper"])
    assert [i["id"] for i in coco["images"]] == [1, 2] and [a["id"] for a in coco["annotations"]] == [1, 2, 3]
    assert [i["file_name"] for i in coco["images"]] == ["images/img1.jpg", "images/img3.jpg"]  # img2 had no detections
    assert {lic["name"] for lic in coco["licenses"]} == {"CC BY-SA 4.0", "CC0 1.0"}
    assert coco["images"][0]["author"] == "A. Author" and coco["images"][0]["coco_url"].startswith("https://")
    ann_masked, ann_box = coco["annotations"][0], coco["annotations"][1]
    assert ann_masked["segmentation"] and ann_masked["area"] <= ann_masked["bbox"][2] * ann_masked["bbox"][3]
    assert ann_box["segmentation"] == [] and ann_box["area"] == pytest.approx(
        ann_box["bbox"][2] * ann_box["bbox"][3], rel=1e-3
    )
    assert [c["name"] for c in coco["categories"]] == ["robot_arm", "gripper"]
    assert validate_coco(coco, tmp_path).ok


# --------------------------------------------------------------- pipeline ----
class StubDetector:
    """TEST DOUBLE: returns pre-programmed boxes, counts calls, can raise."""

    def __init__(self, plan: dict[int, list[Detection] | Exception]) -> None:
        self.plan, self.calls = plan, 0

    def detect(self, image_bgr, prompts):
        self.calls += 1
        item = self.plan[int(image_bgr[0, 0, 0])]  # image identity is encoded in the top-left pixel value
        if isinstance(item, Exception):
            raise item
        return [Detection(d.box_xyxy, d.score, d.category) for d in item]


class StubSegmenter:
    """TEST DOUBLE: returns the filled box as the 'mask'."""

    def segment(self, image_bgr, boxes):
        masks = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros(image_bgr.shape[:2], bool)
            m[int(y1) : int(y2), int(x1) : int(x2)] = True
            masks.append(m)
        return masks


def make_manifest(tmp_path: Path, n: int = 4) -> Path:
    records = []
    for i in range(n):
        (tmp_path / "images").mkdir(exist_ok=True)
        img = np.full((100, 150, 3), 10 * (i + 1), np.uint8)
        (tmp_path / "images" / f"img{i}.jpg").write_bytes(encode_jpeg(img, 100))
        records.append(_record(i))
    (tmp_path / "images" / "img9.jpg").write_bytes(b"not a jpeg")
    records.append(_record(9))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"accepted": records, "summary": {}, "source": "test"}))
    return path


def pixel_value(i: int) -> int:
    """Identity pixel of fixture image ``i`` (flat images survive JPEG within +-1, handled by ``Plan``)."""
    return 10 * (i + 1)


def test_pipeline_end_to_end_with_stubs_failures_and_cache(tmp_path):
    manifest = make_manifest(tmp_path)
    cfg = AnnotationConfig(cache_dir=str(tmp_path / "cache"), use_sam=True, min_box_area_frac=0.001)

    class Plan(dict):
        def __missing__(self, key):  # tolerate +-1 JPEG rounding of the identity pixel
            for k in (key - 1, key + 1):
                if k in self:
                    return self[k]
            raise KeyError(key)

    plan = Plan(
        {
            pixel_value(0): [det((10, 10, 70, 60)), det((80, 20, 140, 90), cat="gripper")],
            pixel_value(1): [],  # model finds nothing -> dropped from COCO
            pixel_value(2): RuntimeError("CUDA out of memory"),  # inference failure -> recorded, run continues
            pixel_value(3): [det((20, 20, 100, 80))],
        }
    )
    detector = StubDetector(plan)
    coco, summary = annotate_collection(manifest, tmp_path, detector, StubSegmenter(), cfg)
    assert summary.images_in_manifest == 5 and summary.images_processed == 3 and summary.images_failed == 2
    assert summary.images_with_detections == 2 and summary.images_without_detections == 1
    reasons = " ".join(f["error"] for f in summary.failures)
    assert "CUDA out of memory" in reasons and "ImageReadError" in reasons  # corrupt file + model failure both recorded
    assert summary.annotations == 3 == len(coco["annotations"]) and summary.annotations_with_masks == 3
    assert summary.per_category == {"robot_arm": 2, "gripper": 1}
    write_and_validate(coco, tmp_path / "instances.json", tmp_path)  # real validation incl. file dimensions
    assert validate_coco(load_coco(tmp_path / "instances.json"), tmp_path).ok

    # second run: every successful image comes from the cache, detector only retried for the failed one
    calls_before = detector.calls
    plan[pixel_value(2)] = [det((30, 30, 90, 70))]
    coco2, summary2 = annotate_collection(manifest, tmp_path, detector, StubSegmenter(), cfg)
    assert (
        detector.calls - calls_before == 1
    )  # only the previously failed image is re-run; the corrupt file fails before inference
    assert summary2.images_failed == 1  # the corrupt file still fails, nothing is invented for it
    assert summary2.annotations == 4

    # changing a setting that affects model output invalidates the cache
    cfg.box_threshold = 0.5
    calls_before = detector.calls
    annotate_collection(manifest, tmp_path, detector, StubSegmenter(), cfg)
    assert detector.calls - calls_before == 4


def test_config_fingerprint_changes_with_model_settings_only():
    a, b = AnnotationConfig(), AnnotationConfig()
    assert config_fingerprint(a) == config_fingerprint(b)
    b.box_threshold = 0.5
    assert config_fingerprint(a) != config_fingerprint(b)
    c = AnnotationConfig(cache_dir="elsewhere", output_json="other.json", device="cpu")
    assert config_fingerprint(a) == config_fingerprint(c)


def test_images_without_detections_can_be_kept_as_negatives(tmp_path):
    manifest = make_manifest(tmp_path, 2)
    cfg = AnnotationConfig(cache_dir=str(tmp_path / "c"), use_sam=False, drop_images_without_detections=False)
    detector = StubDetector({10: [], 20: [det((10, 10, 70, 60))]})
    coco, summary = annotate_collection(manifest, tmp_path, detector, None, cfg)
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 1


# ------------------------------------------- real transformers, tiny random ----
@pytest.mark.slow
def test_grounding_dino_wrapper_interface_with_random_weights(tmp_path):
    """Verifies our calls match the installed transformers API (random weights: no meaning in outputs)."""
    torch = pytest.importorskip("torch")
    from transformers import (
        BertConfig,
        BertTokenizerFast,
        GroundingDinoConfig,
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
        SwinConfig,
    )
    from transformers.models.grounding_dino.image_processing_grounding_dino import GroundingDinoImageProcessor

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "robot", "arm", "industrial", "gripper"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=True)
    swin = SwinConfig(
        embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 1, 2, 2], window_size=7, out_indices=[2, 3, 4], image_size=224
    )
    text = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
    )
    cfg = GroundingDinoConfig(
        backbone_config=swin,
        text_config=text,
        use_timm_backbone=False,
        num_queries=20,
        encoder_layers=1,
        decoder_layers=1,
        d_model=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        num_feature_levels=4,
        max_text_len=16,
        backbone=None,
        use_pretrained_backbone=False,
    )
    detector = object.__new__(GroundingDinoDetector)
    detector.torch, detector.device = torch, "cpu"
    detector.model = GroundingDinoForObjectDetection(cfg).eval()
    detector.processor = GroundingDinoProcessor(
        image_processor=GroundingDinoImageProcessor(size={"shortest_edge": 128, "longest_edge": 192}), tokenizer=tok
    )
    detector.box_threshold = detector.text_threshold = 0.0
    img = np.random.default_rng(0).integers(0, 255, (100, 160, 3)).astype(np.uint8)
    dets = detector.detect(img, {"robot_arm": ["industrial robot arm"], "gripper": ["robot gripper"]})
    assert dets and {d.category for d in dets} == {"robot_arm", "gripper"}
    for d in dets:  # boxes come back in absolute pixel coordinates of the *original* image
        assert len(d.box_xyxy) == 4 and np.all(np.isfinite(d.box_xyxy)) and 0 <= d.score <= 1
        assert -200 < d.box_xyxy[0] < 360 and -200 < d.box_xyxy[1] < 300


@pytest.mark.slow
def test_sam_wrapper_interface_with_random_weights():
    torch = pytest.importorskip("torch")
    from transformers import SamConfig, SamModel, SamProcessor, SamVisionConfig
    from transformers.models.sam.image_processing_sam import SamImageProcessor

    vision = SamVisionConfig(
        hidden_size=32,
        output_channels=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_dim=64,
        global_attn_indexes=[1],
    )
    seg = object.__new__(SamSegmenter)
    seg.torch, seg.device = torch, "cpu"
    seg.model = SamModel(SamConfig(vision_config=vision)).eval()
    seg.processor = SamProcessor(SamImageProcessor())
    img = np.random.default_rng(0).integers(0, 255, (120, 200, 3)).astype(np.uint8)
    masks = seg.segment(img, np.array([[10.0, 10.0, 80.0, 90.0], [100.0, 20.0, 190.0, 100.0]]))
    assert len(masks) == 2 and all(m.shape == (120, 200) and m.dtype == bool for m in masks)
    assert seg.segment(img, np.zeros((0, 4))) == []
