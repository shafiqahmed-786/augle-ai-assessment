"""COCO schema validation: positive case, many negative cases, and a pycocotools cross-check."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from augle.coco import (
    iou_matrix_xyxy,
    iou_xyxy,
    load_coco,
    nms_xyxy,
    validate_coco,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


@pytest.fixture()
def coco(coco_path: Path) -> dict:
    return load_coco(coco_path)


def errors_of(coco: dict, root: Path, **kw) -> list[str]:
    return validate_coco(coco, root, **kw).errors


def test_valid_dataset_passes(coco, synthetic_root):
    rep = validate_coco(coco, synthetic_root)
    assert rep.ok, rep.summary()
    assert rep.stats["images"] == len(coco["images"]) and rep.stats["files_checked"] is True


def test_box_format_conversions_roundtrip():
    box = (10.0, 20.0, 30.0, 40.0)
    assert xywh_to_xyxy(box) == (10.0, 20.0, 40.0, 60.0)
    assert xyxy_to_xywh(xywh_to_xyxy(box)) == box


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda c: c.pop("images"), "missing required top-level key 'images'"),
        (lambda c: c["annotations"].append(copy.deepcopy(c["annotations"][0])), "duplicate annotation id"),
        (lambda c: c["images"].append(copy.deepcopy(c["images"][0])), "duplicate image id"),
        (lambda c: c["categories"].append(copy.deepcopy(c["categories"][0])), "duplicate category id"),
        (lambda c: c["annotations"][0].update(category_id=999), "category_id 999"),
        (lambda c: c["annotations"][0].update(image_id=99999), "image_id 99999"),
        (lambda c: c["annotations"][0].update(bbox=[10, 10, 0, 5]), "degenerate bbox"),
        (lambda c: c["annotations"][0].update(bbox=[10, 10, -3, 5]), "degenerate bbox"),
        (lambda c: c["annotations"][0].update(bbox=[10, 10, float("nan"), 5]), "4 finite numbers"),
        (lambda c: c["annotations"][0].update(bbox=[10, 10, 5]), "4 finite numbers"),
        (lambda c: c["annotations"][0].update(bbox=[-50, 10, 20, 5]), "outside image"),
        (lambda c: c["annotations"][0].update(bbox=[0, 0, 5000, 5000]), "outside image"),
        (lambda c: c["annotations"][0].update(area=-1), "area must be"),
        (lambda c: c["annotations"][0].update(area=10**7), "exceeds bbox area"),
        (lambda c: c["annotations"][0].update(iscrowd=2), "iscrowd"),
        (lambda c: c["annotations"][0].update(segmentation=[[1, 2, 3]]), "even-length list"),
        (lambda c: c["images"][0].update(width=0), "positive integers"),
        (lambda c: c["images"][0].update(width=9999), "JSON says 9999x"),
        (lambda c: c["images"][0].update(file_name="images/does_not_exist.jpg"), "does not exist"),
        (lambda c: c["images"][0].update(license=77), "license id 77"),
    ],
)
def test_invalid_variants_are_detected(coco, synthetic_root, mutate, needle):
    bad = copy.deepcopy(coco)
    mutate(bad)
    errs = errors_of(bad, synthetic_root)
    assert any(needle in e for e in errs), (needle, errs[:5])


def test_polygon_outside_its_bbox_is_flagged(coco, synthetic_root):
    bad = copy.deepcopy(coco)
    ann = next(a for a in bad["annotations"] if a["segmentation"])
    x, y, w, h = ann["bbox"]
    ann["segmentation"] = [[x + w + 40, y, x + w + 60, y, x + w + 60, y + 10]]
    assert any("extends outside its bbox" in e for e in errors_of(bad, synthetic_root))


def test_bbox_only_annotation_requires_matching_area(coco, synthetic_root):
    bad = copy.deepcopy(coco)
    ann = bad["annotations"][0]
    ann["segmentation"] = []
    ann["area"] = ann["bbox"][2] * ann["bbox"][3] * 0.5
    assert any("!= bbox w*h" in e for e in errors_of(bad, synthetic_root))
    ann["area"] = ann["bbox"][2] * ann["bbox"][3]
    assert not errors_of(bad, synthetic_root)


def test_images_without_annotations_are_a_warning_not_an_error(coco, synthetic_root):
    c = copy.deepcopy(coco)
    c["annotations"] = [a for a in c["annotations"] if a["image_id"] != c["images"][0]["id"]]
    rep = validate_coco(c, synthetic_root)
    assert rep.ok and any("no annotations" in w for w in rep.warnings)
    assert rep.stats["images_without_annotations"] == 1


def test_missing_image_root_skips_file_checks_with_warning(coco):
    rep = validate_coco(coco, None)
    assert rep.ok and rep.stats["files_checked"] is False and rep.warnings


def test_load_coco_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_coco(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_coco(bad)


def test_raise_if_invalid_carries_summary(coco, synthetic_root):
    bad = copy.deepcopy(coco)
    bad["annotations"][0]["category_id"] = 12345
    with pytest.raises(ValueError, match="FAILED"):
        validate_coco(bad, synthetic_root).raise_if_invalid()


def test_pycocotools_agrees_with_our_rasteriser_and_boxes(coco_path):
    """Independent cross-check against the reference COCO API (pycocotools)."""
    pycocotools = pytest.importorskip("pycocotools.coco")
    from dataset import segmentation_to_mask

    api = pycocotools.COCO(str(coco_path))
    assert len(api.getImgIds()) == len(load_coco(coco_path)["images"])
    ious = []
    for ann_id in api.getAnnIds():
        ann = api.loadAnns(ann_id)[0]
        img = api.imgs[ann["image_id"]]
        ref = api.annToMask(ann).astype(bool)
        ours = segmentation_to_mask(ann["segmentation"], img["height"], img["width"]).astype(bool)
        assert ref.shape == ours.shape == (img["height"], img["width"])
        ious.append((ref & ours).sum() / max((ref | ours).sum(), 1))
        # stored bbox is the tight pixel box of our rasterised polygon, within 1px of pycocotools' box
        ys, xs = np.nonzero(ref)
        x, y, w, h = ann["bbox"]
        assert abs(xs.min() - x) <= 1 and abs(ys.min() - y) <= 1
        assert abs(xs.max() + 1 - (x + w)) <= 1 and abs(ys.max() + 1 - (y + h)) <= 1
    assert min(ious) > 0.75 and float(np.mean(ious)) > 0.9, (min(ious), float(np.mean(ious)))


def test_iou_and_nms_helpers():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert nms_xyxy(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    assert iou_matrix_xyxy(boxes, boxes).shape == (3, 3)
    assert iou_matrix_xyxy(np.zeros((0, 4)), boxes).shape == (0, 3)
