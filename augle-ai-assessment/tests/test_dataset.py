"""Custom Dataset / DataLoader behaviour on COCO fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from augle.config import DataConfig
from dataset import CocoDetectionDataset, build_dataloader, collate_fn, draw_annotations, make_splits

CFG = DataConfig(image_size=96)


def make(coco_path: Path, root: Path, **kw) -> CocoDetectionDataset:
    return CocoDetectionDataset(coco_path, root, cfg=kw.pop("cfg", CFG), **kw)


def test_item_types_shapes_and_ranges(coco_path, synthetic_root):
    ds = make(coco_path, synthetic_root, train=True, seed=3)
    assert len(ds) == 24 and ds.num_classes == 2 and ds.class_names == ["robot_arm", "gripper"]
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (3, 96, 96) and item["image"].dtype == torch.float32
        n = len(item["boxes"])
        assert item["boxes"].shape == (n, 4) and item["labels"].shape == (n,) and item["labels"].dtype == torch.int64
        assert item["masks"] is None or item["masks"].shape == (n, 96, 96)
        if n:
            assert (item["boxes"][:, 2] > item["boxes"][:, 0]).all() and (
                item["boxes"][:, 3] > item["boxes"][:, 1]
            ).all()
            assert item["boxes"].min() >= 0 and item["boxes"].max() <= 96
            assert int(item["labels"].max()) < 2 and int(item["labels"].min()) >= 0
        assert torch.isfinite(item["image"]).all()


def test_eval_split_has_no_augmentation_and_is_epoch_independent(coco_path, synthetic_root):
    ds = make(coco_path, synthetic_root, train=False, seed=1)
    a = ds[2]
    ds.set_epoch(5)
    b = ds[2]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["boxes"], b["boxes"])


def test_train_augmentation_is_reproducible_and_varies_with_epoch(coco_path, synthetic_root):
    ds1, ds2 = make(coco_path, synthetic_root, train=True, seed=9), make(coco_path, synthetic_root, train=True, seed=9)
    assert torch.equal(ds1[4]["image"], ds2[4]["image"])  # same (seed, epoch, index) -> identical
    ds2.set_epoch(1)
    assert not torch.equal(ds1[4]["image"], ds2[4]["image"])  # new epoch -> new augmentation


def test_boxes_match_masks_for_every_sample(coco_path, synthetic_root):
    from dataset import mask_tight_boxes

    ds = make(
        coco_path, synthetic_root, train=True, seed=2, cfg=DataConfig(image_size=96, refine_boxes_with_masks=True)
    )
    for i in range(len(ds)):
        item = ds[i]
        if item["masks"] is not None and len(item["boxes"]):
            assert np.array_equal(mask_tight_boxes(item["masks"].numpy()), item["boxes"].numpy())


def test_collate_and_dataloader(coco_path, synthetic_root):
    ds = make(coco_path, synthetic_root, train=True, seed=0)
    loader = build_dataloader(ds, 5, shuffle=True, seed=0)
    batch = next(iter(loader))
    assert batch["images"].shape == (5, 3, 96, 96)
    assert len(batch["boxes"]) == len(batch["labels"]) == len(batch["masks"]) == 5
    assert batch["image_ids"].dtype == torch.long and len(batch["file_names"]) == 5
    assert collate_fn([ds[0], ds[1]])["images"].shape[0] == 2


def test_shuffle_is_seeded_and_workers_do_not_change_content(coco_path, synthetic_root):
    ds = make(coco_path, synthetic_root, train=True, seed=4)
    a = [b["image_ids"].tolist() for b in build_dataloader(ds, 6, shuffle=True, seed=7)]
    b = [b["image_ids"].tolist() for b in build_dataloader(ds, 6, shuffle=True, seed=7)]
    assert a == b
    w0 = next(iter(build_dataloader(ds, 6, shuffle=False, seed=0, num_workers=0)))
    w2 = next(iter(build_dataloader(ds, 6, shuffle=False, seed=0, num_workers=2)))
    assert torch.equal(w0["images"], w2["images"])  # augmentation RNG is keyed by (seed, epoch, index)


def test_make_splits_deterministic_disjoint_complete():
    ids = list(range(1, 51))
    s1, s2 = make_splits(ids, (0.7, 0.15, 0.15), 42), make_splits(ids, (0.7, 0.15, 0.15), 42)
    assert s1 == s2 and make_splits(ids, (0.7, 0.15, 0.15), 43) != s1
    all_ids = s1["train"] + s1["val"] + s1["test"]
    assert sorted(all_ids) == ids and len(set(all_ids)) == 50
    assert len(s1["val"]) == 8 and len(s1["test"]) == 8
    tiny = make_splits([1, 2, 3], (0.7, 0.15, 0.15), 0)
    assert all(len(v) == 1 for v in tiny.values())
    with pytest.raises(ValueError):
        make_splits([1, 2], (0.7, 0.15, 0.15), 0)
    with pytest.raises(ValueError):
        make_splits(ids, (0.5, 0.5, 0.5), 0)


def test_images_without_annotations_yield_empty_targets(coco_path, synthetic_root, tmp_path):
    coco = json.loads(coco_path.read_text())
    victim = coco["images"][0]["id"]
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != victim]
    path = tmp_path / "noann.json"
    path.write_text(json.dumps(coco))
    ds = make(path, synthetic_root, train=True, seed=0)
    idx = [i for i, (im, _) in enumerate(ds.records) if im["id"] == victim][0]
    item = ds[idx]
    assert item["boxes"].shape == (0, 4) and item["labels"].shape == (0,)
    assert make(path, synthetic_root, skip_empty=True).__len__() == 23


def test_boundary_overshoot_is_clipped_and_degenerate_dropped(coco_path, synthetic_root, tmp_path):
    coco = json.loads(coco_path.read_text())
    im = coco["images"][0]
    anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
    anns[0]["bbox"] = [im["width"] - 10, 5, 10.8, 20]  # overshoots the right edge by 0.8 px (inside tolerance)
    anns[0]["segmentation"], anns[0]["area"] = [], 10.8 * 20
    coco["annotations"] = anns
    coco["images"] = [im]
    path = tmp_path / "edge.json"
    path.write_text(json.dumps(coco))
    ds = make(path, synthetic_root, cfg=DataConfig(image_size=im["width"], load_masks=False))
    sample = ds.load_sample(0)
    assert sample.boxes[:, 2].max() <= im["width"] + 1e-6


def test_invalid_coco_is_rejected_at_construction(coco_path, synthetic_root, tmp_path):
    coco = copy.deepcopy(json.loads(coco_path.read_text()))
    coco["annotations"][0]["category_id"] = 4242
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="category_id"):
        make(path, synthetic_root)
    with pytest.raises(KeyError):
        make(coco_path, synthetic_root, image_ids=[10**6])


def test_size_mismatch_between_json_and_file_is_an_error(coco_path, synthetic_root, tmp_path):
    coco = json.loads(coco_path.read_text())
    coco["images"][0]["width"] += 5
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] == coco["images"][0]["id"]]
    for a in coco["annotations"]:
        a.update(segmentation=[], area=a["bbox"][2] * a["bbox"][3])
    coco["images"] = coco["images"][:1]
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="JSON says"):
        make(path, synthetic_root)[0]


def test_draw_annotations_returns_new_image_with_marks():
    img = np.zeros((64, 64, 3), np.uint8)
    out = draw_annotations(
        img, np.array([[8, 8, 40, 40]]), [0], ["a"], scores=[0.9], masks=np.ones((1, 64, 64), np.uint8)
    )
    assert out.shape == img.shape and out.sum() > 0 and img.sum() == 0
