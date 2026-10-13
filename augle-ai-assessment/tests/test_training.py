"""Model, loss, targets, decoding, metrics, optimisation utilities, checkpoints and a smoke fit."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch

from augle.config import Config
from train import (
    STRIDE,
    AugleNet,
    CenterNetLoss,
    EarlyStopping,
    average_precision,
    build_optimizer,
    build_scheduler,
    count_parameters,
    decode_predictions,
    encode_targets,
    evaluate,
    evaluate_detections,
    fit,
    focal_loss,
    gaussian_radius,
    json_safe,
    load_checkpoint,
    make_tracker,
    masked_l1,
    save_checkpoint,
    seed_everything,
    select_device,
)


def tiny_batch(bsz: int = 2, size: int = 64, classes: int = 2):
    boxes = [torch.tensor([[8.0, 8.0, 40.0, 24.0]]), torch.tensor([[10.0, 20.0, 30.0, 60.0], [40.0, 4.0, 60.0, 20.0]])][
        :bsz
    ]
    labels = [torch.tensor([0]), torch.tensor([1, 0])][:bsz]
    images = torch.randn(bsz, 3, size, size)
    return images, boxes, labels


# ----------------------------------------------------------------- model ----
def test_forward_shapes_and_nonsquare_input():
    model = AugleNet(3, base_width=16)
    out = model(torch.randn(2, 3, 64, 96))
    assert out["heatmap"].shape == (2, 3, 16, 24)
    assert out["wh"].shape == (2, 2, 16, 24) and out["offset"].shape == (2, 2, 16, 24)
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(1, 3, 70, 64))


def test_custom_architecture_has_residual_skips_and_no_pretrained_backbone():
    import inspect

    import train as train_module
    from train import ResidualBlock

    model = AugleNet(2, base_width=16)
    blocks = [m for m in model.modules() if isinstance(m, ResidualBlock)]
    assert len(blocks) == 1 + 2 + 3 + 2
    assert any(isinstance(b.shortcut, torch.nn.Identity) for b in blocks)  # identity skips exist
    assert any(not isinstance(b.shortcut, torch.nn.Identity) for b in blocks)  # projection skips exist
    assert count_parameters(model) > 10_000
    source = inspect.getsource(train_module)
    assert "torchvision" not in source and "pretrained" not in inspect.getsource(AugleNet).lower()


def test_heatmap_head_prior_bias_and_small_output_init():
    model = AugleNet(2, base_width=16)
    assert torch.allclose(model.head_hm[-1].bias, torch.full_like(model.head_hm[-1].bias, -2.19))
    assert float(model.head_wh[-1].weight.detach().std()) < 0.05


def test_train_and_eval_modes_control_batchnorm_statistics():
    model = AugleNet(2, base_width=16)
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64) * 3 + 1)
    assert torch.equal(bn.running_mean, before)  # eval: statistics frozen
    model.train()
    model(torch.randn(2, 3, 64, 64) * 3 + 1)
    assert not torch.equal(bn.running_mean, before)  # train: statistics updated


# --------------------------------------------------------------- targets ----
def test_encode_targets_places_gaussian_peak_wh_offset_and_index():
    boxes, labels = [torch.tensor([[8.0, 12.0, 40.0, 28.0]])], [torch.tensor([1])]
    t = encode_targets(boxes, labels, 2, 16, 16)
    cx, cy = 24 / STRIDE, 20 / STRIDE  # 6.0, 5.0 in grid units
    assert t["heatmap"].shape == (1, 2, 16, 16)
    assert t["heatmap"][0, 1, 5, 6] == 1.0 and t["heatmap"][0, 0].sum() == 0
    assert t["heatmap"][0, 1].max() == 1.0 and (t["heatmap"][0, 1] == 1.0).sum() == 1
    assert t["wh"][0, 0].tolist() == [8.0, 4.0]
    assert t["offset"][0, 0].tolist() == [cx - 6, cy - 5]
    assert int(t["index"][0, 0]) == 5 * 16 + 6 and t["mask"][0].sum() == 1


def test_encode_targets_handles_empty_images_and_degenerate_boxes():
    t = encode_targets(
        [torch.zeros(0, 4), torch.tensor([[5.0, 5.0, 5.0, 9.0]])], [torch.zeros(0, dtype=torch.long)] * 2, 2, 8, 8
    )
    assert t["heatmap"].sum() == 0 and t["mask"].sum() == 0


def test_gaussian_radius_grows_with_object_size():
    assert gaussian_radius((4, 4)) < gaussian_radius((20, 20)) < gaussian_radius((60, 60))


def test_decode_inverts_encode():
    """Perfect network output built from the targets must decode back to the GT boxes."""
    boxes = [torch.tensor([[8.0, 12.0, 40.0, 28.0], [44.0, 36.0, 60.0, 60.0]])]
    labels = [torch.tensor([0, 1])]
    t = encode_targets(boxes, labels, 2, 16, 16)
    hm = t["heatmap"].clamp(1e-4, 1 - 1e-4)
    out = {"heatmap": torch.log(hm / (1 - hm)), "wh": torch.zeros(1, 2, 16, 16), "offset": torch.zeros(1, 2, 16, 16)}
    for k in range(2):
        idx = int(t["index"][0, k])
        out["wh"][0, :, idx // 16, idx % 16] = t["wh"][0, k]
        out["offset"][0, :, idx // 16, idx % 16] = t["offset"][0, k]
    det = decode_predictions(out, 64, 0.5, 10)[0]
    assert len(det["boxes"]) == 2
    order = np.argsort(det["labels"])
    assert np.allclose(det["boxes"][order], boxes[0].numpy(), atol=1e-3)
    assert det["labels"][order].tolist() == [0, 1]


# ------------------------------------------------------------------ loss ----
def test_losses_are_finite_and_behave_sensibly():
    _, boxes, labels = tiny_batch()
    t = encode_targets(boxes, labels, 2, 16, 16)
    rng = torch.Generator().manual_seed(0)
    out = {
        "heatmap": torch.randn(2, 2, 16, 16, generator=rng),
        "wh": torch.randn(2, 2, 16, 16, generator=rng),
        "offset": torch.rand(2, 2, 16, 16, generator=rng),
    }
    losses = CenterNetLoss()(out, t)
    assert all(torch.isfinite(v) for v in losses.values())
    # better heat-map predictions -> lower focal loss
    good = torch.where(t["heatmap"] == 1, torch.tensor(6.0), torch.tensor(-6.0))
    assert focal_loss(good, t["heatmap"]) < focal_loss(out["heatmap"], t["heatmap"])
    # no objects at all: focal loss stays finite (pure background), L1 terms are exactly zero
    empty = encode_targets([torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)], 2, 16, 16)
    e = CenterNetLoss()(
        {"heatmap": torch.randn(1, 2, 16, 16), "wh": torch.randn(1, 2, 16, 16), "offset": torch.randn(1, 2, 16, 16)},
        empty,
    )
    assert torch.isfinite(e["total"]) and e["wh"] == 0 and e["off"] == 0
    assert masked_l1(out["wh"], t["wh"], t["index"], torch.zeros_like(t["mask"])) == 0


def test_backward_gives_finite_gradients_everywhere_and_optimizer_updates_weights():
    seed_everything(0)
    model = AugleNet(2, base_width=16)
    images, boxes, labels = tiny_batch()
    tgt = encode_targets(boxes, labels, 2, 16, 16)
    opt = build_optimizer(model, 1e-3, 1e-4)
    before = copy.deepcopy(model.state_dict())
    opt.zero_grad()
    loss = CenterNetLoss()(model(images), tgt)["total"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum() > 0) for g in grads) / len(grads) > 0.9  # gradients reach (nearly) every tensor
    opt.step()
    changed = [
        not torch.equal(before[k], v)
        for k, v in model.state_dict().items()
        if "running" not in k and "num_batches" not in k
    ]
    assert all(changed)


def test_optimizer_applies_weight_decay_only_to_weights():
    model = AugleNet(2, base_width=16)
    opt = build_optimizer(model, 1e-3, 0.5)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.5 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decay["params"]) and all(p.ndim <= 1 for p in no_decay["params"])


def test_overfits_a_single_batch():
    """Sanity check of the whole forward/loss/backward/step chain: loss must drop a lot."""
    seed_everything(0)
    model, loss_fn = AugleNet(2, base_width=16), CenterNetLoss()
    images, boxes, labels = tiny_batch()
    tgt = encode_targets(boxes, labels, 2, 16, 16)
    opt = build_optimizer(model, 3e-3, 0.0)
    first = last = 0.0
    for _step in range(40):
        opt.zero_grad()
        loss = loss_fn(model(images), tgt)["total"]
        loss.backward()
        opt.step()
        first = first or loss.item()
        last = loss.item()
    assert last < 0.5 * first, (first, last)


def test_device_selection():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested"):
            select_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in this environment")
def test_forward_backward_on_cuda():  # pragma: no cover - only runs on GPU machines
    model = AugleNet(2, base_width=16).cuda()
    images, boxes, labels = tiny_batch()
    tgt = {k: v.cuda() for k, v in encode_targets(boxes, labels, 2, 16, 16).items()}
    CenterNetLoss()(model(images.cuda()), tgt)["total"].backward()
    assert all(p.grad.is_cuda for p in model.parameters())


# ------------------------------------------------------ schedule / stopping ----
def test_scheduler_warms_up_then_decays():
    model = AugleNet(2, base_width=16)
    opt = build_optimizer(model, 1e-2, 0.0)
    sched = build_scheduler(opt, epochs=10, warmup_epochs=3)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[1] < lrs[2] <= 1e-2 + 1e-12
    assert lrs[3] >= lrs[5] > lrs[8] and lrs[-1] < 0.2 * 1e-2
    no_warm = build_scheduler(build_optimizer(AugleNet(2, base_width=16), 1e-2, 0.0), 5, 0)
    assert no_warm.get_last_lr()[0] == pytest.approx(1e-2)


def test_early_stopping_min_mode_patience_and_best_tracking():
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
    flags = [es.step(v, e) for e, v in enumerate([1.0, 0.8, 0.795, 0.9, 0.85], start=1)]
    assert flags == [True, True, False, False, False]  # 0.795 is not a >=0.01 improvement over 0.8
    assert es.best == 0.8 and es.best_epoch == 2 and es.num_bad == 3 and es.should_stop


def test_early_stopping_max_mode_and_nan_handling():
    es = EarlyStopping(patience=2, mode="max")
    assert es.step(0.1, 1) and es.step(0.3, 2) and not es.step(float("nan"), 3)
    assert not es.should_stop and not es.step(0.2, 4) and es.should_stop
    with pytest.raises(ValueError):
        EarlyStopping(1, mode="sideways")


def test_early_stopping_state_roundtrip():
    a = EarlyStopping(5)
    a.step(1.0, 1), a.step(2.0, 2)
    b = EarlyStopping(5)
    b.load_state_dict(a.state_dict())
    assert (b.best, b.best_epoch, b.num_bad) == (1.0, 1, 1)


# ---------------------------------------------------------------- metrics ----
def _gt(boxes, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "labels": np.array(labels, int)}


def _pred(boxes, scores, labels):
    return {
        "boxes": np.array(boxes, float).reshape(-1, 4),
        "scores": np.array(scores, float),
        "labels": np.array(labels, int),
    }


def test_ap_perfect_none_and_partial():
    gts = [_gt([[0, 0, 10, 10]], [0]), _gt([[20, 20, 40, 40]], [0])]
    perfect = [_pred([[0, 0, 10, 10]], [0.9], [0]), _pred([[20, 20, 40, 40]], [0.8], [0])]
    assert evaluate_detections(perfect, gts, 1)["ap50"] == pytest.approx(1.0)
    assert evaluate_detections(perfect, gts, 1)["ap"] == pytest.approx(1.0)
    none = [_pred([], [], []), _pred([], [], [])]
    assert evaluate_detections(none, gts, 1)["ap50"] == 0.0
    half = [perfect[0], _pred([[100, 100, 120, 120]], [0.8], [0])]
    assert evaluate_detections(half, gts, 1)["ap50"] == pytest.approx(0.5, abs=0.01)
    assert math.isnan(average_precision(np.array([]), np.array([]), 0))


def test_classes_without_ground_truth_are_ignored_in_the_mean():
    gts = [_gt([[0, 0, 10, 10]], [0])]
    preds = [_pred([[0, 0, 10, 10]], [0.9], [0])]
    res = evaluate_detections(preds, gts, 2)
    assert res["ap50"] == pytest.approx(1.0) and res["per_class_ap50"][1] is None


def test_ap_matches_pycocotools_cocoeval():
    """Independent check of our AP implementation against the reference COCOeval."""
    pycocotools = pytest.importorskip("pycocotools.coco")
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(0)
    images, anns, dets, gts, preds = [], [], [], [], []
    ann_id = 1
    for img_id in range(1, 21):
        images.append({"id": img_id, "width": 200, "height": 200})
        g_boxes, g_labels, p_boxes, p_scores, p_labels = [], [], [], [], []
        for _ in range(int(rng.integers(0, 5))):
            x, y = rng.uniform(0, 140, 2)
            w, h = rng.uniform(10, 60, 2)
            c = int(rng.integers(0, 3))
            g_boxes.append([x, y, x + w, y + h]), g_labels.append(c)
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            if rng.random() < 0.8:  # detected, with localisation noise
                n = rng.normal(0, 4, 4)
                p_boxes.append([x + n[0], y + n[1], x + w + n[2], y + h + n[3]]), p_labels.append(c)
                p_scores.append(float(rng.uniform(0.3, 1.0)))
        for _ in range(int(rng.integers(0, 3))):  # false positives
            x, y = rng.uniform(0, 140, 2)
            (
                p_boxes.append([x, y, x + 30, y + 30]),
                p_labels.append(int(rng.integers(0, 3))),
                p_scores.append(float(rng.uniform(0, 0.7))),
            )
        gts.append(_gt(g_boxes, g_labels))
        preds.append(_pred(p_boxes, p_scores, p_labels))
        for b, s, c in zip(p_boxes, p_scores, p_labels):
            dets.append(
                {"image_id": img_id, "category_id": c + 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
            )
    api = pycocotools.COCO()
    api.dataset = {"images": images, "annotations": anns, "categories": [{"id": i, "name": str(i)} for i in (1, 2, 3)]}
    api.createIndex()
    ev = COCOeval(api, api.loadRes(dets), "bbox")
    ev.evaluate(), ev.accumulate(), ev.summarize()
    ours = evaluate_detections(preds, gts, 3)
    assert ours["ap"] == pytest.approx(float(ev.stats[0]), abs=2e-3)
    assert ours["ap50"] == pytest.approx(float(ev.stats[1]), abs=2e-3)


def test_json_safe_converts_nan_inf_and_numpy():
    safe = json_safe({"a": float("nan"), "b": np.float32(1.5), "c": np.int64(3), "d": [float("inf"), 1]})
    assert safe == {"a": None, "b": 1.5, "c": 3, "d": [None, 1]}
    json.dumps(safe, allow_nan=False)


# ------------------------------------------------------------ checkpoints ----
def test_checkpoint_roundtrip_reproduces_outputs(tmp_path):
    seed_everything(1)
    model = AugleNet(2, base_width=16)
    opt = build_optimizer(model, 1e-3, 1e-4)
    sched = build_scheduler(opt, 5, 1)
    es = EarlyStopping(3)
    es.step(0.5, 1)
    x = torch.randn(2, 3, 64, 64)
    model.eval()
    ref = model(x)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, opt, sched, 3, es, {"class_names": ["a", "b"]})
    assert not list(tmp_path.glob("*.tmp"))
    clone = AugleNet(2, base_width=16)
    opt2 = build_optimizer(clone, 1e-3, 1e-4)
    sched2 = build_scheduler(opt2, 5, 1)
    ckpt = load_checkpoint(path, clone, opt2, sched2)
    clone.eval()
    out = clone(x)
    assert all(torch.equal(ref[k], out[k]) for k in ref)
    assert ckpt["epoch"] == 3 and ckpt["meta"]["class_names"] == ["a", "b"] and ckpt["early_stopping"]["best"] == 0.5
    assert opt2.state_dict()["param_groups"][0]["lr"] == opt.state_dict()["param_groups"][0]["lr"]


def test_checkpoint_rejects_arbitrary_pickles(tmp_path):
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    path = tmp_path / "evil.pt"
    path.write_bytes(pickle.dumps({"model": Evil()}))
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path, AugleNet(2, base_width=16))


# --------------------------------------------------------- end-to-end fit ----
def smoke_cfg(tracker: str = "none", epochs: int = 2) -> Config:
    cfg = Config()
    cfg.data.image_size = 64
    cfg.train.epochs, cfg.train.batch_size, cfg.train.base_width = epochs, 4, 16
    cfg.train.tracker, cfg.train.patience, cfg.train.warmup_epochs, cfg.train.num_log_images = tracker, 5, 1, 2
    return cfg


@pytest.mark.slow
def test_end_to_end_smoke_fit_writes_all_artifacts(coco_path, synthetic_root, tmp_path):
    run = tmp_path / "run"
    res = fit(smoke_cfg("tensorboard", 2), coco_path, synthetic_root, run)
    for name in ("best.pt", "last.pt", "metrics.json", "history.jsonl", "splits.json"):
        assert (run / name).is_file(), name
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["epochs_completed"] == 2 and len(metrics["history"]) == 2
    assert {"train_loss", "val_loss", "val_ap50", "val_ap", "lr"} <= set(metrics["history"][0])
    assert metrics["best"]["epoch"] in (1, 2) and metrics["test"] is not None
    assert metrics["dataset"]["split_sizes"] == {"train": 16, "val": 4, "test": 4}
    assert len((run / "history.jsonl").read_text().splitlines()) == 2
    assert len(list((run / "samples").glob("epoch_*_augmented.jpg"))) == 2
    assert res.epochs_run == 2
    # splits are disjoint: validation images never appear in training
    splits = json.loads((run / "splits.json").read_text())
    assert not set(splits["train"]["image_ids"]) & set(splits["val"]["image_ids"])
    assert not set(splits["train"]["image_ids"]) & set(splits["test"]["image_ids"])
    # TensorBoard really received scalars and augmented sample images at every epoch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run / "tensorboard"), size_guidance={"images": 0, "scalars": 0})
    acc.Reload()
    assert "train_loss" in acc.Tags()["scalars"] and "val_loss" in acc.Tags()["scalars"]
    assert len(acc.Scalars("train_loss")) == 2
    image_tags = acc.Tags()["images"]
    assert any(t.startswith("augmented_train_samples") for t in image_tags)
    assert len(acc.Images(sorted(t for t in image_tags if t.startswith("augmented_train_samples"))[0])) == 2


@pytest.mark.slow
def test_runs_are_reproducible_with_same_seed(coco_path, synthetic_root, tmp_path):
    a = fit(smoke_cfg(epochs=2), coco_path, synthetic_root, tmp_path / "a")
    b = fit(smoke_cfg(epochs=2), coco_path, synthetic_root, tmp_path / "b")
    ha = json.loads(a.metrics_path.read_text())["history"]
    hb = json.loads(b.metrics_path.read_text())["history"]
    assert [h["train_loss"] for h in ha] == [h["train_loss"] for h in hb]
    assert [h["val_loss"] for h in ha] == [h["val_loss"] for h in hb]


@pytest.mark.slow
def test_early_stopping_triggers_in_fit_and_resume_continues(coco_path, synthetic_root, tmp_path):
    cfg = smoke_cfg(epochs=6)
    cfg.train.patience, cfg.train.min_delta, cfg.train.lr = (
        1,
        1e6,
        1e-3,
    )  # nothing can improve by 1e6 -> stop after epoch 2
    res = fit(cfg, coco_path, synthetic_root, tmp_path / "es")
    metrics = json.loads(res.metrics_path.read_text())
    assert metrics["early_stopping"]["triggered"] and metrics["epochs_completed"] == 2
    cfg2 = smoke_cfg(epochs=3)
    res2 = fit(cfg2, coco_path, synthetic_root, tmp_path / "es", resume=tmp_path / "es" / "last.pt")
    assert json.loads(res2.metrics_path.read_text())["epochs_completed"] >= 3


def test_evaluate_never_changes_weights(coco_path, synthetic_root):
    from dataset import CocoDetectionDataset, build_dataloader

    cfg = smoke_cfg()
    ds = CocoDetectionDataset(coco_path, synthetic_root, cfg=cfg.data, train=False)
    loader = build_dataloader(ds, 4, shuffle=False, seed=0)
    model = AugleNet(2, base_width=16)
    before = copy.deepcopy(model.state_dict())
    evaluate(model, loader, CenterNetLoss(), torch.device("cpu"), 2, 64, 0.05, 20)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_wandb_tracker_works_offline(tmp_path, monkeypatch):
    """W&B integration verified without network: WANDB_MODE=offline writes a local run file."""
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    tracker = make_tracker("wandb", tmp_path, "pytest-run", {"lr": 0.1})
    tracker.log_scalars(1, {"train_loss": 1.5, "val_loss": float("nan"), "val_ap50": 0.1})  # NaN is skipped, not logged
    tracker.log_images("augmented_train_samples", 1, [np.zeros((32, 32, 3), np.uint8)])
    tracker.close()
    assert list(tmp_path.glob("wandb/offline-run-*/run-*.wandb"))


def test_make_tracker_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="unknown tracker"):
        make_tracker("mlflow", tmp_path, "r", {})
    assert make_tracker("none", tmp_path, "r", {}).__class__.__name__ == "NullTracker"
