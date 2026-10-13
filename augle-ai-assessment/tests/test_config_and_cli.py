"""Config loading and the collect_and_annotate CLI wiring (models replaced by stubs)."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import yaml

import collect_and_annotate as cna
from augle.annotation import Detection
from augle.config import Config, config_to_dict, load_config
from tests.test_collection import MockCommons, jpeg, meta_json

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_loads_and_roundtrips():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert isinstance(cfg, Config) and cfg.data.image_size == 320 and cfg.train.monitor == "val_loss"
    assert cfg.data.split_fractions == (0.7, 0.15, 0.15) and isinstance(cfg.data.scale_range, tuple)
    assert "robot_arm" in cfg.annotation.categories and cfg.collection.min_long_side == 1024
    assert config_to_dict(cfg)["collection"]["queries"][0] == "industrial robot arm"
    json.dumps(config_to_dict(cfg))
    assert load_config(None) == Config() or load_config(None).data.image_size == 320


def test_yaml_matches_dataclass_defaults_where_declared():
    """configs/default.yaml documents the defaults; they must not silently drift from the code."""
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    defaults = config_to_dict(Config())
    for section, values in raw.items():
        for key, value in values.items():
            if section == "train" and key == "output_dir":
                continue
            assert defaults[section][key] == (list(value) if isinstance(value, list) else value), (section, key)


def test_unknown_or_malformed_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  epochz: 3\n")
    with pytest.raises(ValueError, match="unknown TrainConfig keys"):
        load_config(bad)
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_validate_subcommand_exit_codes(coco_path, synthetic_root, tmp_path, capsys):
    assert cna.main(["validate", "--coco", str(coco_path), "--image-root", str(synthetic_root)]) == 0
    assert "PASSED" in capsys.readouterr().out
    coco = json.loads(coco_path.read_text())
    coco["annotations"][0]["category_id"] = 999
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(coco))
    assert cna.main(["validate", "--coco", str(broken), "--image-root", str(synthetic_root)]) == 1
    assert "FAILED" in capsys.readouterr().out


@pytest.fixture()
def mock_commons():
    MockCommons.files, MockCommons.blobs, MockCommons.seen_user_agents, MockCommons.flaky_hits = {}, {}, [], 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockCommons)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.mark.browser
def test_collect_then_annotate_cli_with_stub_models(mock_commons, tmp_path, monkeypatch):
    """collect (real Chromium, mock server) -> annotate (stub detector/segmenter) -> validated COCO + metadata."""
    base = mock_commons
    for i in range(4):
        title = f"File:Robot {i}.jpg"
        MockCommons.files[title] = meta_json(title, f"{base}/img/{i}.jpg", w=640, h=480)
        MockCommons.blobs[f"/img/{i}.jpg"] = jpeg(300 + i, 480, 640)
    out = tmp_path / "raw"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "collection": {
                    "base_url": base,
                    "queries": ["robot"],
                    "target_images": 4,
                    "results_per_page": 5,
                    "max_pages_per_query": 1,
                    "min_long_side": 600,
                    "min_short_side": 400,
                    "request_delay_s": 0.0,
                    "max_retries": 2,
                    "out_dir": str(out),
                },
                "annotation": {
                    "cache_dir": str(tmp_path / "cache"),
                    "output_json": str(tmp_path / "ann" / "instances.json"),
                },
            }
        )
    )
    monkeypatch.setenv("COLLECTOR_USER_AGENT", "AugleTest/0.1 (pytest)")
    try:
        assert cna.main(["--config", str(cfg_path), "collect"]) == 0
    except Exception as exc:
        if "Executable doesn't exist" in str(exc) or "BrowserType.launch" in str(exc):
            pytest.skip(f"Chromium not available: {exc}")
        raise
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["summary"]["accepted"] == 4 and set(MockCommons.seen_user_agents) == {"AugleTest/0.1 (pytest)"}

    class StubDetector:  # TEST DOUBLE for Grounding DINO
        def __init__(self, *a, **k):
            pass

        def detect(self, image_bgr, prompts):
            h, w = image_bgr.shape[:2]
            return [
                Detection((w * 0.2, h * 0.2, w * 0.6, h * 0.7), 0.9, "robot_arm"),
                Detection((w * 0.55, h * 0.1, w * 0.8, h * 0.3), 0.8, "gripper"),
            ]

    class StubSegmenter:  # TEST DOUBLE for SAM: filled boxes
        def __init__(self, *a, **k):
            pass

        def segment(self, image_bgr, boxes):
            out = []
            for x1, y1, x2, y2 in boxes:
                m = np.zeros(image_bgr.shape[:2], bool)
                m[int(y1) : int(y2), int(x1) : int(x2)] = True
                out.append(m)
            return out

    monkeypatch.setattr(cna, "GroundingDinoDetector", StubDetector)
    monkeypatch.setattr(cna, "SamSegmenter", StubSegmenter)
    assert (
        cna.main(
            ["--config", str(cfg_path), "annotate", "--manifest", str(out / "manifest.json"), "--image-root", str(out)]
        )
        == 0
    )
    coco = json.loads((tmp_path / "ann" / "instances.json").read_text())
    assert (
        len(coco["images"]) == 4
        and len(coco["annotations"]) == 8
        and all(a["segmentation"] for a in coco["annotations"])
    )
    assert (
        coco["images"][0]["license"] == 1
        and coco["licenses"][0]["name"] == "CC BY-SA 4.0"
        and coco["images"][0]["author"] == "Ann & Bo"
    )
    meta = json.loads((tmp_path / "ann" / "dataset_metadata.json").read_text())
    assert meta["collection"]["accepted"] == 4 and meta["annotation"]["annotations"] == 8
    assert cna.main(["validate", "--coco", str(tmp_path / "ann" / "instances.json"), "--image-root", str(out)]) == 0


@pytest.mark.browser
def test_preflight_script_against_mock_server(mock_commons, capsys):
    """scripts/check_commons.py runs the same three live steps; verified here against the local mock server."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_commons", ROOT / "scripts" / "check_commons.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = mock_commons
    assert module.check("robot", base) == 1  # nothing served yet -> step 1 reports no links
    assert "no links matched" in capsys.readouterr().out
    MockCommons.files["File:Robot a.jpg"] = meta_json("File:Robot a.jpg", f"{base}/img/a.jpg", w=640, h=480)
    MockCommons.blobs["/img/a.jpg"] = jpeg(7, 480, 640)
    assert module.check("robot", base) == 0
    out = capsys.readouterr().out
    assert (
        "[1/3]" in out and "[2/3]" in out and "downloaded and decoded File:Robot a.jpg: 640x480" in out and "OK" in out
    )


def test_annotate_fails_fast_on_missing_manifest_without_loading_models(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("models must not be loaded when the manifest is missing")

    monkeypatch.setattr(cna, "GroundingDinoDetector", boom)
    monkeypatch.setattr(cna, "SamSegmenter", boom)
    code = cna.main(["annotate", "--manifest", str(tmp_path / "nope.json"), "--image-root", str(tmp_path)])
    assert code == 2 and "collection manifest not found" in capsys.readouterr().err


def test_annotate_reports_model_download_failure_cleanly(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"accepted": [], "summary": {}}))

    def offline(*a, **k):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr(cna, "GroundingDinoDetector", offline)
    code = cna.main(["annotate", "--manifest", str(manifest), "--image-root", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and "could not load the foundation models" in err and "huggingface.co" in err


@pytest.mark.browser
def test_collect_with_zero_accepted_images_is_a_failure(mock_commons, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "collection": {
                    "base_url": mock_commons,
                    "queries": ["robot"],
                    "max_pages_per_query": 1,
                    "request_delay_s": 0.0,
                    "max_retries": 1,
                    "out_dir": str(tmp_path / "raw"),
                }
            }
        )
    )
    code = cna.main(["--config", str(cfg_path), "collect"])  # the mock serves no files -> nothing can be accepted
    assert code == 2 and "no images were accepted" in capsys.readouterr().err
