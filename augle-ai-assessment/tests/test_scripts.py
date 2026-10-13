"""scripts/summarize_run.py (README fill + synthetic guard) and scripts/run_full_pipeline.py (orchestration)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from train import fit
from tests.test_training import smoke_cfg

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summ = load_script("summarize_run")
pipeline = load_script("run_full_pipeline")


@pytest.fixture(scope="module")
def run_artifacts(tmp_path_factory, synthetic_root):
    """metrics.json from a genuine (synthetic-data) 2-epoch fit, plus hand-built manifest/metadata files."""
    base = tmp_path_factory.mktemp("run")
    res = fit(smoke_cfg(epochs=2), synthetic_root / "instances.json", synthetic_root, base / "run")
    manifest = {
        "source": "wikimedia_commons",
        "summary": {
            "attempted": 300,
            "accepted": 211,
            "rejected_by_reason": {"too_small": 9},
            "accepted_by_license": {"CC BY-SA 4.0": 211},
        },
    }
    meta = {
        "annotation": {
            "images_processed": 211,
            "images_with_detections": 187,
            "images_without_detections": 24,
            "images_failed": 0,
            "annotations": 402,
            "annotations_with_masks": 400,
            "per_category": {"robot_arm": 220, "gripper": 182},
            "detections_dropped": {"box_too_small": 31},
        }
    }
    (base / "manifest.json").write_text(json.dumps(manifest))
    (base / "meta.json").write_text(json.dumps(meta))
    return base, res.metrics_path


def args_for(base: Path, metrics: Path, coco: Path, readme: Path, *extra: str) -> list[str]:
    return [
        "--manifest",
        str(base / "manifest.json"),
        "--dataset-meta",
        str(base / "meta.json"),
        "--coco",
        str(coco),
        "--metrics",
        str(metrics),
        "--llm-transcript",
        str(base / "llm.json"),
        "--readme",
        str(readme),
        *extra,
    ]


def test_summary_contains_exactly_the_numbers_in_the_artifacts(run_artifacts, coco_path):
    base, metrics = run_artifacts
    block = summ.build_summary(
        str(base / "manifest.json"), str(base / "meta.json"), str(coco_path), str(metrics), str(base / "llm.json")
    )
    m = json.loads(metrics.read_text())
    assert "accepted: **211**" in block and "with detections: **187**" in block and "annotations: **402**" in block
    assert (
        f"best checkpoint: epoch **{m['best']['epoch']}**" in block
        and f"{m['model']['trainable_parameters']:,} parameters" in block
    )
    assert f"AP50 {m['test']['ap50']:.3f}" in block
    assert "not available (no real-LLM transcript" in block  # nothing is claimed that has no artifact


def test_missing_artifacts_are_reported_not_invented(tmp_path):
    block = summ.build_summary(*(str(tmp_path / n) for n in ("a.json", "b.json", "c.json", "d.json", "e.json")))
    assert block.count("not available") == 4 and "accepted" not in block


def test_real_llm_transcript_is_listed_but_offline_fallback_is_not(run_artifacts, coco_path, tmp_path):
    base, metrics = run_artifacts
    answer = {
        "question": "Are there suspiciously small bounding boxes?",
        "mode": "llm",
        "provider": "anthropic",
        "model": "m",
        "stop_reason": "answered",
        "tool_trace": [{"name": "analyze_bbox_areas", "ok": True}],
    }
    llm = tmp_path / "llm.json"
    llm.write_text(json.dumps([answer]))
    args = (str(base / "manifest.json"), str(base / "meta.json"), str(coco_path), str(metrics))
    assert "tools executed: `analyze_bbox_areas`" in summ.build_summary(*args, str(llm))
    llm.write_text(json.dumps([{**answer, "mode": "offline-fallback"}]))
    assert "tools executed" not in summ.build_summary(*args, str(llm))


def test_write_readme_refuses_synthetic_data_and_leaves_readme_untouched(run_artifacts, coco_path, tmp_path):
    base, metrics = run_artifacts
    readme = tmp_path / "README.md"
    original = f"intro\n{summ.START}\nplaceholder\n{summ.END}\noutro\n"
    readme.write_text(original)
    assert summ.is_synthetic(json.loads(coco_path.read_text()))
    assert summ.main(args_for(base, metrics, coco_path, readme, "--write-readme")) == 1
    assert readme.read_text() == original


def test_write_readme_replaces_only_the_marked_block_and_is_idempotent(run_artifacts, coco_path, tmp_path):
    base, metrics = run_artifacts
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{summ.START}\nplaceholder\n{summ.END}\noutro\n")
    cli = args_for(base, metrics, coco_path, readme, "--write-readme", "--allow-synthetic")
    assert summ.main(cli) == 0 and summ.main(cli) == 0
    text = readme.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n") and "placeholder" not in text
    assert text.count(summ.START) == 1 and text.count("accepted: **211**") == 1
    with pytest.raises(ValueError, match="markers"):
        summ.splice("no markers here", "x")


def test_real_looking_coco_is_not_flagged_synthetic():
    assert not summ.is_synthetic(
        {"info": {"description": "Industrial robot arm dataset auto-labelled with Grounding DINO (+SAM)"}}
    )
    assert not summ.is_synthetic({})


def test_readme_has_markers_for_the_results_block():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(summ.START) == 1 and text.count(summ.END) == 1 and text.index(summ.START) < text.index(summ.END)


# -------------------------------------------------------------- orchestrator ----
def test_dry_run_lists_every_stage_in_order(capsys):
    assert pipeline.main(["--dry-run", "--epochs", "7", "--provider", "offline", "--annotate-limit", "5"]) == 0
    out = capsys.readouterr().out
    positions = [out.index(f"[{s}]") for s in pipeline.STAGES]
    assert positions == sorted(positions)
    assert (
        "train.py --coco data/annotations/instances.json --image-root data/raw --output-dir artifacts/run --epochs 7"
        in out
    )
    assert "annotate --limit 5" in out and "--provider offline demo" in out and "summarize_run.py --write-readme" in out


def test_stage_selection_and_failure_stops_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1] if len(cmd) > 1 else cmd)
        code = 3 if "collect" in cmd else 0  # the `collect` sub-command (exact argument, not the script name)
        return subprocess.CompletedProcess(cmd, code, stdout="out", stderr="boom" if code else "")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    assert pipeline.main(["--from-stage", "preflight", "--to-stage", "train"]) == 3
    assert [Path(c).name for c in calls] == [
        "check_commons.py",
        "collect_and_annotate.py",
    ]  # stopped after the failing stage
    transcript = (tmp_path / pipeline.TRANSCRIPT).read_text()
    assert "STOPPED: stage 'collect' failed" in transcript and "exit code: 3" in transcript
    assert "STOPPED at stage 'collect'" in capsys.readouterr().err
    calls.clear()
    assert pipeline.main(["--from-stage", "validate", "--to-stage", "train"]) == 0
    assert len(calls) == 2  # validate + train only
