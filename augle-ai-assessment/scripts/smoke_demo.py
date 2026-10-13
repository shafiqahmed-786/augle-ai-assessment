#!/usr/bin/env python3
"""End-to-end SMOKE demo on SYNTHETIC data (no network, no model weights, no API key needed).

Runs the real command-line entry points in sequence:

    1. generate a synthetic COCO dataset (procedural images, exact masks)   [augle/synthetic.py]
    2. collect_and_annotate.py validate   (strict COCO validation incl. real image sizes)
    3. train.py                           (custom CNN, manual loop, TensorBoard, early stopping)
    4. llm_auditor.py demo                (RAG + tools; offline fallback unless an API key is set)

Everything printed is measured on the synthetic data and must never be reported as a result of the
real assessment dataset. Transcripts are saved under ``artifacts/demo_outputs/``.

Usage:  python scripts/smoke_demo.py [--images 60] [--epochs 40] [--provider auto|offline|anthropic|openai]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd: list[str], log: list[str]) -> str:
    """Run a command from the repo root, echo it, and keep the transcript."""
    shown = " ".join(["python" if c == sys.executable else c for c in cmd])
    print(f"\n$ {shown}", flush=True)
    log.append(f"$ {shown}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    print(out[-6000:], flush=True)
    log.append(out)
    if proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}: {shown}")
    return out


def main() -> int:
    """Run the four stages and save transcripts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--provider", default="auto", choices=["auto", "offline", "anthropic", "openai"])
    args = ap.parse_args()

    from augle.synthetic import build_synthetic_dataset

    syn, run_dir, out_dir = (
        ROOT / "data" / "synthetic",
        ROOT / "artifacts" / "smoke_run",
        ROOT / "artifacts" / "demo_outputs",
    )
    for d in (syn, run_dir):
        shutil.rmtree(d, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = ["SMOKE DEMO ON SYNTHETIC DATA - not results of the real assessment dataset", ""]
    coco = build_synthetic_dataset(syn, n_images=args.images, seed=5, size=(160, 224))
    log.append(f"[1] synthetic dataset written to {coco.relative_to(ROOT)}")
    py = sys.executable
    rel = lambda path: Path(path).relative_to(ROOT).as_posix()  # noqa: E731 - repo-relative, copy-pasteable commands
    run([py, "collect_and_annotate.py", "validate", "--coco", rel(coco), "--image-root", rel(syn)], log)
    run(
        [
            py,
            "train.py",
            "--coco",
            rel(coco),
            "--image-root",
            rel(syn),
            "--output-dir",
            rel(run_dir),
            "--epochs",
            str(args.epochs),
            "--image-size",
            str(args.image_size),
            "--batch-size",
            "8",
            "--seed",
            "42",
            "--tracker",
            "tensorboard",
        ],
        log,
    )
    run(
        [
            py,
            "llm_auditor.py",
            "--coco",
            rel(coco),
            "--metrics",
            rel(run_dir / "metrics.json"),
            "--dataset-meta",
            rel(syn / "none.json"),
            "--index-dir",
            rel(run_dir / "audit_index"),
            "--provider",
            args.provider,
            "demo",
            "--out-dir",
            rel(out_dir),
        ],
        log,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    summary = {
        "WARNING": "SYNTHETIC smoke-test data; not results on the real dataset",
        "dataset": metrics["dataset"],
        "model": metrics["model"],
        "best": metrics["best"],
        "early_stopping": metrics["early_stopping"],
        "epochs_completed": metrics["epochs_completed"],
        "test": metrics["test"],
        "history_first_last": [metrics["history"][0], metrics["history"][-1]],
    }
    (out_dir / "smoke_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy(run_dir / "metrics.json", out_dir / "smoke_metrics.json")
    last = sorted((run_dir / "samples").glob("epoch_*_augmented.jpg"))[-1]
    shutil.copy(last, out_dir / "smoke_augmented_samples_last_epoch.jpg")
    (out_dir / "smoke_demo_transcript.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\nSaved transcripts and small artifacts to {out_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
