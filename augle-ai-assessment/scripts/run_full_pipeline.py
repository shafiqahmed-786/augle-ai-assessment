#!/usr/bin/env python3
"""Run the REAL assessment workflow end to end on your machine and record a transcript.

Stages (each is the documented CLI command; the run stops at the first failure and says which one):

    1 preflight   scripts/check_commons.py            live Commons + Chromium check
    2 collect     collect_and_annotate.py collect      Playwright -> data/raw (resumable)
    3 annotate    collect_and_annotate.py annotate     Grounding DINO (+SAM) -> validated COCO
    4 validate    collect_and_annotate.py validate     strict COCO validation with real file sizes
    5 train       train.py                              custom CNN on the real COCO file
    6 audit       llm_auditor.py demo                   real LLM if a key is configured, else labelled fallback
    7 summarize   scripts/summarize_run.py --write-readme   fills README section 10 from the artifacts

Usage:
    python scripts/run_full_pipeline.py                 # everything (long: collection + model downloads + training)
    python scripts/run_full_pipeline.py --dry-run       # print the commands only
    python scripts/run_full_pipeline.py --from-stage annotate --epochs 60 --provider anthropic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["preflight", "collect", "annotate", "validate", "train", "audit", "summarize"]
TRANSCRIPT = Path("artifacts/demo_outputs/real_run_transcript.txt")


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    """The exact command for every stage (relative paths, repo root as working directory)."""
    py = sys.executable
    audit = [py, "llm_auditor.py", "--provider", args.provider, "demo", "--out-dir", "artifacts/demo_outputs/real"]
    annotate = [py, "collect_and_annotate.py", "annotate"] + (
        ["--limit", str(args.annotate_limit)] if args.annotate_limit else []
    )
    return {
        "preflight": [py, "scripts/check_commons.py"],
        "collect": [py, "collect_and_annotate.py", "collect"],
        "annotate": annotate,
        "validate": [
            py,
            "collect_and_annotate.py",
            "validate",
            "--coco",
            "data/annotations/instances.json",
            "--image-root",
            "data/raw",
        ],
        "train": [
            py,
            "train.py",
            "--coco",
            "data/annotations/instances.json",
            "--image-root",
            "data/raw",
            "--output-dir",
            "artifacts/run",
            "--epochs",
            str(args.epochs),
            "--tracker",
            args.tracker,
        ],
        "audit": audit,
        "summarize": [py, "scripts/summarize_run.py", "--write-readme"],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the selected stages in order; returns the first failing exit code (or 0)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-stage", choices=STAGES, default="preflight")
    ap.add_argument("--to-stage", choices=STAGES, default="summarize")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tracker", default="tensorboard", choices=["tensorboard", "wandb", "none"])
    ap.add_argument("--provider", default="auto", choices=["auto", "anthropic", "openai", "offline"])
    ap.add_argument("--annotate-limit", type=int, help="annotate only N images (quick model trial)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    commands = build_commands(args)
    selected = STAGES[STAGES.index(args.from_stage) : STAGES.index(args.to_stage) + 1]
    transcript = [f"REAL RUN transcript, started {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    for stage in selected:
        shown = " ".join(["python" if c == sys.executable else c for c in commands[stage]])
        print(f"\n=== [{stage}] $ {shown}", flush=True)
        transcript.append(f"=== [{stage}] $ {shown}")
        if args.dry_run:
            continue
        proc = subprocess.run(commands[stage], cwd=ROOT, capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        print(out[-4000:], flush=True)
        transcript += [out, f"exit code: {proc.returncode}", ""]
        if proc.returncode != 0:
            transcript.append(f"STOPPED: stage '{stage}' failed")
            _save(transcript)
            print(
                f"\nSTOPPED at stage '{stage}' (exit code {proc.returncode}). Transcript: {TRANSCRIPT}", file=sys.stderr
            )
            return proc.returncode
    if not args.dry_run:
        _save(transcript)
        print(f"\nAll selected stages finished. Transcript: {TRANSCRIPT}")
    return 0


def _save(lines: list[str]) -> None:
    path = ROOT / TRANSCRIPT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
