#!/usr/bin/env python3
"""Phase 1 entry point: Playwright collection -> zero-shot annotation -> validated COCO JSON.

Sub-commands
------------
collect   Browser-based collection of freely licensed images (Playwright) + provenance manifest.
annotate  Grounding DINO boxes (+ optional SAM masks) -> COCO JSON, validated before returning.
validate  Strict COCO schema / geometry / file validation of an existing annotation file.
all       collect, then annotate, then validate.

Examples
--------
    python collect_and_annotate.py all --config configs/default.yaml
    python collect_and_annotate.py collect --target-images 20 --out-dir data/raw_smoke
    python collect_and_annotate.py validate --coco data/annotations/instances.json --image-root data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from augle.annotation import (
    GroundingDinoDetector,
    SamSegmenter,
    annotate_collection,
    write_and_validate,
)
from augle.coco import load_coco, save_json_atomic, validate_coco
from augle.collection import (
    DEFAULT_USER_AGENT,
    CommonsAdapter,
    ImageCollector,
    PlaywrightSession,
)
from augle.config import Config, config_to_dict, load_config
from augle.utils import load_dotenv, setup_logging

LOGGER = logging.getLogger("collect_and_annotate")


class EmptyCollectionError(RuntimeError):
    """The collector finished without a single usable image."""


class ModelLoadError(RuntimeError):
    """The Grounding DINO / SAM weights could not be loaded (typically a network or cache problem)."""


def run_collect(cfg: Config, out_dir: str | None = None) -> Path:
    """Run the Playwright collector; returns the manifest path."""
    ccfg = cfg.collection
    out = Path(out_dir or ccfg.out_dir)
    session = PlaywrightSession(
        user_agent=os.environ.get("COLLECTOR_USER_AGENT", DEFAULT_USER_AGENT),
        headless=ccfg.headless,
        executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
        timeout_ms=ccfg.page_timeout_ms,
        delay_s=ccfg.request_delay_s,
        max_retries=ccfg.max_retries,
    )
    adapter = CommonsAdapter(ccfg.base_url, ccfg.results_per_page)
    with session:
        summary = ImageCollector(ccfg, session, adapter, out).run()
    LOGGER.info("collection finished: %s", json.dumps(summary.__dict__, indent=2))
    if summary.accepted == 0:
        raise EmptyCollectionError(
            "no images were accepted. Common causes: no network access to the source, blocked requests, selectors "
            "that no longer match (run scripts/check_commons.py), or filters that reject everything "
            "(see 'rejected' in manifest.json)."
        )
    if summary.accepted < 150:
        LOGGER.warning(
            "only %d images accepted (assessment asks for >= 150). Raise collection.target_images / "
            "max_pages_per_query or add queries, then re-run: collection resumes from the manifest.",
            summary.accepted,
        )
    return out / "manifest.json"


def run_annotate(cfg: Config, manifest: str, image_root: str, limit: int | None, use_sam: bool | None) -> Path:
    """Run Grounding DINO (+SAM), build COCO, validate, write dataset metadata."""
    acfg = cfg.annotation
    if use_sam is not None:
        acfg.use_sam = use_sam
    if not Path(manifest).is_file():  # fail before spending minutes on model downloads
        raise FileNotFoundError(
            f"collection manifest not found: {manifest} (run `collect_and_annotate.py collect` first)"
        )
    try:
        detector = GroundingDinoDetector(acfg.detector_model, acfg.box_threshold, acfg.text_threshold, acfg.device)
        segmenter = SamSegmenter(acfg.sam_model, acfg.device) if acfg.use_sam else None
    except OSError as exc:  # transformers raises OSError when weights cannot be downloaded or found in the cache
        raise ModelLoadError(
            f"could not load the foundation models ({type(exc).__name__}: {str(exc)[:200]}). The weights come from "
            "huggingface.co: check connectivity (or allow huggingface.co and its CDN hosts), or pre-download them and "
            "set HF_HOME / HF_HUB_OFFLINE=1."
        ) from exc
    coco, summary = annotate_collection(manifest, image_root, detector, segmenter, acfg, limit=limit)
    out_json = Path(acfg.output_json)
    write_and_validate(coco, out_json, image_root)
    manifest_data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    save_json_atomic(
        {
            "collection": manifest_data.get("summary", {}),
            "collection_source": manifest_data.get("source"),
            "annotation": summary.__dict__,
            "annotation_config": config_to_dict(acfg),
            "note": "Labels are zero-shot model pseudo-labels (not human verified).",
        },
        out_json.parent / "dataset_metadata.json",
    )
    LOGGER.info("annotation finished: %s", json.dumps(summary.__dict__, indent=2, default=str))
    if len(coco["images"]) < 150:
        LOGGER.warning("only %d annotated images (assessment asks for >= 150 usable images)", len(coco["images"]))
    return out_json


def run_validate(coco_path: str, image_root: str | None) -> bool:
    """Validate a COCO file and print the report; returns True when valid."""
    report = validate_coco(load_coco(coco_path), image_root, check_files=image_root is not None)
    print(report.summary())
    return report.ok


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Playwright image collection")
    p_collect.add_argument("--out-dir")
    p_collect.add_argument("--target-images", type=int)

    p_annot = sub.add_parser("annotate", help="zero-shot annotation to COCO")
    p_annot.add_argument("--manifest", default="data/raw/manifest.json")
    p_annot.add_argument("--image-root", default="data/raw")
    p_annot.add_argument("--limit", type=int, help="annotate only the first N images (smoke runs)")
    p_annot.add_argument("--no-sam", action="store_true", help="boxes only, skip SAM masks")

    p_val = sub.add_parser("validate", help="validate a COCO JSON file")
    p_val.add_argument("--coco", default="data/annotations/instances.json")
    p_val.add_argument("--image-root")

    p_all = sub.add_parser("all", help="collect + annotate + validate")
    p_all.add_argument("--no-sam", action="store_true")
    p_all.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    load_dotenv()
    setup_logging(args.log_level, "logs/collect_and_annotate.log")
    cfg = load_config(args.config)
    try:
        if args.command == "collect":
            if args.target_images:
                cfg.collection.target_images = args.target_images
            run_collect(cfg, args.out_dir)
        elif args.command == "annotate":
            run_annotate(cfg, args.manifest, args.image_root, args.limit, False if args.no_sam else None)
        elif args.command == "validate":
            return 0 if run_validate(args.coco, args.image_root) else 1
        elif args.command == "all":
            manifest = run_collect(cfg)
            run_annotate(cfg, str(manifest), cfg.collection.out_dir, args.limit, False if args.no_sam else None)
    except (EmptyCollectionError, ModelLoadError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
