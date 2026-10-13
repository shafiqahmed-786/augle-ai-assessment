#!/usr/bin/env python3
"""Preflight for the live collector: 1 search page + 1 metadata call + 1 download, in about 20 seconds.

Run this on your own machine BEFORE the long collection. It exercises exactly the code the collector
uses (Playwright rendering of the search page, the MediaWiki imageinfo API, a real image download)
and tells you which step breaks if Wikimedia's markup or your network differs from what the code expects.

Usage:  python scripts/check_commons.py [--query "industrial robot arm"] [--base-url https://commons.wikimedia.org]
Exit code 0 = all steps worked, 1 = something failed (the message says what).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augle.collection import DEFAULT_USER_AGENT, CommonsAdapter, FetchError, PlaywrightSession  # noqa: E402
from augle.imaging import ImageReadError, decode_image  # noqa: E402
from augle.utils import load_dotenv  # noqa: E402


def check(query: str, base_url: str) -> int:
    """Run the three preflight steps; returns a process exit code."""
    load_dotenv(ROOT / ".env")
    adapter = CommonsAdapter(base_url, results_per_page=10)
    session = PlaywrightSession(
        user_agent=os.environ.get("COLLECTOR_USER_AGENT", DEFAULT_USER_AGENT),
        executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
        delay_s=0.5,
        max_retries=2,
    )
    try:
        with session:
            titles = adapter.search_titles(session, query, 0)
            print(f"[1/3] search page rendered in Chromium: {len(titles)} File: links found")
            if not titles:
                print(
                    "      -> no links matched the selector in CommonsAdapter.search_titles (augle/collection.py); "
                    "the search markup differs from what the code expects, or the page was blocked."
                )
                return 1
            metas = adapter.fetch_metadata(session, titles[:5])
            print(f"[2/3] imageinfo API returned metadata for {len(metas)} of {len(titles[:5])} titles")
            if not metas:
                return 1
            for m in metas[:3]:
                print(f"      {m.title}: {m.width}x{m.height} {m.mime} | licence={m.license_short!r}")
                print(f"        author={m.author[:40]!r}")
            first = next((m for m in metas if m.mime.startswith("image/") and m.size_bytes < 8_000_000), metas[0])
            img = decode_image(session.get_bytes(first.original_url))
            print(f"[3/3] downloaded and decoded {first.title}: {img.shape[1]}x{img.shape[0]} px")
    except (FetchError, ImageReadError) as exc:
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:  # e.g. Chromium not installed
        print(f"FAILED to run the browser: {type(exc).__name__}: {exc}\nTry: playwright install chromium")
        return 1
    print("OK - the live collector path works from this machine.")
    return 0


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default="industrial robot arm")
    ap.add_argument("--base-url", default="https://commons.wikimedia.org")
    args = ap.parse_args()
    return check(args.query, args.base_url)


if __name__ == "__main__":
    sys.exit(main())
