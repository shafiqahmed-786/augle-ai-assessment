"""Small cross-cutting helpers: logging, ``.env`` loading, hashing, environment info."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure structured, timestamped logging to stderr and (optionally) a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def load_dotenv(path: str | Path = ".env") -> bool:
    """Load ``KEY=VALUE`` lines from ``path`` into ``os.environ`` without overriding existing vars.

    Returns True when a file was read. Values are never logged.
    """
    p = Path(path)
    if not p.is_file():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment_info() -> dict[str, Any]:
    """Interpreter / platform / library versions recorded with every training run."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "numpy", "cv2"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = None
    return info
