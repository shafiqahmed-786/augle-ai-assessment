"""Shared pytest fixtures (all data here is synthetic and generated on the fly)."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from augle.synthetic import build_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Minimal tensorflow stub so that tensorboard's EventAccumulator works on
# machines where tensorflow is not installed (or is a broken stub).
# tensorboard 2.17 lazy-loads `tf.compat.as_str_any` and `tf.io.gfile.join`
# at import time; without a real tf these raise AttributeError.
# ---------------------------------------------------------------------------
def _ensure_tf_stub() -> None:
    """Inject a minimal fake tensorflow into sys.modules if needed."""
    try:
        import tensorflow as tf  # type: ignore[import-untyped]  # noqa: F401
        # Check that the pieces tensorboard actually uses exist.
        _ = tf.compat.as_str_any
        _ = tf.io.gfile
        return  # real (or good-enough) tensorflow already present
    except (ImportError, AttributeError):
        pass

    # Build a minimal stub that satisfies tensorboard's io_wrapper checks.
    tf_stub = types.ModuleType("tensorflow")

    compat_stub = types.ModuleType("tensorflow.compat")
    compat_stub.as_str_any = lambda x: x if isinstance(x, str) else x.decode("utf-8", errors="replace")

    gfile_stub = types.ModuleType("tensorflow.io.gfile")
    gfile_stub.join = os.path.join

    io_stub = types.ModuleType("tensorflow.io")
    io_stub.gfile = gfile_stub

    tf_stub.compat = compat_stub
    tf_stub.io = io_stub

    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.compat"] = compat_stub
    sys.modules["tensorflow.io"] = io_stub
    sys.modules["tensorflow.io.gfile"] = gfile_stub

    # Reload tensorboard's lazy proxy so it picks up the stub.
    import importlib
    try:
        import tensorboard.lazy as tb_lazy  # type: ignore[import-untyped]
        tb_lazy._cache.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        importlib.invalidate_caches()
    except Exception:
        pass


_ensure_tf_stub()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "browser: needs a launchable Playwright Chromium (skipped otherwise)")
    config.addinivalue_line("markers", "slow: slower smoke tests")


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 24-image synthetic COCO dataset shared by the whole session (read-only)."""
    root = tmp_path_factory.mktemp("synthetic")
    build_synthetic_dataset(root, n_images=24, seed=7, size=(128, 176))
    return root


@pytest.fixture()
def coco_path(synthetic_root: Path) -> Path:
    return synthetic_root / "instances.json"
