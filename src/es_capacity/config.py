"""Filesystem configuration shared by the experiment runners.

Environment variables are read when a path is requested, not when the package
is imported. Importing this module does not create directories or load models.
"""

from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """Local checkpoint mirror, configured by MODELS_DIR (default: ~/models)."""
    return Path(os.environ.get("MODELS_DIR", Path.home() / "models"))


def ethics_data_dir() -> Path:
    """Unpacked ETHICS subsets; retain the original /tmp default."""
    return Path(os.environ.get("ETHICS_DATA_DIR", "/tmp/ethics/ethics"))


def model_config_path(path: Path | None = None) -> Path | None:
    """Explicit configuration takes precedence over ES_CAPACITY_CONFIG."""
    selected = path if path is not None else os.environ.get("ES_CAPACITY_CONFIG")
    return Path(selected).expanduser().resolve() if selected is not None else None
