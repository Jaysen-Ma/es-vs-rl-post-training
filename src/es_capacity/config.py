"""Filesystem configuration shared by the experiment runners.

Configuration and fallback environment variables are read when a path is
requested. Importing this module does not create directories or load models.
"""

from __future__ import annotations

import os
from pathlib import Path


class ModelConfigurationError(ValueError):
    """Invalid shared configuration, reported before model execution."""


def load_config(path: Path) -> dict[str, dict[str, Path]]:
    """Load model and workflow paths relative to a single TOML file."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    path = path.expanduser().resolve()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ModelConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    if "models" not in data or set(data) - {"models", "paths"}:
        raise ModelConfigurationError(f"{path}: expected a [models] table and optional [paths] table")
    resolved = {}
    for section in ("models", "paths"):
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ModelConfigurationError(f"{path}: [{section}] must be a table")
        if section == "paths":
            unknown = set(values) - {"math_eval_dir", "ethics_data_dir"}
            if unknown:
                raise ModelConfigurationError(f"{path}: unknown path settings: {', '.join(sorted(unknown))}")
        resolved[section] = {}
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ModelConfigurationError(f"{path}: {section}.{name} must be a nonempty local path string")
            local = Path(value).expanduser()
            resolved[section][name] = (local if local.is_absolute() else path.parent / local).resolve()
    return resolved


def configured_path(name: str, config: Path | None = None) -> Path | None:
    selected = model_config_path(config)
    return load_config(selected)["paths"].get(name) if selected is not None else None


def models_dir() -> Path:
    """Local checkpoint mirror, configured by MODELS_DIR (default: ~/models)."""
    return Path(os.environ.get("MODELS_DIR", Path.home() / "models"))


def ethics_data_dir(
    config: Path | None = None, *, default: Path | None = Path("/tmp/ethics/ethics"),
) -> Path | None:
    """Unpacked ETHICS root, with the existing environment/default fallback."""
    local = configured_path("ethics_data_dir", config)
    if local is not None:
        return local
    fallback = os.environ.get("ETHICS_DATA_DIR", default)
    return Path(fallback).expanduser().resolve() if fallback is not None else None


def math_eval_dir(config: Path | None = None) -> Path:
    """Resolve and check the external maths harness before starting a run."""
    local = configured_path("math_eval_dir", config)
    if local is None and os.environ.get("MATH_EVAL_DIR"):
        local = Path(os.environ["MATH_EVAL_DIR"]).expanduser().resolve()
    if local is None and os.environ.get("WORKSPACE"):
        local = Path(os.environ["WORKSPACE"]).expanduser().resolve() / "limit-of-RLVR/math/examples/math_eval"
    if local is None:
        raise ModelConfigurationError("set paths.math_eval_dir in config.toml and select it with --config or ES_CAPACITY_CONFIG")
    if not (local / "run_sharded_eval.sh").is_file():
        raise FileNotFoundError(f"maths harness must contain run_sharded_eval.sh: {local}")
    return local


def model_config_path(path: Path | None = None) -> Path | None:
    """Explicit configuration takes precedence over ES_CAPACITY_CONFIG."""
    selected = path if path is not None else os.environ.get("ES_CAPACITY_CONFIG")
    return Path(selected).expanduser().resolve() if selected is not None else None
