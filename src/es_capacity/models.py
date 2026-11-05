"""The study's six model arms and user-supplied local checkpoint paths."""

from __future__ import annotations

from pathlib import Path

from es_capacity.config import model_config_path, models_dir

# Retain the RL labels used in saved evaluation outputs and existing commands.
MODELS = {
    "1.5B-base": "Qwen/Qwen2.5-1.5B",
    "1.5B-ES": "zocrate/Qwen2.5-1.5B-ES-math",
    "1.5B-RL": "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
    "7B-base": "Qwen/Qwen2.5-7B",
    "7B-ES": "zocrate/Qwen2.5-7B-ES-math",
    "7B-RL": "hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo",
}


class ModelConfigurationError(ValueError):
    """Invalid model configuration, reported before model execution."""


def load_model_paths(path: Path) -> dict[str, Path]:
    """Read paths relative to the configuration file, independent of cwd."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    path = path.expanduser().resolve()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ModelConfigurationError(f"cannot read model configuration {path}: {exc}") from exc
    if set(data) != {"models"} or not isinstance(data["models"], dict):
        raise ModelConfigurationError(f"{path}: expected a [models] table only")
    unknown = set(data["models"]) - MODELS.keys()
    if unknown:
        raise ModelConfigurationError(f"{path}: unknown model arms: {', '.join(sorted(unknown))}")
    paths = {}
    for arm, value in data["models"].items():
        if not isinstance(value, str) or not value.strip():
            raise ModelConfigurationError(f"{path}: {arm} must be a nonempty local path string")
        local = Path(value).expanduser()
        paths[arm] = (local if local.is_absolute() else path.parent / local).resolve()
    return paths


def local_checkpoint_dir(
    arm: str, root: Path | None = None, *, config: Path | None = None,
) -> Path:
    """Return a configured path; retain MODELS_DIR as a local-only fallback."""
    # Accept published IDs as aliases, while keeping the study's stable arm names.
    arm = next((name for name, hub_id in MODELS.items() if hub_id == arm), arm)
    candidate = Path(arm).expanduser()
    if arm not in MODELS and (
        candidate.is_absolute() or arm.startswith((".", "~")) or candidate.is_dir()
    ):
        return candidate.resolve()
    selected = model_config_path(config)
    if arm in MODELS and selected is not None:
        paths = load_model_paths(selected)
        if arm not in paths:
            raise ModelConfigurationError(
                f"{selected}: no local path configured for {arm}; add it under [models]. "
                f"Download: https://huggingface.co/{MODELS[arm]}"
            )
        return paths[arm]
    hub_id = MODELS.get(arm, arm)
    return (models_dir() if root is None else root).expanduser() / hub_id.split("/")[-1]


def resolve_model(
    arm: str, root: Path | None = None, *, config: Path | None = None,
) -> str:
    """Validate and return an absolute local path; never fall back to a Hub ID."""
    local = local_checkpoint_dir(arm, root, config=config).resolve()
    if not (local / "config.json").is_file():
        source = MODELS.get(arm)
        if source is None and arm in MODELS.values():
            source = arm
        download = f"\nDownload: https://huggingface.co/{source}" if source is not None else ""
        raise FileNotFoundError(
            f"{arm}: local checkpoint must contain config.json: {local}\n"
            "Download the checkpoint yourself and set its path in config.example.toml's "
            "[models] format. Select your file with --config or ES_CAPACITY_CONFIG."
            + download
        )
    return str(local)
