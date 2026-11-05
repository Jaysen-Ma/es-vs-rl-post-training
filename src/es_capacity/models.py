"""The study's six model arms and their existing local-mirror resolution rules."""

from __future__ import annotations

from pathlib import Path

from es_capacity.config import models_dir

# Retain the RL labels used in saved evaluation outputs and existing commands.
MODELS = {
    "1.5B-base": "Qwen/Qwen2.5-1.5B",
    "1.5B-ES": "zocrate/Qwen2.5-1.5B-ES-math",
    "1.5B-RL": "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
    "7B-base": "Qwen/Qwen2.5-7B",
    "7B-ES": "zocrate/Qwen2.5-7B-ES-math",
    "7B-RL": "hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo",
}


def local_checkpoint_dir(arm: str, root: Path | None = None) -> Path:
    """Return the checkpoint's basename under the configured local mirror."""
    hub_id = MODELS.get(arm, arm)
    return (models_dir() if root is None else root) / hub_id.split("/")[-1]


def resolve_model(arm: str, root: Path | None = None) -> str:
    """Use a local mirror with config.json, otherwise the Hub ID or input.

    This preserves the runners' existing basename-matching behaviour, including
    for custom Hub IDs or paths passed through --arms. It never downloads a model.
    """
    local = local_checkpoint_dir(arm, root)
    return str(local) if (local / "config.json").is_file() else MODELS.get(arm, arm)
