"""Run model evaluations using configured local checkpoints."""

import argparse
from pathlib import Path

from es_capacity.cli.models import add_config_argument, validate_model_args
from es_capacity.evaluation import mmlu
from es_capacity.models import MODELS


def main(argv=None, *, default_mmlu_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "mmlu", help="evaluate MMLU with one model per worker process",
        description="Evaluate downloaded models on MMLU in an existing GPU "
        "evaluation environment. Use es-capacity-mmlu to analyse saved outputs.",
    )
    command.add_argument("--arms", nargs="+", choices=list(MODELS), default=list(MODELS))
    command.add_argument("--out", type=Path, default=default_mmlu_output,
                         required=default_mmlu_output is None, help="output root for ARM/ result files")
    command.add_argument("--dtype", default="bfloat16")
    command.add_argument("--num-fewshot", type=int, default=0)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    command.add_argument("--max-model-len", type=int, default=1280)
    command.add_argument("--max-num-batched-tokens", type=int, default=2048)
    command.add_argument("--batch-size", type=int, default=1024,
                         help="explicit integer batch size; automatic sizing is not used")
    command.add_argument("--limit", type=float, default=None,
                         help="fraction of each subject to score, e.g. 0.05 for a quick check")
    command.add_argument("--_arm", choices=list(MODELS), default=None, help=argparse.SUPPRESS)
    add_config_argument(command)
    args = parser.parse_args(argv)
    validate_model_args(command, args)
    try:
        mmlu.run(args)
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.split(".")[0] not in {
            "datasets", "lm_eval", "vllm", "torch", "transformers",
        }:
            raise
        command.error(
            f"missing evaluation dependency {exc.name!r}; use an existing GPU "
            "evaluation environment with lm_eval, Datasets and vLLM, then install "
            "this package into that environment. See DEVELOPMENT.md. "
            "The analysis extra does not install the model runtime."
        )


if __name__ == "__main__":
    main()
