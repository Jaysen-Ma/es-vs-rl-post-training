"""Run model evaluations using configured local checkpoints."""

import argparse
from pathlib import Path

from es_capacity.cli.models import add_config_argument, validate_model_args
from es_capacity.evaluation import ethics_commonsense, ethics_rest, gpqa, mmlu
from es_capacity.config import ethics_data_dir
from es_capacity.models import MODELS


def _add_model_arguments(command, default_output, *, gpu_memory, max_model_len, default_arms=None):
    command.add_argument("--arms", nargs="+", choices=list(MODELS),
                         default=list(MODELS) if default_arms is None else list(default_arms))
    command.add_argument("--out", type=Path, default=default_output,
                         required=default_output is None, help="output root for evaluation files")
    command.add_argument("--dtype", default="bfloat16")
    command.add_argument("--gpu-memory-utilization", type=float, default=gpu_memory)
    command.add_argument("--max-model-len", type=int, default=max_model_len)
    command.add_argument("--_arm", choices=list(MODELS), default=None, help=argparse.SUPPRESS)
    add_config_argument(command)


def main(argv=None, *, default_mmlu_output=None, default_gpqa_output=None, default_ethics_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "mmlu", help="evaluate MMLU with one model per worker process",
        description="Evaluate downloaded models on MMLU in an existing GPU "
        "evaluation environment. Use es-capacity-mmlu to analyse saved outputs.",
    )
    _add_model_arguments(command, default_mmlu_output, gpu_memory=0.25, max_model_len=1280)
    command.add_argument("--num-fewshot", type=int, default=0)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--max-num-batched-tokens", type=int, default=2048)
    command.add_argument("--batch-size", type=int, default=1024,
                         help="explicit integer batch size; automatic sizing is not used")
    command.add_argument("--limit", type=float, default=None,
                         help="fraction of each subject to score, e.g. 0.05 for a quick check")
    command = commands.add_parser(
        "gpqa", help="evaluate GPQA over answer-option permutations",
        description="Evaluate GPQA-diamond with one model per worker and multiple "
        "answer-order seeds. Use es-capacity-gpqa to analyse saved outputs.",
    )
    _add_model_arguments(command, default_gpqa_output, gpu_memory=0.5, max_model_len=4096)
    command.add_argument("--seeds", nargs="+", type=int, default=list(range(10)),
                         help="distinct nonnegative answer-order seeds; default: 0 through 9")
    command.add_argument("--batch-size", default="auto")
    command.add_argument("--enforce-eager", action="store_true", default=True,
                         help="retain the study's eager execution setting (enabled by default)")
    command = commands.add_parser(
        "ethics", help="evaluate the ETHICS benchmark subsets",
        description="Evaluate moral scenarios using local task data. "
        "Use es-capacity-ethics to analyse saved predictions.",
    )
    _add_model_arguments(command, default_ethics_output, gpu_memory=0.25, max_model_len=2048,
                         default_arms=ethics_commonsense.DEFAULT_ARMS)
    command.set_defaults(arms=None)
    command.add_argument("--subset", choices=("commonsense", "rest"), required=True)
    command.add_argument("--data-dir", type=Path, default=None,
                         help="unpacked deontology/justice/virtue/utilitarianism data (rest only)")
    command.add_argument("--csv", type=Path, default=None,
                         help="official cm_test.csv; otherwise reuse local data or download ethics.tar")
    command.add_argument("--num-fewshot", type=int, default=0)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--max-num-batched-tokens", type=int, default=2048)
    command.add_argument("--batch-size", type=int, default=1024)
    command.add_argument("--limit", type=float, default=None,
                         help="fraction or count of scenarios for a quick check")
    args = parser.parse_args(argv)
    command = commands.choices[args.command]
    if args.command == "ethics" and args.arms is None:
        args.arms = list(ethics_commonsense.DEFAULT_ARMS if args.subset == "commonsense"
                         else ethics_rest.DEFAULT_ARMS)
    if args.command == "gpqa":
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            command.error("--seeds must be distinct nonnegative integers")
    validate_model_args(command, args)
    if args.command == "ethics" and args.subset == "rest":
        if args.csv is not None or args.num_fewshot != 0:
            command.error("--csv and nonzero --num-fewshot apply only to commonsense")
        args.data_dir = (args.data_dir if args.data_dir is not None else ethics_data_dir()).expanduser().resolve()
        if not args.data_dir.is_dir():
            command.error(f"missing {args.data_dir}; set --data-dir or ETHICS_DATA_DIR to the unpacked ETHICS directory")
    elif args.command == "ethics" and args.data_dir is not None:
        command.error("--data-dir applies only to --subset rest; use --csv for commonsense")
    if args.command == "ethics" and args.csv is not None:
        args.csv = args.csv.expanduser().resolve()
        if not args.csv.is_file():
            command.error(f"CSV file does not exist: {args.csv}")
    try:
        if args.command == "ethics":
            runner = ethics_rest if args.subset == "rest" else ethics_commonsense
        else:
            runner = {"mmlu": mmlu, "gpqa": gpqa}[args.command]
        runner.run(args)
    except ModuleNotFoundError as exc:
        if args.command == "ethics" and exc.name == "scipy":
            command.error("SciPy is required for ETHICS statistical summaries; install "
                          "the package's '[analysis]' extra in the existing evaluation environment.")
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
    except OSError as exc:
        if args.command != "ethics":
            raise
        command.error(str(exc))


if __name__ == "__main__":
    main()
