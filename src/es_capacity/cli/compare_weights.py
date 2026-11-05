"""Compare downloaded checkpoints on the CPU and write weight-statistics CSVs."""

import argparse
from pathlib import Path

from es_capacity.analysis import weight_comparison
from es_capacity.cli.models import add_config_argument
from es_capacity.models import ModelConfigurationError, resolve_model


def main(argv=None, *, default_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(weight_comparison.SCALES), required=True)
    parser.add_argument("--arms", nargs="+", choices=("ES", "RL"), default=["ES", "RL"])
    parser.add_argument("--out", type=Path, default=default_output, required=default_output is None,
                        help="output root for SCALE-ARM CSV directories; existing CSVs are replaced")
    add_config_argument(parser)
    args = parser.parse_args(argv)
    try:
        paths = {arm: Path(resolve_model(f"{args.scale}-{arm}", config=args.config))
                 for arm in ["base", *args.arms]}
    except (ModelConfigurationError, FileNotFoundError) as exc:
        parser.error(str(exc))

    try:
        spec = weight_comparison.SCALES[args.scale]
        for arm in args.arms:
            print(f"{args.scale} {arm}: {spec['base']} -> {spec[arm]}", flush=True)
            weight_comparison.compare(paths["base"], paths[arm], args.out / f"{args.scale}-{arm}",
                                      f"{args.scale} {arm}")
    except ModuleNotFoundError as exc:
        if exc.name not in ("torch", "safetensors"):
            raise
        parser.error("PyTorch and SafeTensors are required in the checkpoint-comparison "
                     "environment; the analysis extra does not install them. See DEVELOPMENT.md.")
    except OSError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
