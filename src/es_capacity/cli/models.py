"""Inspect user-downloaded model checkpoints without loading model libraries."""

import argparse
from pathlib import Path

from es_capacity.config import model_config_path
from es_capacity.models import MODELS, ModelConfigurationError, resolve_model


def add_config_argument(parser):
    parser.add_argument(
        "--config", type=Path, help="model paths TOML file (or set ES_CAPACITY_CONFIG)",
    )


def validate_model_args(parser, args):
    """Validate selected arms before dataset setup or worker creation."""
    args.config = model_config_path(args.config)
    arms = [args._arm] if args._arm is not None else args.arms
    try:
        for arm in arms:
            resolve_model(arm, config=args.config)
    except (ModelConfigurationError, FileNotFoundError) as exc:
        parser.error(str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check selected local paths without loading models")
    add_config_argument(check)
    check.add_argument("--arms", nargs="+", choices=list(MODELS), default=list(MODELS))
    path = commands.add_parser("path", help="print one validated checkpoint path")
    add_config_argument(path)
    path.add_argument("arm", help="model arm or existing local checkpoint directory")
    args = parser.parse_args(argv)

    if args.command == "path":
        try:
            print(resolve_model(args.arm, config=args.config))
        except (ModelConfigurationError, FileNotFoundError) as exc:
            parser.error(str(exc))
        return

    failed = False
    for arm in args.arms:
        try:
            local = resolve_model(arm, config=args.config)
        except ModelConfigurationError as exc:
            parser.error(str(exc))
        except FileNotFoundError as exc:
            print(f"MISSING {exc}")
            failed = True
        else:
            print(f"OK {arm}: {local}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
