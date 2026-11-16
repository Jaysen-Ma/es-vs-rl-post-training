"""Command-line interface for summarizing saved ETHICS predictions."""

import argparse
from pathlib import Path

from es_capacity.analysis import ethics_commonsense, ethics_rest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved ETHICS predictions without running models.",
        epilog="Writes summary.csv (commonsense) or rest_summary.csv (rest) in "
        "the results directory, replacing an existing summary of that name.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="directory containing one subdirectory per model arm",
    )
    parser.add_argument(
        "--subset", choices=("commonsense", "rest"), required=True,
        help="commonsense morality, or deontology/justice/virtue/utilitarianism",
    )
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")

    filename = "samples.jsonl" if args.subset == "commonsense" else "rest_samples.jsonl"
    arms = [
        path.name for path in sorted(args.results_dir.iterdir())
        if (path / filename).is_file()
    ]
    if not arms:
        parser.error(f"no ARM/{filename} files found in {args.results_dir}")

    try:
        if args.subset == "commonsense":
            ethics_commonsense.summarize(args.results_dir, arms)
        else:
            ethics_rest.summarize(args.results_dir)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        parser.error(
            "SciPy is required; from the checkout run "
            "'uv sync --locked --group analysis', or install the wheel with '[analysis]'"
        )


if __name__ == "__main__":
    main()
