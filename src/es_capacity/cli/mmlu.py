"""Generate tables from saved MMLU outputs without running models."""

import argparse
from pathlib import Path

from es_capacity.analysis import mmlu


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Writes scores.csv in --out, replacing an existing file of that "
        "name, and prints the Markdown table to standard output.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="directory containing ARM/results.json and ARM/samples.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory for scores.csv")
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")

    try:
        rows = mmlu.summarize(args.results_dir)
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / "scores.csv"
        mmlu.write_scores(rows, out)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        parser.error(
            "SciPy is required for paired comparisons; run "
            "'uv sync --locked --group analysis', or install the wheel with '[analysis]'"
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    print(mmlu.format_table(rows))
    print(f"\nn = {rows[0]['n']} questions, seed 0. wrote {out}")


if __name__ == "__main__":
    main()
