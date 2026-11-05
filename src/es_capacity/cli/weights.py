"""Plot weight distributions from saved CSVs without loading checkpoints."""

import argparse
from pathlib import Path

from es_capacity.analysis.weights import render_figures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Writes five PNG figures in --out, replacing files of the same names.",
    )
    parser.add_argument("--scale", required=True,
                        help="model-scale label used in input directories, e.g. 1.5B or 7B")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="directory containing SCALE-ES/ and SCALE-RL/ CSV files")
    parser.add_argument("--out", type=Path, required=True, help="output directory for figures")
    args = parser.parse_args(argv)
    if not args.scale or args.scale in (".", "..") or "/" in args.scale or "\\" in args.scale:
        parser.error("--scale must be a single directory-name component")
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")
    try:
        paths = render_figures(args.results_dir, args.scale, args.out)
    except ModuleNotFoundError as exc:
        if exc.name not in ("numpy", "matplotlib"):
            raise
        parser.error(
            "NumPy and Matplotlib are required; run "
            "'uv sync --locked --group analysis', or install the wheel with '[analysis]'"
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"wrote {len(paths)} figures to {args.out}/ for {args.scale}")


if __name__ == "__main__":
    main()
