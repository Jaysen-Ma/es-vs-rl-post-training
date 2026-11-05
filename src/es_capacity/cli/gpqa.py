"""Check GPQA seed sweeps, summarize saved scores and render Markdown tables."""

import argparse
from pathlib import Path

from es_capacity.analysis import gpqa


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "validate matched questions and distinct permutations; write nothing"),
        ("summarize", "write CSV summaries from a saved seed sweep"),
        ("tables", "print Markdown tables from reduced CSV summaries"),
    ):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--results-dir", type=Path, required=True,
                             help="seed-sweep directory, or CSV directory for tables")
        if name == "summarize":
            command.add_argument("--out", type=Path, required=True,
                                 help="output directory; generated CSVs are replaced")
            command.add_argument("--pair", action="append", default=[], metavar="ARM:BASELINE",
                                 help="paired comparison; repeat once per pair")
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")

    try:
        if args.command == "tables":
            print(gpqa.format_tables(args.results_dir), end="")
            return

        sweep = gpqa.load_sweep(args.results_dir)
        seeds = sorted(sweep)
        arms = sorted({arm for values in sweep.values() for arm in values})
        print(f"loaded {len(seeds)} seeds x {len(arms)} arms: {', '.join(arms)}\n")
        problems = gpqa.check_invariants(sweep)
        if problems:
            print("PROBLEMS FOUND:")
            for problem in problems:
                print(f"  ! {problem}")
            raise SystemExit(1)
        print(f"invariants OK: arms agree on items within each seed; "
              f"all {len(seeds)} seeds are distinct permutations\n")
        if args.command == "check":
            return

        tables = gpqa.summarize(sweep, args.pair)
        args.out.mkdir(parents=True, exist_ok=True)
        gpqa.write_summaries(tables, args.out)
        print(gpqa.format_console_summary(tables), end="")
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        parser.error(
            "SciPy is required for paired comparisons; run "
            "'uv sync --locked --group analysis', or install the wheel with '[analysis]'"
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
