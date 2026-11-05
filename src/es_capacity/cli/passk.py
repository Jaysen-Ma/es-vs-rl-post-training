"""Summarize saved pass@k scores without running models."""

import argparse
from contextlib import ExitStack
from importlib import resources
import json
from pathlib import Path
import tempfile

from es_capacity.analysis import passk


def main(argv=None, *, default_results_root=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Outputs BENCHMARK_summary.json and, with --plot, "
        "BENCHMARK_passk.png. Existing files of those names are replaced.",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--results-dir", type=Path, help="run directory containing LABEL/BENCHMARK/*.jsonl")
    inputs.add_argument("--example", action="store_true", help="analyse bundled synthetic scores; no models or data downloads")
    if default_results_root is not None:
        inputs.add_argument("--run-tag", help="legacy run directory beneath this checkout's results/")
    parser.add_argument("--out", type=Path, required=default_results_root is None, help="output directory")
    parser.add_argument("--label", dest="labels", action="append", help="repeat for each input label; at least two required")
    parser.add_argument("--baseline", help="reference label (default: first --label)")
    parser.add_argument("--benchmark", action="append", help="repeat to select benchmarks; default: all four study benchmarks")
    parser.add_argument("--plot", action="store_true", help="also write pass@k plots")
    args = parser.parse_args(argv)
    if args.example and args.out is None:
        parser.error("--example requires --out so outputs survive after the temporary inputs are removed")

    with ExitStack() as stack:
        if args.example:
            run_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="es-capacity-passk-")))
            scores = json.loads(resources.files("es_capacity").joinpath("examples/passk.json").read_text(encoding="utf-8"))
            for label, rows in scores.items():
                directory = run_root / label / "synthetic"
                directory.mkdir(parents=True)
                (directory / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            labels = args.labels if args.labels is not None else list(scores)
            benchmarks = {"synthetic": "Synthetic example"}
            print("Synthetic correctness scores only; these are not model evaluation results.")
        else:
            run_root = args.results_dir
            if run_root is None:
                run_root = Path(default_results_root) / args.run_tag
            labels = args.labels or []
            benchmarks = dict(passk.BENCHMARKS)

        if len(labels) < 2:
            parser.error(f"need at least 2 --label arms, got {len(labels)}")
        if len(set(labels)) != len(labels):
            parser.error("--label values must be unique")
        if any(label in ("", ".", "..") or "/" in label or "\\" in label for label in labels):
            parser.error("--label values must be directory names, without path separators")
        baseline = args.baseline or labels[0]
        if baseline not in labels:
            parser.error(f"--baseline {baseline!r} is not among the --label values: {labels}")
        selected = list(dict.fromkeys(args.benchmark or benchmarks))
        unknown = set(selected) - benchmarks.keys()
        if unknown:
            parser.error(f"unknown benchmarks: {', '.join(sorted(unknown))}; choose from {', '.join(benchmarks)}")
        if not run_root.is_dir():
            parser.error(f"results directory does not exist: {run_root}")
        out = args.out if args.out is not None else run_root
        if out.exists() and not out.is_dir():
            parser.error(f"output path is not a directory: {out}")

        available = []
        for benchmark in selected:
            label_dirs = {label: str(run_root / label / benchmark) for label in labels}
            missing = [label for label, path in label_dirs.items() if not Path(path).is_dir()]
            if missing:
                if args.benchmark:
                    parser.error(f"{benchmark}: no directory for {missing}")
                print(f"Skipping {benchmark}: no directory for {missing}")
                continue
            available.append((benchmark, label_dirs))
        if not available:
            parser.error("no benchmarks have input directories for every requested label")

        for benchmark, label_dirs in available:
            title = benchmarks[benchmark]
            print(f"\n=== {title} ({benchmark}) ===")
            try:
                passk.analyze_benchmark(label_dirs, baseline, title, str(out / benchmark), args.plot)
            except ModuleNotFoundError as exc:
                if exc.name not in ("numpy", "matplotlib"):
                    raise
                parser.error("analysis dependencies are required; run 'uv sync --locked --group analysis' "
                             "from the checkout or install the wheel with '[analysis]'")
            except (OSError, ValueError, AssertionError, KeyError, TypeError) as exc:
                parser.error(f"{benchmark}: {exc}")


if __name__ == "__main__":
    main()
