#!/usr/bin/env python3
"""Compatibility launcher for saved MMLU tables.

New commands should use es-capacity-mmlu with explicit --results-dir and --out.
This launcher preserves --sweep-root and writes scores.csv beside the inputs.
"""

import argparse
from pathlib import Path

from es_capacity.analysis.mmlu import ARM_ORDER, TASK, correctness, load
from es_capacity.cli.mmlu import main as analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=Path("results/mmlu_results"))
    args = parser.parse_args(argv)
    analyze(["--results-dir", str(args.sweep_root), "--out", str(args.sweep_root)])


if __name__ == "__main__":
    main()
