#!/usr/bin/env python3
"""Compatibility launcher for GPQA analysis; prefer es-capacity-gpqa."""

import argparse

from es_capacity.analysis.gpqa import (
    TASK, check_invariants, load_arm, load_sweep, mcnemar_exact, wilcoxon_signed_rank,
)
from es_capacity.cli.gpqa import main as analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--out")
    parser.add_argument("--pair", action="append", default=[], metavar="ARM:BASELINE")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        analyze(["check", "--results-dir", args.sweep_root])
        return
    if args.out is None:
        parser.error("--out is required unless --check")
    command = ["summarize", "--results-dir", args.sweep_root, "--out", args.out]
    for pair in args.pair:
        command.extend(["--pair", pair])
    analyze(command)


if __name__ == "__main__":
    main()
