#!/usr/bin/env python3
"""Compatibility launcher for GPQA Markdown tables; prefer es-capacity-gpqa tables."""

import argparse

from es_capacity.analysis.gpqa import read
from es_capacity.cli.gpqa import main as analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/gpqa")
    args = parser.parse_args(argv)
    analyze(["tables", "--results-dir", args.dir])


if __name__ == "__main__":
    main()
