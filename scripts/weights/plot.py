#!/usr/bin/env python3
"""Compatibility launcher for saved weight figures; prefer es-capacity-weights."""

import argparse
from pathlib import Path

from es_capacity.analysis.weights import (
    COLORS, NAMES, N_BINS, MODULE_ORDER, auto_rebin, choose_factor, common_factor,
    density, gaussian_ref, masked, read_hist, read_stats, rebin,
)
from es_capacity.cli.weights import main as plot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--dir", type=Path, default=Path("results/weights"))
    args = parser.parse_args(argv)
    plot(["--scale", args.scale, "--results-dir", str(args.dir),
          "--out", str(args.dir / "figures")])


if __name__ == "__main__":
    main()
