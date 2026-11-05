#!/usr/bin/env python3
"""Compatibility launcher for pass@k analysis.

Existing --run-tag commands continue to use results/RUN beneath this checkout.
The installed es-capacity-passk command uses explicit input and output paths.
"""

from pathlib import Path

from es_capacity.analysis.passk import (
    BENCHMARKS, DEFAULT_COLORS, DISPLAY, analyze_benchmark, four_way_breakdown,
    load_scores, pass_at_k_curve, plot_passk, power_of_2_ks,
)
from es_capacity.cli.passk import main as cli_main

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def main(argv=None):
    cli_main(argv, default_results_root=Path(REPO_ROOT) / "results")


if __name__ == "__main__":
    main()
