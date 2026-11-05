#!/usr/bin/env python3
"""Compatibility launcher for es-capacity-compare-weights."""

from pathlib import Path

from es_capacity.analysis.weight_comparison import (
    CHUNK_ELEMS, MODULES, N_BINS, SCALES, SERIES, SIGMA_SPAN, Checkpoint, Pass1, Pass2,
    compare, depth_of, groups_of, is_matrix, module_of, shared_names, tensor_map,
    write_hist, write_per_tensor, write_stats,
)
from es_capacity.cli.compare_weights import main as compare_main


def main(argv=None):
    compare_main(argv, default_output=Path("results/weights"))


if __name__ == "__main__":
    main()
