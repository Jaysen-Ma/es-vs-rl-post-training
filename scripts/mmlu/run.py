#!/usr/bin/env python3
"""Compatibility launcher for es-capacity-evaluate mmlu.

Preserves the existing flags and default results/mmlu_results output directory.
Use the installed command with an explicit --out for new workflows.
"""

from pathlib import Path
import sys

from es_capacity.evaluation.mmlu import MODELS, TASK, resolve, run_one_arm, subprocess
from es_capacity.cli.evaluate import main as evaluate


def main(argv=None):
    evaluate(["mmlu", *(sys.argv[1:] if argv is None else argv)],
             default_mmlu_output=Path("results/mmlu_results"))


if __name__ == "__main__":
    main()
