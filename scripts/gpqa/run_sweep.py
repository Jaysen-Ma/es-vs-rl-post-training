#!/usr/bin/env python3
"""Compatibility launcher for es-capacity-evaluate gpqa.

Preserves the existing flags and default results/gpqa_results output directory.
Use the installed command with an explicit --out for new workflows.
"""

from pathlib import Path
import sys

from es_capacity.evaluation.gpqa import MODELS, TASK, resolve, run_one_arm, subprocess
from es_capacity.cli.evaluate import main as evaluate


def main(argv=None):
    evaluate(["gpqa", *(sys.argv[1:] if argv is None else argv)],
             default_gpqa_output=Path("results/gpqa_results"))


if __name__ == "__main__":
    main()
