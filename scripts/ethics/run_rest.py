#!/usr/bin/env python3
"""Compatibility launcher for es-capacity-evaluate ethics --subset rest."""

from pathlib import Path
import sys

from es_capacity.analysis.ethics_rest import (
    DEFAULT_ARMS, GROUP_SIZE, TASKS, _mcnemar, _units, load_task, summarize,
)
from es_capacity.evaluation.ethics_rest import CHOICES, _rows, prepare, resolve, run_one_arm, subprocess
from es_capacity.cli.evaluate import main as evaluate

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    evaluate(["ethics", "--subset", "rest", *(sys.argv[1:] if argv is None else argv)],
             default_ethics_output=ROOT / "results" / "ethics_results")


if __name__ == "__main__":
    main()
