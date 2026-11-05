#!/usr/bin/env python3
"""Compatibility launcher for es-capacity-evaluate ethics --subset commonsense."""

from pathlib import Path
import sys

from es_capacity.analysis.ethics_commonsense import load_preds, summarize
from es_capacity.evaluation.ethics_commonsense import (
    DEFAULT_ARMS, ETHICS_URL, MODELS, TASK_NAME, TASK_TEMPLATE,
    ensure_test_jsonl, resolve, run_one_arm, subprocess, urllib, write_task_yaml,
)
from es_capacity.cli.evaluate import main as evaluate

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    evaluate(["ethics", "--subset", "commonsense", *(sys.argv[1:] if argv is None else argv)],
             default_ethics_output=ROOT / "results" / "ethics_results")


if __name__ == "__main__":
    main()
