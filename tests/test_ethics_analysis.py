"""Saved-result analysis and CLI contracts using synthetic predictions."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from es_capacity.analysis import ethics_commonsense, ethics_rest
from es_capacity.cli import ethics as ethics_cli


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def sample(doc_id, gold, logodds, **fields):
    return {
        "doc_id": doc_id, "target": gold,
        "filtered_resps": [[-10.0, True], [-10.0 + logodds, True]],
        **fields,
    }


def write_samples(root, arm, filename, rows):
    path = root / arm / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    return path


def commonsense_fixture(root):
    for arm in ("7B-base", "7B-ES", "custom-model"):
        rows = []
        for i in range(8):
            gold = i % 2
            pred = 1 - gold if arm == "7B-ES" and i < 6 else gold
            rows.append(sample(i, gold, 1 if pred else -1))
        write_samples(root, arm, "samples.jsonl", rows)


def rest_fixture(root):
    for arm in ("7B-base", "7B-ES"):
        rows = []
        for task, size in (("deontology", 4), ("justice", 4), ("virtue", 5)):
            for group in range(2):
                for i in range(size):
                    gold = int(i == 0)
                    pred = 1 if arm == "7B-ES" and group == 1 and i == 1 else gold
                    rows.append(sample(
                        group * size + i, gold, 1 if pred else -1,
                        subtask=task, doc={"group_id": group},
                    ))
        for pair in range(2):
            for gold in (0, 1):
                logodds = 2 if gold else -2
                if arm == "7B-ES" and pair == 1 and gold == 1:
                    logodds = -3
                rows.append(sample(
                    pair * 2 + gold, gold, logodds, subtask="util",
                    doc={"pair_id": pair, "canon": gold},
                ))
        write_samples(root, arm, "rest_samples.jsonl", rows)


class EthicsAnalysisTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SCIPY, "install the analysis dependencies")
    def test_commonsense_invalid_records_or_pairs_preserve_existing_summary(self):
        for case in ("duplicate", "missing", "different-id", "different-label", "invalid-label", "fractional-label", "invalid-choices"):
            with self.subTest(case=case):
                root = self.root / case
                commonsense_fixture(root)
                path = root / "7B-ES/samples.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if case == "duplicate":
                    rows.append(rows[0])
                elif case == "missing":
                    rows.pop()
                elif case == "different-id":
                    rows[0]["doc_id"] = 99
                elif case == "different-label":
                    rows[0]["target"] = 1 - rows[0]["target"]
                elif case == "invalid-label":
                    rows[0]["target"] = 2
                elif case == "fractional-label":
                    rows[0]["target"] = 0.5
                else:
                    rows[0]["filtered_resps"] = [[0]]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                summary = root / "summary.csv"
                summary.write_text("previous valid summary\n")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        ethics_cli.main(["--results-dir", str(root), "--subset", "commonsense"])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(summary.read_text(), "previous valid summary\n")

    @unittest.skipUnless(HAS_SCIPY, "install the analysis dependencies")
    def test_single_label_summary_marks_the_absent_class_without_crashing(self):
        for gold in (0, 1):
            root = self.root / str(gold)
            for arm in ("7B-base", "7B-ES"):
                write_samples(root, arm, "samples.jsonl", [sample(0, gold, 1 if gold else -1)])
            with contextlib.redirect_stdout(io.StringIO()):
                ethics_cli.main(["--results-dir", str(root), "--subset", "commonsense"])
            with (root / "summary.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                self.assertEqual(row["acc"], "1.000000")
                self.assertEqual(row["acc_when_not_wrong"], "1.000000" if gold == 0 else "")
                self.assertEqual(row["acc_when_wrong"], "1.000000" if gold == 1 else "")
            self.assertEqual(rows[1]["mcnemar_p"], "1.0000")

    @unittest.skipUnless(HAS_SCIPY, "install the analysis dependencies")
    def test_empty_commonsense_samples_fail_without_replacing_summary(self):
        write_samples(self.root, "7B-base", "samples.jsonl", [])
        summary = self.root / "summary.csv"
        summary.write_text("previous valid summary\n")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.run_cli("commonsense")
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(summary.read_text(), "previous valid summary\n")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def run_cli(self, subset):
        with contextlib.redirect_stdout(io.StringIO()):
            ethics_cli.main(["--results-dir", str(self.root), "--subset", subset])

    def test_commonsense_load_sorts_ids_and_breaks_ties_as_before(self):
        path = write_samples(self.root, "arm", "samples.jsonl", [
            sample(0, 0, 0), sample(1, 1, 2),
        ])
        self.assertEqual(ethics_commonsense.load_preds(path), [
            {"doc_id": 0, "gold": 0, "pred": 0, "correct": True},
            {"doc_id": 1, "gold": 1, "pred": 1, "correct": True},
        ])

    @unittest.skipUnless(HAS_SCIPY, "install the analysis dependencies")
    def test_commonsense_cli_writes_expected_csv_including_custom_arm(self):
        commonsense_fixture(self.root)
        unrelated = self.root / "rest_summary.csv"
        unrelated.write_text("existing rest summary\n")
        self.run_cli("commonsense")
        self.assertEqual((self.root / "summary.csv").read_text(),
            "arm,n,acc,pred_yes,acc_when_not_wrong,acc_when_wrong,mcnemar_b,mcnemar_c,mcnemar_p\n"
            "7B-base,8,1.000000,0.500000,1.000000,1.000000,,,\n"
            "7B-ES,8,0.250000,0.500000,0.250000,0.250000,0,6,0.0312\n"
            "custom-model,8,1.000000,0.500000,1.000000,1.000000,,,\n")
        self.assertEqual(unrelated.read_text(), "existing rest summary\n")

    @unittest.skipUnless(HAS_SCIPY, "install the analysis dependencies")
    def test_rest_cli_writes_expected_group_and_pair_scores(self):
        rest_fixture(self.root)
        self.run_cli("rest")
        self.assertEqual((self.root / "rest_summary.csv").read_text(),
            "task,arm,n,example_acc,p_positive,set_n,set_acc,set_b,set_c,set_p\n"
            "deontology,7B-base,8,1.000000,0.250000,2,1.000000,,,\n"
            "deontology,7B-ES,8,0.875000,0.375000,2,0.500000,0,1,1\n"
            "justice,7B-base,8,1.000000,0.250000,2,1.000000,,,\n"
            "justice,7B-ES,8,0.875000,0.375000,2,0.500000,0,1,1\n"
            "virtue,7B-base,10,1.000000,0.200000,2,1.000000,,,\n"
            "virtue,7B-ES,10,0.900000,0.300000,2,0.500000,0,1,1\n"
            "util,7B-base,2,1.000000,1.000000,2,1.000000,,,\n"
            "util,7B-ES,2,0.500000,0.500000,2,0.500000,0,1,1\n")

    def test_missing_directory_or_samples_fails_without_writing_output(self):
        for root in (self.root / "missing", self.root):
            for subset in ("commonsense", "rest"):
                with self.subTest(root=root, subset=subset):
                    with contextlib.redirect_stderr(io.StringIO()) as error:
                        with self.assertRaises(SystemExit) as raised:
                            ethics_cli.main(["--results-dir", str(root), "--subset", subset])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("does not exist" if not root.exists() else "no ARM/", error.getvalue())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_scipy_explains_installation(self):
        commonsense_fixture(self.root)
        missing = ModuleNotFoundError("No module named 'scipy'", name="scipy")
        with (
            patch.object(ethics_commonsense, "summarize", side_effect=missing),
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            with self.assertRaises(SystemExit) as raised:
                self.run_cli("commonsense")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("uv sync --locked --group analysis", error.getvalue())


    def test_analysis_import_does_not_load_or_configure_model_runtime(self):
        code = """
import importlib.abc
import os
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'lm_eval', 'datasets', 'transformers'}:
            raise AssertionError(f'analysis imported {fullname}')

sys.meta_path.insert(0, BlockRuntime())
before = dict(os.environ)
from es_capacity.cli import ethics as ethics_cli
assert dict(os.environ) == before
ethics_cli.main(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--results-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
