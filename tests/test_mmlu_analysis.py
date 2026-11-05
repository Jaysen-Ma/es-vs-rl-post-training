"""Saved MMLU table, pairing and installed-command contracts."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from es_capacity.analysis import mmlu
from es_capacity.cli.mmlu import main


ROOT = Path(__file__).resolve().parents[1]


def write_arm(root, arm, scores):
    directory = root / arm
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({
        "results": {"mmlu": {"acc,none": sum(scores) / len(scores)}},
        "n-samples": {"subject_a": {"effective": 4}, "subject_b": {"effective": 4}},
        "group_subtasks": {"mmlu": ["category"], "category": ["subject_a", "subject_b"]},
        "config": {"num_fewshot": 0, "wall_seconds": 1.5},
    }))
    # IDs repeat across subjects and records need not be in the same order.
    rows = [{"subtask": "subject_a" if i < 4 else "subject_b",
             "doc_id": i % 4, "acc": int(score)} for i, score in enumerate(scores)]
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows))
    )


def fixture(root):
    for arm, scores in (
        ("1.5B-base", [True] * 6 + [False] * 2),
        ("1.5B-ES", [False] * 8),
        ("1.5B-RL", [True] * 6 + [False] * 2),
        ("7B-base", [False] * 6 + [True] * 2),
        ("7B-ES", [True] * 8),
        ("7B-RL", [False] * 6 + [True] * 2),
        ("custom-model", [True, False] * 4),
    ):
        write_arm(root, arm, scores)


class MmluAnalysisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = self.root / "saved results"
        self.out = self.root / "output summaries"

    def call_cli(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--results-dir", str(self.inputs), "--out", str(self.out)])
        return output.getvalue()

    def test_csv_and_table_preserve_scores_order_and_paired_tests(self):
        fixture(self.inputs)
        before = {p: p.read_bytes() for p in self.inputs.rglob("*") if p.is_file()}
        output = self.call_cli()
        self.assertEqual((self.out / "scores.csv").read_text(),
            "arm,acc,n,gained,lost,mcnemar_p,mcnemar_p_adj,num_fewshot,wall_seconds\n"
            "1.5B-base,0.75,8,,,,,0,1.5\n"
            "1.5B-ES,0.0,8,0,6,0.03125,0.125,0,1.5\n"
            "1.5B-RL,0.75,8,0,0,1.0,1.0,0,1.5\n"
            "7B-base,0.25,8,,,,,0,1.5\n"
            "7B-ES,1.0,8,6,0,0.03125,0.125,0,1.5\n"
            "7B-RL,0.25,8,0,0,1.0,1.0,0,1.5\n"
            "custom-model,0.5,8,,,,,0,1.5\n")
        self.assertEqual(output,
            "| Arm | MMLU 0-shot | McNemar p vs. base |\n"
            "|---|---|---|\n"
            "| 1.5B-base | 75.00% | — |\n"
            "| 1.5B-ES | 0.00% | 0.12 |\n"
            "| 1.5B-RL | 75.00% | 1.00 |\n"
            "| 7B-base | 25.00% | — |\n"
            "| 7B-ES | 100.00% | 0.12 |\n"
            "| 7B-RL | 25.00% | 1.00 |\n"
            "| custom-model | 50.00% | — |\n"
            f"\nn = 8 questions, seed 0. wrote {self.out / 'scores.csv'}\n")
        self.assertEqual(before, {p: p.read_bytes() for p in self.inputs.rglob("*") if p.is_file()})

    def test_legacy_launcher_preserves_default_and_explicit_output_locations(self):
        fixture(self.inputs)
        expected = self.call_cli()
        spec = importlib.util.spec_from_file_location("mmlu_tables", ROOT / "scripts/mmlu/tables.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertIs(launcher.correctness, mmlu.correctness)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            launcher.main(["--sweep-root", str(self.inputs)])
        self.assertEqual((self.inputs / "scores.csv").read_bytes(), (self.out / "scores.csv").read_bytes())
        self.assertEqual(output.getvalue(), expected.replace(str(self.out), str(self.inputs)))
        with patch.object(launcher, "analyze") as analyze:
            launcher.main([])
            analyze.assert_called_once_with([
                "--results-dir", "results/mmlu_results", "--out", "results/mmlu_results",
            ])

    def test_invalid_inputs_fail_before_creating_output(self):
        fixture(self.inputs)
        samples = self.inputs / "7B-ES/samples.jsonl"
        original = samples.read_text()
        cases = (
            (original.splitlines()[1:], "question IDs do not match"),
            ([*original.splitlines(), original.splitlines()[0]], "duplicate question ID"),
            ([*original.splitlines(), json.dumps({"subtask": "extra", "doc_id": 9, "acc": 1})],
             "question IDs do not match"),
            (["invalid JSON"], "Expecting value"),
        )
        for lines, message in cases:
            with self.subTest(message=message):
                samples.write_text("\n".join(lines) + "\n")
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        self.call_cli()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, error.getvalue())
                self.assertFalse(self.out.exists())
        samples.unlink()
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):
                self.call_cli()
        self.assertIn("samples.jsonl", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_missing_or_empty_results_directory_fails_clearly(self):
        for create in (False, True):
            if create:
                self.inputs.mkdir()
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    self.call_cli()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("no arms" if create else "does not exist", error.getvalue())
            self.assertFalse(self.out.exists())

    def test_missing_scipy_explains_installation_without_creating_output(self):
        fixture(self.inputs)
        with patch.object(mmlu, "summarize", side_effect=ModuleNotFoundError(name="scipy")):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    self.call_cli()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("uv sync --locked --group analysis", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_help_imports_no_optional_analysis_or_model_libraries(self):
        code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy', 'numpy', 'torch', 'vllm', 'lm_eval', 'datasets', 'transformers'}:
            raise AssertionError(f'CLI help imported {fullname}')
sys.meta_path.insert(0, BlockOptional())
from es_capacity.cli.mmlu import main
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--results-dir", result.stdout)
        self.assertIn("--out", result.stdout)


if __name__ == "__main__":
    unittest.main()
