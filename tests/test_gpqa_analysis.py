"""Synthetic GPQA sweep, report and installed-command contracts."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from es_capacity.analysis import gpqa
from es_capacity.cli.gpqa import main


ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    for seed, base, candidate in ((0, [1, 0, 0, 0], [1, 1, 1, 0]),
                                  (1, [0, 1, 0, 0], [0, 0, 1, 1])):
        for arm, scores in (("7B-base", base), ("7B-ES", candidate)):
            directory = root / f"seed{seed}" / arm
            directory.mkdir(parents=True)
            (directory / "results_example.json").write_text(json.dumps({
                "results": {gpqa.TASK: {"acc,none": sum(scores) / 4, "acc_stderr,none": 0.1}},
                "n-samples": {gpqa.TASK: {"effective": 4}},
                "config": {"model_args": {"pretrained": f"/synthetic/{arm}"}},
            }))
            rows = []
            for doc_id, correct in enumerate(scores):
                slot = 2 if arm == "7B-ES" else (doc_id + seed) % 4
                rows.append({"doc_id": doc_id, "acc": correct,
                             "doc_hash": f"seed{seed}-doc{doc_id}",
                             "filtered_resps": [[int(i == slot)] for i in range(4)]})
            (directory / f"samples_{gpqa.TASK}_example.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(rows))
            )


EXPECTED_CSVS = {
    "scores.csv":
        "seed,arm,acc,n_correct,n_total,acc_stderr,has_per_question,model_path\n"
        "0,7B-ES,0.75,3,4,0.1,True,/synthetic/7B-ES\n"
        "0,7B-base,0.25,1,4,0.1,True,/synthetic/7B-base\n"
        "1,7B-ES,0.5,2,4,0.1,True,/synthetic/7B-ES\n"
        "1,7B-base,0.25,1,4,0.1,True,/synthetic/7B-base\n",
    "scores_summary.csv":
        "arm,n_seeds,acc_mean,acc_std,acc_min,acc_max\n"
        "7B-ES,2,0.625,0.176777,0.5,0.75\n"
        "7B-base,2,0.25,0.0,0.25,0.25\n",
    "per_question.csv":
        "doc_id,n_seeds,7B-ES,7B-base\n"
        "0,2,1,1\n1,2,1,1\n2,2,2,0\n3,2,1,0\n",
    "mcnemar.csv":
        "seed,arm,baseline,acc_arm,acc_baseline,delta_pts,gained,lost,p_mcnemar_exact\n"
        "0,7B-ES,7B-base,0.75,0.25,50.0,2,0,0.5\n"
        "1,7B-ES,7B-base,0.5,0.25,25.0,2,1,1.0\n",
    "pairs_summary.csv":
        "arm,baseline,n_seeds,delta_mean_pts,delta_std_pts,delta_min_pts,delta_max_pts,"
        "seeds_arm_better,seeds_tied,n_questions_differing,p_wilcoxon_per_question\n"
        "7B-ES,7B-base,2,37.5,17.68,25.0,50.0,2,0,2,0.5\n",
    "position_bias.csv":
        "arm,n_decisions,pct_A,pct_B,pct_C,pct_D,spread_pts,favoured_slot\n"
        "7B-ES,8,0.0,0.0,100.0,0.0,100.0,C\n"
        "7B-base,8,25.0,25.0,25.0,25.0,0.0,A\n",
}

EXPECTED_TABLES = (
    "| Arm | mean acc over 2 permutations | sd | min | max |\n"
    "|---|---|---|---|---|\n"
    "| 7B-base | 25.00% | 0.00 | 25.00% | 25.00% |\n"
    "| 7B-ES | 62.50% | 17.68 | 50.00% | 75.00% |\n\n"
    "| Comparison | mean delta (pts) | range | arm ahead on | Wilcoxon p |\n"
    "|---|---|---|---|---|\n"
    "| 7B-ES vs 7B-base | **+37.50 ± 17.68** | +25.00 … +50.00 | 2/2 | 0.5 |\n\n"
    "| Arm | picks (A) | (B) | (C) | (D) | spread |\n"
    "|---|---|---|---|---|---|\n"
    "| 7B-base | 25.0% | 25.0% | 25.0% | 25.0% | 0.0 pts |\n"
    "| 7B-ES | 0.0% | 0.0% | 100.0% | 0.0% | 100.0 pts |\n"
)


class GpqaAnalysisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = self.root / "saved sweep"
        self.out = self.root / "output summaries"
        fixture(self.inputs)

    def run_cli(self, command="summarize", *extra):
        args = [command, "--results-dir", str(self.inputs)]
        if command == "summarize":
            args.extend(["--out", str(self.out)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([*args, *extra])
        return output.getvalue()

    def test_six_csvs_and_markdown_preserve_scores_and_statistics(self):
        before = {p: p.read_bytes() for p in self.inputs.rglob("*") if p.is_file()}
        self.run_cli("summarize", "--pair", "7B-ES:7B-base")
        self.assertEqual({p.name: p.read_text() for p in self.out.iterdir()}, EXPECTED_CSVS)
        self.assertEqual(gpqa.format_tables(self.out), EXPECTED_TABLES)
        self.assertEqual(before, {p: p.read_bytes() for p in self.inputs.rglob("*") if p.is_file()})

    def test_check_writes_nothing(self):
        before = set(self.root.rglob("*"))
        output = self.run_cli("check")
        self.assertIn("invariants OK", output)
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_installed_command_works_from_external_directory(self):
        command = Path(sys.executable).parent / "es-capacity-gpqa"
        for args in (
            ["check", "--results-dir", str(self.inputs)],
            ["summarize", "--results-dir", str(self.inputs), "--out", str(self.out),
             "--pair", "7B-ES:7B-base"],
            ["tables", "--results-dir", str(self.out)],
        ):
            result = subprocess.run([str(command), *args], cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, EXPECTED_TABLES)

    def test_unpaired_run_removes_old_pair_outputs_and_still_renders_tables(self):
        self.run_cli("summarize", "--pair", "7B-ES:7B-base")
        keep = self.out / "keep.txt"
        keep.write_text("unrelated output")
        self.run_cli()
        self.assertFalse((self.out / "mcnemar.csv").exists())
        self.assertFalse((self.out / "pairs_summary.csv").exists())
        self.assertEqual(keep.read_text(), "unrelated output")
        self.assertIn("| 7B-ES | 62.50%", gpqa.format_tables(self.out))

    def test_custom_arm_names_are_included_in_tables(self):
        for seed in self.inputs.iterdir():
            (seed / "7B-ES").rename(seed / "candidate")
            (seed / "7B-base").rename(seed / "reference")
        self.run_cli()
        table = gpqa.format_tables(self.out)
        self.assertIn("| candidate | 62.50%", table)
        self.assertIn("| reference | 25.00%", table)

    def test_inconsistent_sweeps_fail_before_creating_outputs(self):
        sweep = gpqa.load_sweep(self.inputs)
        cases = []
        bad = copy.deepcopy(sweep)
        del bad[1]["7B-ES"]
        cases.append((bad, "model arms differ"))
        bad = copy.deepcopy(sweep)
        for mapping in bad[0]["7B-ES"][1:]:
            mapping[99] = mapping[0]
        cases.append((bad, "different items"))
        bad = copy.deepcopy(sweep)
        bad[0]["7B-ES"][2][0] = ""
        cases.append((bad, "missing doc_hash"))
        bad = copy.deepcopy(sweep)
        for arm in bad[1]:
            bad[1][arm][2].update(bad[0][arm][2])
        cases.append((bad, "IDENTICAL permutation"))
        bad = copy.deepcopy(sweep)
        for data in bad[1].values():
            for mapping in data[1:]:
                mapping[99] = mapping.pop(3)
        cases.append((bad, "question IDs differ between seeds"))
        cases.append(({}, "no seeds"))
        for bad, message in cases:
            with self.subTest(message=message):
                with patch.object(gpqa, "load_sweep", return_value=bad):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            self.run_cli()
                self.assertEqual(error.exception.code, 1)
                self.assertIn(message, "; ".join(gpqa.check_invariants(bad)))
                self.assertFalse(self.out.exists())

    def test_bad_pairs_and_missing_inputs_fail_clearly(self):
        for pair in ("7B-ES", "7B-ES:", "7B-ES:7B-base:extra", "absent:7B-base"):
            with self.subTest(pair=pair):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        self.run_cli("summarize", "--pair", pair)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("missing" if "absent" in pair else "expected ARM:BASELINE", error.getvalue())
                self.assertFalse(self.out.exists())
        for path in (self.root / "absent", self.root):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(["check", "--results-dir", str(path)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("no seed" if path.exists() else "does not exist", error.getvalue())

    def test_missing_scipy_produces_installation_hint_instead_of_nan(self):
        original_import = __import__

        def import_without_scipy(name, *args, **kwargs):
            if name == "scipy.stats":
                raise ModuleNotFoundError(name="scipy")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_scipy):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli("summarize", "--pair", "7B-ES:7B-base")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("uv sync --locked --group analysis", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_legacy_launchers_delegate_and_preserve_flags(self):
        modules = []
        for script in ("analyze.py", "tables.py"):
            spec = importlib.util.spec_from_file_location(script, ROOT / "scripts/gpqa" / script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        analysis, tables = modules
        self.assertIs(analysis.load_arm, gpqa.load_arm)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            analysis.main(["--sweep-root", str(self.inputs), "--out", str(self.out),
                           "--pair", "7B-ES:7B-base"])
        self.assertIn("invariants OK", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            tables.main(["--dir", str(self.out)])
        self.assertEqual(output.getvalue(), EXPECTED_TABLES)
        with patch.object(analysis, "analyze") as run:
            analysis.main(["--sweep-root", "input", "--check"])
            run.assert_called_once_with(["check", "--results-dir", "input"])
        with patch.object(tables, "analyze") as run:
            tables.main([])
            run.assert_called_once_with(["tables", "--results-dir", "results/gpqa"])

    def test_help_imports_no_optional_analysis_or_model_libraries(self):
        code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy', 'numpy', 'torch', 'vllm', 'lm_eval', 'datasets', 'transformers'}:
            raise AssertionError(f'CLI help imported {fullname}')
sys.meta_path.insert(0, BlockOptional())
from es_capacity.cli.gpqa import main
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("summarize", result.stdout)


if __name__ == "__main__":
    unittest.main()
