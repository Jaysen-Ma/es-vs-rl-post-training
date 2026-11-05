"""Numerical references and CLI contracts for saved-score pass@k analysis."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from es_capacity.analysis import passk
from es_capacity.cli.passk import main


ROOT = Path(__file__).resolve().parents[1]


class PassAtKReferenceTests(unittest.TestCase):
    def test_estimator_matches_independent_combinatorial_reference(self):
        # evaluate.py in limit-of-RLVR uses this formula via a product estimator:
        # 1 - C(n-c, k) / C(n, k). Use exact integer combinations as our reference.
        # No external checkout, model dependencies or copied upstream code needed.
        for n in [*range(1, 17), 31, 32, 128, 512]:
            correct_counts = range(n + 1) if n <= 16 else sorted({0, 1, n // 2, n - 1, n})
            for correct in correct_counts:
                with self.subTest(n=n, correct=correct):
                    ks, values = passk.pass_at_k_curve({0: [True] * correct + [False] * (n - correct)})
                    for k, value in zip(ks, values):
                        expected = 1 - math.comb(n - correct, k) / math.comb(n, k)
                        self.assertAlmostEqual(value, expected, delta=2e-13)

    def test_curve_averages_questions_and_preserves_the_power_of_two_grid(self):
        ks, values = passk.pass_at_k_curve({
            3: [False] * 4, 2: [True] * 4,
            1: [False] * 4, 0: [True, False, False, False],
        })
        self.assertEqual(ks, [1, 2, 4])
        self.assertEqual(values, [5 / 16, 3 / 8, 1 / 2])
        self.assertEqual(passk.pass_at_k_curve({0: [False] * 17})[0], [1, 2, 4, 8, 16])

    def test_all_correct_and_all_incorrect_have_exact_extreme_values(self):
        for n in (1, 3, 8, 128, 512):
            for correct in (False, True):
                with self.subTest(n=n, correct=correct):
                    ks, values = passk.pass_at_k_curve({0: [correct] * n})
                    self.assertEqual(values, [float(correct)] * len(ks))


class PassAtKCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = self.root / "saved results"
        self.out = self.root / "output summaries"

    def write_inputs(self):
        for arm, rows in (
            ("base", [{"idx": 0, "score": [True, False]}, {"idx": 1, "score": [False, False]}]),
            ("trained", [{"idx": 0, "score": [False, False]}, {"idx": 1, "score": [True, True]}]),
        ):
            directory = self.inputs / arm / "math500"
            directory.mkdir(parents=True)
            (directory / "shard.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def arguments(self, *extra):
        return ["--results-dir", str(self.inputs), "--out", str(self.out),
                "--label", "base", "--label", "trained", *extra]

    def test_explicit_paths_write_summary_without_modifying_inputs(self):
        self.write_inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments("--benchmark", "math500"))
        result = json.loads((self.out / "math500_summary.json").read_text())
        self.assertEqual(result["baseline"], "base")
        self.assertEqual(result["n_sampling"], {"base": 2, "trained": 2})
        self.assertEqual(result["pass_at_k"], {"base": [0.25, 0.5], "trained": [0.5, 0.5]})
        self.assertEqual(list(self.inputs.rglob("*_summary.json")), [])

    def test_default_benchmarks_skip_missing_directories_and_process_available_data(self):
        self.write_inputs()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(self.arguments())
        self.assertIn("Skipping aime24", output.getvalue())
        self.assertEqual([path.name for path in self.out.iterdir()], ["math500_summary.json"])

    def test_missing_explicit_benchmark_fails_without_creating_outputs(self):
        self.write_inputs()
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                main(self.arguments("--benchmark", "aime24"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("aime24: no directory", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_bad_arguments_and_missing_inputs_fail_clearly(self):
        cases = (
            (self.arguments(), "does not exist"),
            (self.arguments("--label", "base"), "must be unique"),
            (self.arguments("--baseline", "unknown"), "not among"),
            (self.arguments("--benchmark", "unknown"), "unknown benchmarks"),
            (["--results-dir", str(self.root), "--out", str(self.out), "--label", "../base", "--label", "trained"], "without path separators"),
            (["--results-dir", str(self.root), "--out", str(self.out), "--label", "base"], "at least 2"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, error.getvalue())
        self.assertFalse(self.out.exists())

    def test_empty_input_directory_does_not_report_success(self):
        self.inputs.mkdir()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                main(self.arguments())
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("no benchmarks", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_bundled_example_has_known_curves_and_each_coverage_category(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--example", "--out", str(self.out), "--plot"])
        self.assertIn("not model evaluation results", output.getvalue())
        result = json.loads((self.out / "synthetic_summary.json").read_text())
        self.assertEqual(result["pass_at_k"]["reference"], [5 / 16, 3 / 8, 1 / 2])
        for value, expected in zip(result["pass_at_k"]["candidate"], [3 / 8, 11 / 24, 1 / 2]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(set(result["four_way_breakdown"]["candidate"]["counts"].values()), {1})
        self.assertGreater((self.out / "synthetic_passk.png").stat().st_size, 0)

    def test_missing_analysis_dependencies_have_installation_hint(self):
        self.write_inputs()
        with (
            patch.object(passk, "analyze_benchmark", side_effect=ModuleNotFoundError("No module named 'numpy'", name="numpy")),
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            with self.assertRaises(SystemExit) as raised:
                main(self.arguments("--benchmark", "math500"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("uv sync --locked --group analysis", error.getvalue())

    def test_legacy_launcher_delegates_and_retains_run_tag_output_location(self):
        self.write_inputs()
        spec = importlib.util.spec_from_file_location("legacy_passk", ROOT / "scripts/passk/analyze.py")
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        self.assertIs(legacy.pass_at_k_curve, passk.pass_at_k_curve)
        self.assertIs(legacy.analyze_benchmark, passk.analyze_benchmark)
        # Arrange the old results/RUN layout while leaving the repository untouched.
        results = self.root / "results"
        results.mkdir()
        self.inputs.rename(results / "my-run")
        with patch.object(legacy, "REPO_ROOT", str(self.root)), contextlib.redirect_stdout(io.StringIO()):
            legacy.main(["--run-tag", "my-run", "--label", "base", "--label", "trained"])
        self.assertTrue((results / "my-run/math500_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
