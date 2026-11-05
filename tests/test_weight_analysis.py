"""Saved-histogram calculations, figure output and CLI compatibility."""

import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from es_capacity.analysis import weights
from es_capacity.cli.weights import main


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("raw_weights", "delta_pooled", "weight_change", "delta_by_module", "delta_by_depth")


def fixture(root, scale="7B"):
    """Small synthetic CSVs in compare.py's format; no checkpoint data."""
    for arm, sigma in (("ES", 0.001), ("RL", 0.00002)):
        directory = root / f"{scale}-{arm}"
        directory.mkdir(parents=True)
        stats, hist = [], []
        for series, grouping, group, mean, std in (
            ("base", "pooled", "all", 0.0, 0.02),
            ("trained", "pooled", "all", 0.001, 0.021),
            ("delta", "pooled", "all", 0.0, sigma),
            ("delta", "module", "q_proj", 0.0, sigma),
            ("delta", "depth", "0", 0.0, sigma),
        ):
            grid_std = 0.02 if series in ("base", "trained") else std
            lo, hi = -6 * grid_std, 6 * grid_std
            width = (hi - lo) / weights.N_BINS
            centers = lo + (np.arange(weights.N_BINS) + 0.5) * width
            counts = np.rint(300 * np.exp(-0.5 * ((centers - mean) / std) ** 2)).astype(int)
            prefix = [series, grouping, group]
            stats.append([*prefix, int(counts.sum()) + 3, mean, std, lo, hi, 0.0, 3.5, 0.0])
            hist.append([*prefix, "underflow", "", lo, 1])
            for index, count in enumerate(counts):
                if count:
                    hist.append([*prefix, index, lo + index * width, lo + (index + 1) * width, int(count)])
            hist.append([*prefix, "overflow", hi, "", 2])
        with (directory / "stats.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["series", "grouping", "group", "n", "mean", "std", "min", "max",
                             "excess_kurtosis", "p999_over_sigma", "zero_frac"])
            writer.writerows(stats)
        with (directory / "hist.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["series", "grouping", "group", "bin", "left", "right", "count"])
            writer.writerows(hist)


class WeightAnalysisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = self.root / "saved weights"
        self.out = self.root / "figure outputs"
        fixture(self.inputs)

    def call_cli(self, scale="7B"):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--results-dir", str(self.inputs), "--scale", scale, "--out", str(self.out)])
        return output.getvalue()

    def test_histograms_preserve_grid_counts_tails_and_density_units(self):
        directory = self.inputs / "7B-ES"
        stats = weights.read_stats(directory / "stats.csv")
        hist = weights.read_hist(directory / "hist.csv", stats)
        base = hist[("base", "pooled", "all")]
        trained = hist[("trained", "pooled", "all")]
        np.testing.assert_array_equal(base["centers"], trained["centers"])
        self.assertNotEqual(base["mean"], trained["mean"])
        self.assertNotEqual(base["std"], trained["std"])
        delta = hist[("delta", "pooled", "all")]
        self.assertEqual(delta["counts"].sum() + delta["under"] + delta["over"], delta["n"])
        self.assertEqual((delta["under"], delta["over"]), (1, 2))
        self.assertAlmostEqual(np.nansum(weights.density(delta) * delta["width"]), delta["counts"].sum())
        self.assertTrue(np.isnan(weights.masked(np.array([0, 1]))[0]))
        self.assertAlmostEqual(np.sum(weights.gaussian_ref(delta) * delta["width"]) / delta["n"], 1.0, places=7)

    def test_rebin_preserves_mass_and_uses_mean_bin_centers(self):
        hist = {"counts": np.arange(512, dtype=float), "centers": np.arange(512) * 0.5, "width": 0.5}
        result = weights.rebin(hist, 4)
        self.assertEqual(result["counts"][0], 0 + 1 + 2 + 3)
        self.assertEqual(result["centers"][0], 0.75)
        self.assertEqual(result["width"], 2)
        self.assertEqual(result["counts"].sum(), hist["counts"].sum())
        self.assertEqual(len(result["counts"]), 128)

    def test_installed_command_renders_five_pngs_outside_checkout_without_changing_inputs(self):
        before = {p: p.read_bytes() for p in self.inputs.rglob("*.csv")}
        command = Path(sys.executable).parent / "es-capacity-weights"
        result = subprocess.run([str(command), "--results-dir", str(self.inputs),
                                 "--scale", "7B", "--out", str(self.out)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("wrote 5 figures", result.stdout)
        self.assertEqual({p.name for p in self.out.iterdir()}, {f"{name}_7B.png" for name in FIGURES})
        for path in self.out.iterdir():
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), path.name)
        self.assertEqual(before, {p: p.read_bytes() for p in self.inputs.rglob("*.csv")})

    def test_missing_or_incomplete_inputs_fail_before_creating_output(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                self.call_cli("absent")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("absent-ES/stats.csv", error.getvalue())
        self.assertFalse(self.out.exists())
        (self.inputs / "7B-RL/hist.csv").unlink()
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):
                self.call_cli()
        self.assertIn("7B-RL/hist.csv", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_missing_pooled_histogram_is_reported_before_rendering(self):
        path = self.inputs / "7B-ES/hist.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(row for row in lines if not row.startswith("delta,pooled,")) + "\n")
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                self.call_cli()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("pooled delta histogram", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_unsafe_scale_and_missing_directory_are_rejected(self):
        for scale in ("../7B", "..", "7B/ES", "7B\\ES"):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit):
                    self.call_cli(scale)
            self.assertIn("single directory-name", error.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):
                main(["--results-dir", str(self.root / "missing"), "--scale", "7B", "--out", str(self.out)])
        self.assertIn("does not exist", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_missing_analysis_dependencies_explain_installation(self):
        for name in ("numpy", "matplotlib"):
            with patch("es_capacity.cli.weights.render_figures", side_effect=ModuleNotFoundError(name=name)):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        self.call_cli()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("uv sync --locked --group analysis", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_legacy_launcher_preserves_default_and_explicit_locations(self):
        spec = importlib.util.spec_from_file_location("weight_plot", ROOT / "scripts/weights/plot.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertIs(launcher.read_hist, weights.read_hist)
        for extra, directory in (([], Path("results/weights")), (["--dir", str(self.inputs)], self.inputs)):
            with patch.object(launcher, "plot") as plot:
                launcher.main(["--scale", "7B", *extra])
                plot.assert_called_once_with(["--scale", "7B", "--results-dir", str(directory),
                                              "--out", str(directory / "figures")])

    def test_help_loads_no_plotting_or_model_libraries_and_does_not_change_environment(self):
        code = """
import importlib.abc
import os
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'numpy', 'scipy', 'torch', 'vllm', 'lm_eval', 'datasets', 'transformers'}:
            raise AssertionError(f'CLI help imported {fullname}')
sys.meta_path.insert(0, BlockOptional())
before = dict(os.environ)
from es_capacity.cli.weights import main
assert dict(os.environ) == before
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--results-dir", result.stdout)
        self.assertIn("--out", result.stdout)


if __name__ == "__main__":
    unittest.main()
