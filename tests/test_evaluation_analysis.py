"""CPU regressions for evaluation setup, comparisons, and histogram loading."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


passk = load_script("passk_analysis", "scripts/passk/analyze.py")
gpqa = load_script("gpqa_analysis", "scripts/gpqa/analyze.py")
ethics = load_script("ethics_runner", "scripts/ethics/run.py")
rest = load_script("ethics_rest_runner", "scripts/ethics/run_rest.py")
weights = load_script("weight_plot", "scripts/weights/plot.py")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fresh_ethics_download_creates_parent_directory(self):
        def download(url, destination):
            data = b'label,input\n0,A harmless scenario\n1,A harmful scenario\n'
            member = tarfile.TarInfo("ethics/commonsense/cm_test.csv")
            member.size = len(data)
            with tarfile.open(destination, "w") as archive:
                archive.addfile(member, io.BytesIO(data))

        with patch.object(ethics.urllib.request, "urlretrieve", side_effect=download):
            path = ethics.ensure_test_jsonl(self.root / "new" / "nested", None)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["label"] for row in rows], [0, 1])

    def analyze_passk(self, base, trained):
        dirs = {}
        for arm, scores in (("base", base), ("trained", trained)):
            directory = self.root / arm
            directory.mkdir()
            (directory / "shard.jsonl").write_text("".join(
                json.dumps({"idx": idx, "score": values}) + "\n"
                for idx, values in scores.items()
            ))
            dirs[arm] = str(directory)
        prefix = self.root / "comparison"
        with contextlib.redirect_stdout(io.StringIO()):
            passk.analyze_benchmark(dirs, "base", "Test", str(prefix), False)
        return json.loads(prefix.with_name("comparison_summary.json").read_text())

    def test_passk_rejects_missing_questions_before_writing_summary(self):
        with self.assertRaisesRegex(ValueError, "question IDs differ"):
            self.analyze_passk({0: [True] * 4, 1: [False] * 4}, {0: [True] * 4})
        self.assertFalse((self.root / "comparison_summary.json").exists())

    def test_passk_rejects_different_ids_even_with_equal_counts(self):
        with self.assertRaisesRegex(ValueError, "question IDs differ"):
            self.analyze_passk({0: [True] * 4}, {1: [True] * 4})

    def test_unequal_budgets_preserve_curves_and_omit_breakdown(self):
        result = self.analyze_passk({0: [False, False, False, True]}, {0: [False, False]})
        self.assertEqual(result["ks"], {"base": [1, 2, 4], "trained": [1, 2]})
        self.assertEqual(result["four_way_breakdown"], {})

    def test_equal_budgets_keep_exact_solvability_counts(self):
        result = self.analyze_passk(
            {0: [True, False], 1: [False, False]},
            {0: [False, False], 1: [True, False]},
        )
        counts = result["four_way_breakdown"]["trained"]["counts"]
        self.assertEqual(counts["base_solved_trained_failed"], 1)
        self.assertEqual(counts["base_failed_trained_solved"], 1)
        np.testing.assert_allclose(result["pass_at_k"]["base"], [0.25, 0.5])

    def test_breakdown_rejects_unequal_budgets_directly(self):
        with self.assertRaisesRegex(ValueError, "sample budgets"):
            passk.four_way_breakdown({0: [False] * 4}, {0: [False] * 2})

    def test_empty_sample_lists_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            passk.pass_at_k_curve({0: []})

    def write_gpqa_run(self, stamp, predictions, expected=None):
        directory = self.root / "gpqa"
        directory.mkdir(exist_ok=True)
        results = {
            "results": {gpqa.TASK: {"acc,none": sum(predictions) / len(predictions)}},
            "n-samples": {gpqa.TASK: {"effective": expected or len(predictions)}},
        }
        (directory / f"results_{stamp}.json").write_text(json.dumps(results))
        samples = directory / f"samples_{gpqa.TASK}_{stamp}.jsonl"
        samples.write_text("".join(json.dumps({
            "doc_id": idx, "acc": correct, "doc_hash": f"hash{idx}",
            "filtered_resps": [[0], [1], [0], [0]],
        }) + "\n" for idx, correct in enumerate(predictions)))
        return directory, samples

    def test_gpqa_uses_only_samples_matching_latest_results(self):
        self.write_gpqa_run("2026-01-01", [1, 1])
        directory, _ = self.write_gpqa_run("2026-01-02", [0])
        _, correct, _, picked = gpqa.load_arm(directory)
        self.assertEqual(correct, {0: 0})
        self.assertEqual(picked, {0: 1})

    def test_gpqa_rejects_interrupted_latest_sample_dump(self):
        self.write_gpqa_run("2026-01-01", [1, 1])
        directory, _ = self.write_gpqa_run("2026-01-02", [0], expected=2)
        with self.assertRaisesRegex(ValueError, "incomplete samples"):
            gpqa.load_arm(directory)

    def test_gpqa_does_not_fall_back_to_old_samples(self):
        self.write_gpqa_run("2026-01-01", [1, 1])
        directory, samples = self.write_gpqa_run("2026-01-02", [0, 0])
        samples.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "selected run"):
            gpqa.load_arm(directory)

    def test_gpqa_rejects_duplicate_question_ids(self):
        directory, samples = self.write_gpqa_run("2026-01-01", [1])
        samples.write_text(samples.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
            gpqa.load_arm(directory)

    def test_contrast_sets_require_every_distinct_question(self):
        for task, size in rest.GROUP_SIZE.items():
            with self.subTest(task=task):
                rows = [{"group_id": 5, "doc_id": i, "correct": True}
                        for i in range(size)]
                self.assertEqual(rest._units(rows[:-1], task), {})
                self.assertEqual(rest._units(rows, task), {5: True})
                self.assertEqual(rest._units(rows[:-1] + [rows[0]], task), {})
                rows[-1]["correct"] = False
                self.assertEqual(rest._units(rows, task), {5: False})

    def test_util_pairs_keep_ids_and_skip_incomplete_pairs(self):
        rows = [
            {"pair_id": 7, "gold": 0, "logodds": -2},
            {"pair_id": 7, "gold": 1, "logodds": -1},
            {"pair_id": 8, "gold": 1, "logodds": 0},
        ]
        self.assertEqual(rest._units(rows, "util"), {7: True})

    def write_rest_samples(self, arm, group_id, count):
        directory = self.root / arm
        directory.mkdir()
        records = [{
            "subtask": "deontology", "doc_id": group_id * 4 + i,
            "target": 1, "filtered_resps": [[-2], [-1]],
            "doc": {"group_id": group_id},
        } for i in range(count)]
        (directory / "rest_samples.jsonl").write_text(
            "".join(json.dumps(rec) + "\n" for rec in records)
        )

    def summarize_rest(self):
        with patch.object(rest, "TASKS", ["deontology"]), contextlib.redirect_stdout(io.StringIO()):
            rest.summarize(self.root)
        with (self.root / "rest_summary.csv").open() as fh:
            return list(csv.DictReader(fh))

    def test_summary_handles_no_complete_contrast_sets(self):
        self.write_rest_samples("7B-base", 0, 1)
        result = self.summarize_rest()[0]
        self.assertEqual(result["set_n"], "0")
        self.assertEqual(result["set_acc"], "")

    def test_summary_does_not_pair_different_groups_of_equal_length(self):
        self.write_rest_samples("7B-base", 0, 4)
        self.write_rest_samples("7B-ES", 1, 4)
        rows = {r["arm"]: r for r in self.summarize_rest()}
        self.assertEqual(rows["7B-ES"]["set_n"], "1")
        self.assertEqual(rows["7B-ES"]["set_p"], "")

    def test_histograms_use_stored_grid_even_with_different_statistics(self):
        path = self.root / "hist.csv"
        path.write_text(
            "series,grouping,group,bin,left,right,count\n"
            "trained,pooled,all,underflow,,-6,0\n"
            "trained,pooled,all,256,0,0.0234375,10\n"
            "trained,pooled,all,overflow,6,,0\n"
        )
        key = ("trained", "pooled", "all")
        histogram = weights.read_hist(path, {key: {"mean": 2.0, "std": 3.0, "n": 10}})[key]
        self.assertAlmostEqual(histogram["width"], 12 / 512)
        self.assertAlmostEqual(histogram["centers"][256], 0.01171875)
        self.assertEqual(histogram["counts"].sum(), 10)
        self.assertEqual(np.count_nonzero(histogram["counts"]), 1)
        self.assertEqual(histogram["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
