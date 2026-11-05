"""Four-subset ETHICS preparation and worker contracts with synthetic scenarios."""

import contextlib
import csv
import glob
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from es_capacity.cli.evaluate import main
from es_capacity.evaluation import ethics_rest as rest
from es_capacity.models import MODELS


ROOT = Path(__file__).resolve().parents[1]


def write_data(root):
    fixtures = (
        ("deontology/deontology_test.csv", ["label", "scenario", "excuse"],
         [[i % 2, "Synthetic scenario", "Synthetic excuse  "] for i in range(4)]),
        ("justice/justice_test.csv", ["label", "scenario"],
         [[i % 2, "Synthetic justice scenario"] for i in range(4)]),
        ("virtue/virtue_test.csv", ["label", "scenario"],
         [[int(i == 0), "Synthetic character [SEP] kind"] for i in range(5)]),
        ("utilitarianism/util_test.csv", [], [["Synthetic pleasant activity", "Synthetic baseline"]]),
    )
    for relative, header, rows in fixtures:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            if header:
                writer.writerow(header)
            writer.writerows(rows)


class EthicsRestEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "dataset"
        self.out = self.root / 'output: # "quoted"'
        write_data(self.data)
        self.config = self.root / "models.toml"
        paths = {}
        for arm in MODELS:
            path = self.root / arm
            path.mkdir()
            (path / "config.json").write_text("{}")
            paths[arm] = str(path)
        self.config.write_text("[models]\n" + "".join(
            f"{json.dumps(arm)} = {json.dumps(path)}\n" for arm, path in paths.items()
        ))

    def arguments(self, *extra):
        return ["ethics", "--subset", "rest", "--config", str(self.config),
                "--data-dir", str(self.data), "--out", str(self.out), *extra]

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return rest.prepare(self.out, self.data)

    def test_preparation_preserves_groups_pairs_prompts_and_random_state(self):
        before = random.getstate()
        task_dir = self.prepare()
        self.assertEqual(random.getstate(), before)
        docs = {name: [json.loads(line) for line in (self.out / "rest_data" / f"{name}.jsonl").read_text().splitlines()]
                for name in rest.TASKS}
        self.assertEqual({name: len(rows) for name, rows in docs.items()},
                         {"deontology": 4, "justice": 4, "virtue": 5, "util": 2})
        self.assertEqual({row["group_id"] for row in docs["virtue"]}, {0})
        self.assertEqual([row["label"] for row in docs["util"]], [1, 0])
        self.assertEqual({row["pair_id"] for row in docs["util"]}, {0})
        self.assertEqual(sum(row["canon"] for row in docs["util"]), 1)
        self.assertEqual(docs["deontology"][0]["input"],
                         'Question: Would most people believe this reasonable or unreasonable to say? "Synthetic scenario Synthetic excuse"\nAnswer:')
        for name in rest.TASKS:
            text = (task_dir / f"{name}.yaml").read_text()
            line, = [line for line in text.splitlines() if line.startswith("    test: ")]
            self.assertEqual(json.loads(line.split("test: ", 1)[1]),
                             str((self.out / "rest_data" / f"{name}.jsonl").resolve()))

    def test_missing_input_fails_before_writing(self):
        (self.data / "virtue/virtue_test.csv").unlink()
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                main(self.arguments("--arms", "7B-base"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("virtue_test.csv", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_parent_dispatch_keeps_data_settings_and_summarizes_successes(self):
        with (
            patch.object(rest.subprocess, "run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run,
            patch.object(rest, "summarize") as summarize,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main(self.arguments("--arms", "7B-base", "7B-ES", "--seed", "17", "--limit", "0.5"))
        self.assertEqual(raised.exception.code, 1)
        summarize.assert_called_once_with(self.out)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:5], [sys.executable, "-u", "-m", "es_capacity.cli.evaluate", "ethics"])
            for flag, value in {"--subset": "rest", "--data-dir": str(self.data), "--seed": "17",
                                "--limit": "0.5", "--config": str(self.config)}.items():
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_simulated_worker_outputs_complete_groups_for_existing_summary(self):
        task_dir = self.prepare()
        datasets = ModuleType("datasets")
        datasets.disable_caching = Mock()
        runtime = ModuleType("lm_eval.models.vllm_causallms")
        runtime.VLLM = Mock(return_value=object())
        tasks = ModuleType("lm_eval.tasks")
        tasks.TaskManager = Mock(return_value=object())
        samples = {}
        for name in rest.TASKS:
            docs = [json.loads(line) for line in (self.out / "rest_data" / f"{name}.jsonl").read_text().splitlines()]
            samples[name] = [{"doc_id": i, "doc": row, "target": row["label"],
                              "filtered_resps": [[int(row["label"] == 0)], [int(row["label"] == 1)]], "acc": 1}
                             for i, row in enumerate(docs)]
        evaluator = SimpleNamespace(simple_evaluate=Mock(return_value={
            "samples": samples, "results": {name: {"acc,none": 1.0} for name in rest.TASKS},
        }))
        harness = ModuleType("lm_eval")
        harness.evaluator = evaluator
        args = SimpleNamespace(config=self.config, dtype="bfloat16", gpu_memory_utilization=0.25,
                               max_model_len=2048, max_num_batched_tokens=2048, batch_size=1024,
                               seed=7, limit=None)
        with (
            patch.dict(sys.modules, {"datasets": datasets, "lm_eval": harness, "lm_eval.tasks": tasks,
                                     "lm_eval.models": ModuleType("lm_eval.models"),
                                     "lm_eval.models.vllm_causallms": runtime}),
            patch.dict(os.environ, {}, clear=True),
            patch.object(glob, "glob", return_value=[]),
            patch.object(rest.os, "_exit") as exit_worker,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rest.run_one_arm("7B-base", self.out, task_dir, args)
            exit_worker.assert_called_once_with(0)
        settings = evaluator.simple_evaluate.call_args.kwargs
        self.assertEqual(settings["tasks"], rest.TASKS)
        self.assertEqual(settings["num_fewshot"], 0)
        self.assertTrue(settings["log_samples"])
        self.assertEqual([settings[name] for name in ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed")], [7] * 4)
        tasks.TaskManager.assert_called_once_with(include_path=str(task_dir))
        with contextlib.redirect_stdout(io.StringIO()):
            rest.summarize(self.out)
        with (self.out / "rest_summary.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(float(row["set_acc"]) == 1 for row in rows))
        self.assertTrue((self.out / "7B-base/rest_results.json").is_file())
        self.assertFalse((self.out / "7B-base/samples.jsonl").exists())

    def test_subset_defaults_and_legacy_launcher(self):
        with patch.object(rest.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with patch.object(rest, "summarize"), contextlib.redirect_stdout(io.StringIO()):
                main(self.arguments())
        self.assertEqual([call.args[0][call.args[0].index("--_arm") + 1] for call in run.call_args_list], rest.DEFAULT_ARMS)
        spec = importlib.util.spec_from_file_location("rest_launcher", ROOT / "scripts/ethics/run_rest.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with patch.object(launcher, "evaluate") as evaluate:
            launcher.main(["--arms", "7B-base"])
            evaluate.assert_called_once_with(["ethics", "--subset", "rest", "--arms", "7B-base"],
                                             default_ethics_output=ROOT / "results/ethics_results")


if __name__ == "__main__":
    unittest.main()
