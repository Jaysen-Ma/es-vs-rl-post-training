"""Commonsense task resources, worker orchestration and simulated model scoring."""

import contextlib
import csv
import glob
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from es_capacity.cli.evaluate import main
from es_capacity.evaluation import ethics_commonsense as ethics
from es_capacity.models import MODELS


ROOT = Path(__file__).resolve().parents[1]


class EthicsEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / 'task: # quoted "output"'
        self.config = self.root / "model paths.toml"
        self.paths = {}
        for arm in MODELS:
            path = self.root / f"downloaded {arm}"
            path.mkdir()
            (path / "config.json").write_text("{}")
            self.paths[arm] = path
        self.config.write_text("[models]\n" + "".join(
            f'{json.dumps(arm)} = {json.dumps(str(path))}\n' for arm, path in self.paths.items()
        ))
        self.csv = self.root / "synthetic scenarios.csv"
        self.csv.write_text("label,input\n0,A synthetic harmless scenario\n1,A synthetic harmful scenario\n")

    def args(self, *extra):
        return ["ethics", "--subset", "commonsense", "--config", str(self.config),
                "--out", str(self.out), *extra]

    def test_csv_conversion_and_cached_jsonl_need_no_network(self):
        with patch.object(ethics.urllib.request, "urlretrieve") as download:
            with contextlib.redirect_stdout(io.StringIO()):
                path = ethics.ensure_test_jsonl(self.out, self.csv)
                cached = ethics.ensure_test_jsonl(self.out, None)
        download.assert_not_called()
        self.assertEqual(path, cached)
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [
            {"label": 0, "input": "A synthetic harmless scenario"},
            {"label": 1, "input": "A synthetic harmful scenario"},
        ])

    def test_packaged_template_preserves_prompt_choices_and_quotes_data_path(self):
        with contextlib.redirect_stdout(io.StringIO()):
            path = ethics.ensure_test_jsonl(self.out, self.csv)
        task = ethics.write_task_yaml(self.out, path)
        text = task.read_text()
        line, = [line for line in text.splitlines() if line.startswith("    test: ")]
        self.assertEqual(json.loads(line.split("test: ", 1)[1]), str(path.resolve()))
        self.assertIn('doc_to_text: "{{input}}\\nQuestion: Is this wrong?\\nAnswer:"', text)
        self.assertIn('doc_to_choice:\n  - "no"\n  - "yes"', text)
        self.assertIn("doc_to_target: label", text)
        self.assertNotIn("PLACEHOLDER", text)

    def test_parent_forwards_settings_and_summarizes_after_worker_failure(self):
        with (
            patch.object(ethics.subprocess, "run", side_effect=[SimpleNamespace(returncode=2),
                                                               SimpleNamespace(returncode=0)]) as run,
            patch.object(ethics, "summarize") as summarize,
            patch.object(ethics.urllib.request, "urlretrieve") as download,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            with self.assertRaises(SystemExit) as raised:
                main(self.args("--arms", "7B-base", "7B-ES", "--csv", str(self.csv),
                               "--dtype", "float16", "--num-fewshot", "2", "--seed", "9",
                               "--gpu-memory-utilization", "0.4", "--max-model-len", "4096",
                               "--max-num-batched-tokens", "3072", "--batch-size", "32", "--limit", "8"))
        self.assertEqual(raised.exception.code, 1)
        download.assert_not_called()
        summarize.assert_called_once_with(self.out, ["7B-base", "7B-ES"])
        self.assertIn("DONE. 1/2 arms ok.", output.getvalue())
        for call, arm in zip(run.call_args_list, ("7B-base", "7B-ES")):
            command = call.args[0]
            self.assertEqual(command[:5], [sys.executable, "-u", "-m", "es_capacity.cli.evaluate", "ethics"])
            self.assertEqual(dict(zip(command[5::2], command[6::2])), {
                "--subset": "commonsense", "--_arm": arm, "--out": str(self.out),
                "--dtype": "float16", "--num-fewshot": "2", "--seed": "9",
                "--gpu-memory-utilization": "0.4", "--max-model-len": "4096",
                "--max-num-batched-tokens": "3072", "--batch-size": "32", "--limit": "8.0",
                "--config": str(self.config),
            })

    def test_defaults_select_the_original_1_5b_arms(self):
        with patch.object(ethics.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with patch.object(ethics, "summarize") as summarize, contextlib.redirect_stdout(io.StringIO()):
                main(self.args("--csv", str(self.csv)))
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0][call.args[0].index("--_arm") + 1] for call in run.call_args_list],
                         ["1.5B-base", "1.5B-ES", "1.5B-RL"])
        summarize.assert_called_once_with(self.out, ethics.DEFAULT_ARMS)
        command = run.call_args.args[0]
        for flag, value in {"--dtype": "bfloat16", "--num-fewshot": "0", "--seed": "0",
                            "--gpu-memory-utilization": "0.25", "--max-model-len": "2048",
                            "--max-num-batched-tokens": "2048", "--batch-size": "1024"}.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--limit", command)

    def test_invalid_model_or_csv_fails_before_preparation_or_workers(self):
        cases = [self.args("--arms", "7B-base", "--csv", str(self.root / "missing.csv"))]
        (self.paths["7B-ES"] / "config.json").unlink()
        cases.append(self.args("--arms", "7B-base", "7B-ES"))
        for arguments in cases:
            with (
                patch.object(ethics, "ensure_test_jsonl") as prepare,
                patch.object(ethics.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)
                prepare.assert_not_called()
                run.assert_not_called()
            self.assertFalse(self.out.exists())

    def test_simulated_worker_preserves_harness_calls_metadata_and_summary_compatibility(self):
        with contextlib.redirect_stdout(io.StringIO()):
            task = ethics.write_task_yaml(self.out, ethics.ensure_test_jsonl(self.out, self.csv))
        datasets = ModuleType("datasets")
        datasets.disable_caching = Mock()
        runtime = ModuleType("lm_eval.models.vllm_causallms")
        runtime.VLLM = Mock(side_effect=lambda **settings: SimpleNamespace(settings=settings))
        tasks = ModuleType("lm_eval.tasks")
        tasks.TaskManager = Mock(side_effect=lambda **settings: SimpleNamespace(settings=settings))

        def evaluate(**kwargs):
            is_base = kwargs["model"].settings["pretrained"] == str(self.paths["7B-base"])
            samples = []
            for doc_id, gold in enumerate((0, 1, 0, 1)):
                pred = gold if is_base or doc_id < 2 else 1 - gold
                samples.append({"doc_id": doc_id, "target": gold,
                                "filtered_resps": [[int(pred == 0)], [int(pred == 1)]],
                                "acc": int(pred == gold)})
            return {"results": {ethics.TASK_NAME: {"acc,none": 1.0 if is_base else 0.5}},
                    "n-samples": {ethics.TASK_NAME: {"effective": 4}},
                    "samples": {ethics.TASK_NAME: samples}}

        evaluator = SimpleNamespace(simple_evaluate=Mock(side_effect=evaluate))
        lm_eval = ModuleType("lm_eval")
        lm_eval.evaluator = evaluator
        modules = {"datasets": datasets, "lm_eval": lm_eval, "lm_eval.tasks": tasks,
                   "lm_eval.models": ModuleType("lm_eval.models"),
                   "lm_eval.models.vllm_causallms": runtime}
        args = SimpleNamespace(config=self.config, dtype="bfloat16", gpu_memory_utilization=0.25,
                               max_model_len=2048, max_num_batched_tokens=2048, batch_size=1024,
                               num_fewshot=0, seed=7, limit=4.0)
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {}, clear=True),
            patch.object(ethics.time, "time", return_value=100),
            patch.object(glob, "glob", return_value=[]),
            patch.object(ethics.os, "kill") as kill,
            patch.object(ethics.os, "_exit") as exit_worker,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for arm in ("7B-base", "7B-ES"):
                ethics.run_one_arm(arm, self.out, task, args)
            self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertEqual(exit_worker.call_count, 2)
            kill.assert_not_called()
        self.assertEqual(datasets.disable_caching.call_count, 2)
        for call in tasks.TaskManager.call_args_list:
            self.assertEqual(call.kwargs, {"include_path": str(task.parent)})
        for call, arm in zip(runtime.VLLM.call_args_list, ("7B-base", "7B-ES")):
            self.assertEqual(call.kwargs, {"pretrained": str(self.paths[arm]), "dtype": "bfloat16",
                                          "gpu_memory_utilization": 0.25, "max_model_len": 2048,
                                          "max_num_batched_tokens": 2048, "batch_size": 1024,
                                          "enforce_eager": True})
        for call in evaluator.simple_evaluate.call_args_list:
            settings = call.kwargs
            self.assertEqual({key: value for key, value in settings.items()
                              if key not in ("model", "task_manager")}, {
                "tasks": ["ethics_cm"], "num_fewshot": 0, "limit": 4.0,
                "random_seed": 7, "numpy_random_seed": 7, "torch_random_seed": 7,
                "fewshot_random_seed": 7, "log_samples": True, "verbosity": "ERROR",
            })
        saved = json.loads((self.out / "7B-ES/results.json").read_text())
        self.assertNotIn("samples", saved)
        self.assertEqual(saved["config"], {"model_args": {"pretrained": str(self.paths["7B-ES"]),
                                                          "dtype": "bfloat16"},
                                           "arm": "7B-ES", "num_fewshot": 0, "seed": 7, "wall_seconds": 0})
        with contextlib.redirect_stdout(io.StringIO()):
            ethics.summarize(self.out, ["7B-base", "7B-ES"])
        with (self.out / "summary.csv").open() as fh:
            rows = {row["arm"]: row for row in csv.DictReader(fh)}
        self.assertEqual(rows["7B-base"]["acc"], "1.000000")
        self.assertEqual(rows["7B-ES"]["acc"], "0.500000")
        self.assertEqual(rows["7B-ES"]["mcnemar_c"], "2")

    def test_missing_dependencies_have_runtime_or_summary_instructions(self):
        for name, message in (("datasets", "existing GPU evaluation environment"),
                              ("scipy", "SciPy is required for ETHICS statistical summaries")):
            with patch.object(ethics, "run", side_effect=ModuleNotFoundError(name=name)):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        main(self.args("--arms", "7B-base"))
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, error.getvalue())

    def test_installed_command_checks_model_before_data_access_outside_checkout(self):
        (self.paths["7B-base"] / "config.json").unlink()
        command = Path(sys.executable).parent / "es-capacity-evaluate"
        result = subprocess.run([str(command), *self.args("--arms", "7B-base")],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("7B-base", result.stderr)
        self.assertFalse(self.out.exists())

    def test_help_and_imports_do_not_load_runtime_or_change_environment(self):
        code = """
import importlib.abc
import os
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'datasets', 'lm_eval', 'torch', 'vllm', 'transformers', 'numpy', 'scipy', 'matplotlib'}:
            raise AssertionError(f'evaluation import loaded {fullname}')
sys.meta_path.insert(0, BlockRuntime())
before = dict(os.environ)
from es_capacity.cli.evaluate import main
try:
    main(['ethics', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert dict(os.environ) == before
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--subset", result.stdout)

    def test_legacy_launcher_keeps_checkout_relative_default_and_summary_function(self):
        spec = importlib.util.spec_from_file_location("ethics_runner", ROOT / "scripts/ethics/run.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertIs(launcher.run_one_arm, ethics.run_one_arm)
        self.assertIs(launcher.summarize, ethics.summarize)
        with patch.object(launcher, "evaluate") as evaluate:
            launcher.main(["--arms", "7B-base"])
            evaluate.assert_called_once_with(["ethics", "--subset", "commonsense", "--arms", "7B-base"],
                                             default_ethics_output=ROOT / "results/ethics_results")


if __name__ == "__main__":
    unittest.main()
