"""GPQA worker reuse, seed forwarding and output contracts without model execution."""

import contextlib
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

from es_capacity.analysis import gpqa as analysis
from es_capacity.cli.evaluate import main
from es_capacity.evaluation import gpqa


ROOT = Path(__file__).resolve().parents[1]


class GpqaEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "evaluation outputs"
        self.config = self.root / "model paths.toml"
        self.paths = {}
        for arm in ("7B-base", "7B-ES"):
            path = self.root / f"downloaded {arm}"
            path.mkdir()
            (path / "config.json").write_text("{}")
            self.paths[arm] = path
        self.config.write_text("[models]\n" + "".join(
            f'{json.dumps(arm)} = {json.dumps(str(path))}\n' for arm, path in self.paths.items()
        ))

    def args(self, *extra):
        return ["gpqa", "--config", str(self.config), "--out", str(self.out), *extra]

    def test_parent_forwards_settings_and_seeds_and_reports_worker_failure(self):
        arguments = self.args("--arms", "7B-base", "7B-ES", "--seeds", "3", "11",
                              "--dtype", "float16", "--gpu-memory-utilization", "0.6",
                              "--max-model-len", "8192", "--batch-size", "64", "--enforce-eager")
        before = dict(os.environ)
        with patch.object(gpqa.subprocess, "run", side_effect=[
            SimpleNamespace(returncode=2), SimpleNamespace(returncode=0),
        ]) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as raised:
                    main(arguments)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(run.call_count, 2)
        self.assertIn("SWEEP COMPLETE. 1/2 arms ok.", output.getvalue())
        self.assertIn("failed: 7B-base", output.getvalue())
        self.assertEqual(dict(os.environ), before)
        for call, arm in zip(run.call_args_list, self.paths):
            command = call.args[0]
            self.assertEqual(command[:4], [sys.executable, "-m", "es_capacity.cli.evaluate", "gpqa"])
            for flag, value in {
                "--_arm": arm, "--out": str(self.out), "--dtype": "float16",
                "--gpu-memory-utilization": "0.6", "--max-model-len": "8192",
                "--batch-size": "64", "--config": str(self.config),
            }.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command[command.index("--seeds") + 1:command.index("--config")], ["3", "11"])

    def test_defaults_and_environment_config_reach_worker(self):
        with patch.dict(os.environ, {"ES_CAPACITY_CONFIG": str(self.config)}):
            with patch.object(gpqa.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["gpqa", "--arms", "7B-base", "--out", str(self.out)])
        command = run.call_args.args[0]
        for flag, value in {
            "--dtype": "bfloat16", "--gpu-memory-utilization": "0.5",
            "--max-model-len": "4096", "--batch-size": "auto", "--config": str(self.config),
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(command[command.index("--seeds") + 1:command.index("--config")],
                         [str(seed) for seed in range(10)])

    def test_worker_reuses_model_disables_caching_and_writes_analyzable_seed_outputs(self):
        events, models, calls = [], [], []
        datasets = ModuleType("datasets")
        datasets.disable_caching = Mock(side_effect=lambda: events.append("disable_caching"))

        def load_model(**kwargs):
            events.append("load_model")
            model = SimpleNamespace(settings=kwargs)
            models.append(model)
            return model

        def evaluate(**kwargs):
            events.append(f"evaluate_seed{kwargs['random_seed']}")
            calls.append(kwargs)
            seed = kwargs["random_seed"]
            is_base = kwargs["model"].settings["pretrained"] == str(self.paths["7B-base"])
            scores = [int(seed == 2), int(seed != 2)] if is_base else [1, 1]
            return {
                "results": {gpqa.TASK: {"acc,none": sum(scores) / 2, "acc_stderr,none": 0.1}},
                "n-samples": {gpqa.TASK: {"effective": 2}},
                "samples": {gpqa.TASK: [
                    {"doc_id": i, "acc": score, "doc_hash": f"seed{seed}-doc{i}",
                     "filtered_resps": [[0], [1], [0], [0]]} for i, score in enumerate(scores)
                ]},
            }

        lm_eval = ModuleType("lm_eval")
        lm_eval.evaluator = SimpleNamespace(simple_evaluate=Mock(side_effect=evaluate))
        runtime = ModuleType("lm_eval.models.vllm_causallms")
        runtime.VLLM = Mock(side_effect=load_model)
        modules = {"datasets": datasets, "lm_eval": lm_eval,
                   "lm_eval.models": ModuleType("lm_eval.models"),
                   "lm_eval.models.vllm_causallms": runtime}
        args = SimpleNamespace(config=self.config, dtype="bfloat16", gpu_memory_utilization=0.5,
                               max_model_len=4096, batch_size="auto", enforce_eager=True)
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}),
            patch.object(gpqa.time, "time", return_value=100),
            patch.object(gpqa.time, "strftime", return_value="2026-01-02T03-04-05"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for arm in self.paths:
                gpqa.run_one_arm(arm, [2, 5], self.out, args)
            self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "1")
        self.assertEqual(events, ["disable_caching", "load_model", "evaluate_seed2", "evaluate_seed5"] * 2)
        self.assertEqual(runtime.VLLM.call_count, 2)
        self.assertEqual(datasets.disable_caching.call_count, 2)
        for index, (arm, path) in enumerate(self.paths.items()):
            self.assertEqual(models[index].settings, {
                "pretrained": str(path), "dtype": "bfloat16", "gpu_memory_utilization": 0.5,
                "max_model_len": 4096, "batch_size": "auto", "enforce_eager": True,
            })
            for offset, seed in enumerate((2, 5)):
                self.assertEqual(calls[index * 2 + offset], {
                    "model": models[index], "tasks": [gpqa.TASK], "random_seed": seed,
                    "numpy_random_seed": seed, "torch_random_seed": seed, "fewshot_random_seed": seed,
                    "log_samples": True, "verbosity": "ERROR",
                })
                directory = self.out / f"seed{seed}" / arm
                results = json.loads((directory / "results_2026-01-02T03-04-05.json").read_text())
                self.assertNotIn("samples", results)
                self.assertEqual(results["config"], {
                    "model_args": {"pretrained": str(path), "dtype": "bfloat16"}, "arm": arm, "seed": seed,
                })
                sample_file = directory / f"samples_{gpqa.TASK}_2026-01-02T03-04-05.jsonl"
                self.assertEqual(len(sample_file.read_text().splitlines()), 2)
        sweep = analysis.load_sweep(self.out)
        self.assertEqual(set(sweep), {2, 5})
        self.assertEqual(analysis.check_invariants(sweep), [])
        tables = analysis.summarize(sweep, ["7B-ES:7B-base"])
        self.assertEqual(len(tables["scores.csv"]), 4)
        self.assertEqual(tables["pairs_summary.csv"][0]["delta_mean_pts"], 50.0)

    def test_missing_checkpoint_fails_before_any_worker_or_output(self):
        (self.paths["7B-ES"] / "config.json").unlink()
        with patch.object(gpqa.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(self.args("--arms", "7B-base", "7B-ES"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("7B-ES", error.getvalue())
        run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_empty_duplicate_or_negative_seeds_and_missing_output_are_rejected(self):
        cases = [self.args("--arms", "7B-base", "--seeds", *seeds)
                 for seeds in ([], ["0", "0"], ["-1"], ["1", "-2"])]
        cases.append(["gpqa", "--config", str(self.config), "--arms", "7B-base"])
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with patch.object(gpqa.subprocess, "run") as run:
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main(arguments)
                    self.assertEqual(raised.exception.code, 2)
                    run.assert_not_called()
                self.assertFalse(self.out.exists())

    def test_worker_missing_runtime_produces_environment_instructions(self):
        with patch.object(gpqa, "run", side_effect=ModuleNotFoundError(name="datasets")):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(self.args("--_arm", "7B-base"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("existing GPU evaluation environment", error.getvalue())

    def test_installed_preflight_and_module_help_work_outside_checkout(self):
        (self.paths["7B-base"] / "config.json").unlink()
        command = Path(sys.executable).parent / "es-capacity-evaluate"
        result = subprocess.run([str(command), *self.args("--arms", "7B-base")],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("7B-base", result.stderr)
        self.assertFalse(self.out.exists())
        result = subprocess.run([sys.executable, "-I", "-m", "es_capacity.cli.evaluate", "gpqa", "--help"],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--seeds", result.stdout)

    def test_imports_and_help_do_not_load_runtime_or_change_environment(self):
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
    main(['gpqa', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert dict(os.environ) == before
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--seeds", result.stdout)

    def test_legacy_launcher_delegates_with_existing_output_default(self):
        spec = importlib.util.spec_from_file_location("gpqa_runner", ROOT / "scripts/gpqa/run_sweep.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertIs(launcher.run_one_arm, gpqa.run_one_arm)
        with patch.object(launcher, "evaluate") as evaluate:
            launcher.main(["--arms", "7B-base", "--seeds", "0", "1"])
            evaluate.assert_called_once_with(["gpqa", "--arms", "7B-base", "--seeds", "0", "1"],
                                             default_gpqa_output=Path("results/gpqa_results"))


if __name__ == "__main__":
    unittest.main()
