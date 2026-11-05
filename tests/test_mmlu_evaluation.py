"""MMLU orchestration and worker contracts using a simulated model runtime."""

import contextlib
import copy
import glob
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from es_capacity.cli.evaluate import main
from es_capacity.evaluation import mmlu


ROOT = Path(__file__).resolve().parents[1]


class MmluEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "evaluation outputs"
        self.config = self.root / "model paths.toml"
        self.paths = {}
        for arm in ("7B-base", "7B-ES", "7B-RL"):
            path = self.root / f"downloaded {arm}"
            path.mkdir()
            (path / "config.json").write_text("{}")
            self.paths[arm] = path
        self.config.write_text("[models]\n" + "".join(
            f'{json.dumps(arm)} = {json.dumps(str(path))}\n' for arm, path in self.paths.items()
        ))

    def args(self, *extra):
        return ["mmlu", "--config", str(self.config), "--out", str(self.out), *extra]

    def test_parent_forwards_all_settings_and_continues_after_worker_failure(self):
        extra = ["--arms", "7B-base", "7B-ES", "7B-RL", "--dtype", "float16",
                 "--num-fewshot", "5", "--seed", "9", "--limit", "0.05",
                 "--gpu-memory-utilization", "0.4", "--max-model-len", "4096",
                 "--max-num-batched-tokens", "3072", "--batch-size", "32"]
        before = dict(os.environ)
        with patch.object(mmlu.subprocess, "run", side_effect=[
            SimpleNamespace(returncode=0), SimpleNamespace(returncode=2), SimpleNamespace(returncode=0),
        ]) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as raised:
                    main(self.args(*extra))
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(run.call_count, 3)
        self.assertIn("DONE. 2/3 arms ok.", output.getvalue())
        self.assertIn("failed: 7B-ES", output.getvalue())
        self.assertEqual(dict(os.environ), before)
        for call, arm in zip(run.call_args_list, self.paths):
            command = call.args[0]
            self.assertEqual(command[:5], [sys.executable, "-u", "-m", "es_capacity.cli.evaluate", "mmlu"])
            expected = {
                "--_arm": arm, "--out": str(self.out), "--config": str(self.config),
                "--dtype": "float16", "--num-fewshot": "5", "--seed": "9", "--limit": "0.05",
                "--gpu-memory-utilization": "0.4", "--max-model-len": "4096",
                "--max-num-batched-tokens": "3072", "--batch-size": "32",
            }
            self.assertEqual(dict(zip(command[5::2], command[6::2])), expected)

    def test_default_settings_and_environment_config_reach_worker(self):
        with patch.dict(os.environ, {"ES_CAPACITY_CONFIG": str(self.config)}):
            with patch.object(mmlu.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["mmlu", "--arms", "7B-base", "--out", str(self.out)])
        command = run.call_args.args[0]
        for flag, value in {
            "--dtype": "bfloat16", "--num-fewshot": "0", "--seed": "0",
            "--gpu-memory-utilization": "0.25", "--max-model-len": "1280",
            "--max-num-batched-tokens": "2048", "--batch-size": "1024",
            "--config": str(self.config),
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--limit", command)

    def test_all_paths_are_checked_before_output_or_workers(self):
        (self.paths["7B-ES"] / "config.json").unlink()
        with patch.object(mmlu.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(self.args("--arms", "7B-base", "7B-ES"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("7B-ES", error.getvalue())
        run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_worker_preserves_runtime_arguments_result_format_and_process_cleanup(self):
        result = {
            "results": {"mmlu": {"acc,none": 0.5}},
            "n-samples": {"subject_a": {"effective": 1}, "subject_b": {"effective": 1}},
            "group_subtasks": {"mmlu": ["subject_a", "subject_b"]},
            "samples": {"subject_a": [{"doc_id": 0, "acc": 1}],
                        "subject_b": [{"doc_id": 0, "acc": 0}]},
        }
        datasets = ModuleType("datasets")
        datasets.disable_caching = Mock()
        lm_eval = ModuleType("lm_eval")
        evaluator = SimpleNamespace(simple_evaluate=Mock(return_value=copy.deepcopy(result)))
        lm_eval.evaluator = evaluator
        runtime = ModuleType("lm_eval.models.vllm_causallms")
        model = object()
        runtime.VLLM = Mock(return_value=model)
        args = SimpleNamespace(config=self.config, dtype="bfloat16", num_fewshot=0, seed=7,
                               limit=0.05, gpu_memory_utilization=0.25, max_model_len=1280,
                               max_num_batched_tokens=2048, batch_size=1024)
        original_open = open

        def open_proc(path, *args, **kwargs):
            if str(path) == "/proc/700/stat":
                return io.StringIO("700 (engine core) S 42 0 0")
            if str(path) == "/proc/701/stat":
                return io.StringIO("701 (unrelated) S 99 0 0")
            return original_open(path, *args, **kwargs)

        modules = {"datasets": datasets, "lm_eval": lm_eval,
                   "lm_eval.models": ModuleType("lm_eval.models"),
                   "lm_eval.models.vllm_causallms": runtime}
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {}, clear=True),
            patch.object(mmlu.time, "time", side_effect=[10, 11, 12, 14]),
            patch.object(glob, "glob", return_value=["/proc/700/stat", "/proc/701/stat"]),
            patch.object(mmlu.os, "getpid", return_value=42),
            patch("builtins.open", side_effect=open_proc),
            patch.object(mmlu.os, "kill") as kill,
            patch.object(mmlu.os, "_exit") as exit_worker,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            mmlu.run_one_arm("7B-ES", self.out, args)
            kill.assert_called_once_with(700, signal.SIGKILL)
            exit_worker.assert_called_once_with(0)
            self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        datasets.disable_caching.assert_called_once_with()
        runtime.VLLM.assert_called_once_with(
            pretrained=str(self.paths["7B-ES"]), dtype="bfloat16", gpu_memory_utilization=0.25,
            max_model_len=1280, max_num_batched_tokens=2048, batch_size=1024, enforce_eager=True,
        )
        evaluator.simple_evaluate.assert_called_once_with(
            model=model, tasks=["mmlu"], num_fewshot=0, limit=0.05,
            random_seed=7, numpy_random_seed=7, torch_random_seed=7,
            fewshot_random_seed=7, log_samples=True, verbosity="ERROR",
        )
        saved = self.out / "7B-ES"
        self.assertEqual((saved / "samples.jsonl").read_text(),
                         '{"doc_id": 0, "acc": 1, "subtask": "subject_a"}\n'
                         '{"doc_id": 0, "acc": 0, "subtask": "subject_b"}\n')
        expected = {key: value for key, value in result.items() if key != "samples"}
        expected["config"] = {"model_args": {"pretrained": str(self.paths["7B-ES"]), "dtype": "bfloat16"},
                              "arm": "7B-ES", "num_fewshot": 0, "seed": 7, "wall_seconds": 2}
        self.assertEqual(json.loads((saved / "results.json").read_text()), expected)

    def test_worker_missing_runtime_produces_environment_instructions(self):
        with patch.object(mmlu, "run", side_effect=ModuleNotFoundError(name="lm_eval")):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(self.args("--_arm", "7B-base"))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("existing GPU evaluation environment", error.getvalue())
        self.assertIn("analysis extra does not install", error.getvalue())

    def test_installed_command_preflight_works_from_external_directory(self):
        (self.paths["7B-base"] / "config.json").unlink()
        command = Path(sys.executable).parent / "es-capacity-evaluate"
        result = subprocess.run([str(command), *self.args("--arms", "7B-base")],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("7B-base", result.stderr)
        self.assertNotIn("No module named", result.stderr)
        self.assertFalse(self.out.exists())

    def test_help_and_module_import_leave_environment_unchanged_without_runtime(self):
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
assert dict(os.environ) == before
try:
    main(['mmlu', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert dict(os.environ) == before
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--batch-size", result.stdout)

    def test_legacy_launcher_delegates_with_existing_output_default(self):
        spec = importlib.util.spec_from_file_location("mmlu_runner", ROOT / "scripts/mmlu/run.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertIs(launcher.run_one_arm, mmlu.run_one_arm)
        with patch.object(launcher, "evaluate") as evaluate:
            launcher.main(["--arms", "7B-base"])
            evaluate.assert_called_once_with(["mmlu", "--arms", "7B-base"],
                                             default_mmlu_output=Path("results/mmlu_results"))

    def test_missing_output_or_empty_arm_selection_is_rejected(self):
        for argv in (["mmlu", "--config", str(self.config)], self.args("--arms")):
            with patch.object(mmlu.subprocess, "run") as run:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(argv)
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
