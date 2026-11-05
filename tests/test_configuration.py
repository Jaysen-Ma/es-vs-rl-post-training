"""Configuration contracts, independent of GPU and model dependencies."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from es_capacity.config import ethics_data_dir, models_dir
from es_capacity.models import local_checkpoint_dir, resolve_model


ROOT = Path(__file__).resolve().parents[1]


def load_runner(relative):
    spec = importlib.util.spec_from_file_location(relative, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_known_arm_falls_back_to_published_model(self):
        self.assertEqual(
            resolve_model("7B-ES", self.root), "zocrate/Qwen2.5-7B-ES-math"
        )

    def test_mirror_requires_config_file(self):
        local = self.root / "Qwen2.5-7B-ES-math"
        local.mkdir()
        self.assertEqual(
            resolve_model("7B-ES", self.root), "zocrate/Qwen2.5-7B-ES-math"
        )
        (local / "config.json").write_text("{}")
        self.assertEqual(resolve_model("7B-ES", self.root), str(local))
        self.assertEqual(
            resolve_model("zocrate/Qwen2.5-7B-ES-math", self.root), str(local)
        )

    def test_custom_model_and_path_are_preserved_without_mirror(self):
        for value in ("example/custom-model", str(self.root / "custom checkpoint")):
            with self.subTest(value=value):
                self.assertEqual(resolve_model(value, self.root / "mirror"), value)

    def test_custom_hub_id_can_use_local_mirror(self):
        local = self.root / "custom-model"
        local.mkdir()
        (local / "config.json").write_text("{}")
        self.assertEqual(resolve_model("example/custom-model", self.root), str(local))

    def test_environment_is_read_when_resolving(self):
        local = self.root / "Qwen2.5-1.5B"
        local.mkdir()
        (local / "config.json").write_text("{}")
        with patch.dict(os.environ, {"MODELS_DIR": str(self.root)}):
            self.assertEqual(resolve_model("1.5B-base"), str(local))
        with patch.dict(os.environ, {"MODELS_DIR": str(self.root / "empty")}):
            self.assertEqual(resolve_model("1.5B-base"), "Qwen/Qwen2.5-1.5B")

    def test_explicit_root_overrides_environment(self):
        with patch.dict(os.environ, {"MODELS_DIR": str(self.root / "other")}):
            self.assertEqual(
                local_checkpoint_dir("7B-RL", self.root),
                self.root / "Qwen-2.5-7B-SimpleRL-Zoo",
            )

    def test_original_path_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(models_dir(), Path.home() / "models")
            self.assertEqual(ethics_data_dir(), Path("/tmp/ethics/ethics"))

    def test_ethics_directory_can_be_configured(self):
        with patch.dict(os.environ, {"ETHICS_DATA_DIR": str(self.root)}):
            self.assertEqual(ethics_data_dir(), self.root)


class RunnerConfigurationTests(unittest.TestCase):
    def test_evaluation_runners_use_shared_resolution_without_path_hacks(self):
        old_path = sys.path.copy()
        for name in (
            "scripts/mmlu/run.py",
            "scripts/gpqa/run_sweep.py",
            "scripts/ethics/run.py",
            "scripts/ethics/run_rest.py",
        ):
            with self.subTest(runner=name):
                runner = load_runner(name)
                self.assertIs(runner.resolve, resolve_model)
                self.assertEqual(sys.path, old_path)

    def test_ethics_data_directory_reaches_worker_command(self):
        runner = load_runner("scripts/ethics/run_rest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_data = root / "explicit data"
            cli_data.mkdir()
            env_data = root / "environment data"
            env_data.mkdir()
            out = root / "output"
            for override in ([], ["--data-dir", str(cli_data)]):
                expected = env_data if not override else cli_data
                with self.subTest(override=override):
                    argv = [
                        "run_rest.py", "--arms", "7B-base", "--out", str(out), *override
                    ]
                    with (
                        patch.dict(os.environ, {"ETHICS_DATA_DIR": str(env_data)}),
                        patch.object(sys, "argv", argv),
                        patch.object(runner, "prepare", return_value=out) as prepare,
                        patch.object(runner, "summarize"),
                        patch.object(
                            runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)
                        ) as run,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        runner.main()
                    prepare.assert_called_once_with(out, expected)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--data-dir") + 1], str(expected))

    def test_ethics_worker_uses_selected_directory(self):
        runner = load_runner("scripts/ethics/run_rest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "output"
            argv = [
                "run_rest.py", "--_arm", "7B-base", "--data-dir", str(root),
                "--out", str(out),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(runner, "prepare", return_value=out) as prepare,
                patch.object(runner, "run_one_arm") as evaluate,
            ):
                runner.main()
            prepare.assert_called_once_with(out, root)
            self.assertEqual(evaluate.call_args.args[:3], ("7B-base", out, out))

    def test_missing_ethics_directory_fails_before_worker_start(self):
        runner = load_runner("scripts/ethics/run_rest.py")
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing")
            with (
                patch.object(sys, "argv", ["run_rest.py", "--data-dir", missing]),
                patch.object(runner, "prepare") as prepare,
                patch.object(runner.subprocess, "run") as run,
            ):
                with self.assertRaisesRegex(SystemExit, "set --data-dir or ETHICS_DATA_DIR"):
                    runner.main()
            prepare.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
