"""Configuration contracts, independent of GPU and model dependencies."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from es_capacity.config import ethics_data_dir, models_dir
from es_capacity.cli.models import main as models_main
from es_capacity.models import MODELS, ModelConfigurationError, local_checkpoint_dir, resolve_model


ROOT = Path(__file__).resolve().parents[1]


def load_runner(relative):
    spec = importlib.util.spec_from_file_location(relative, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return path


def model_config(path, paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[models]\n" + "".join(
        f"{json.dumps(arm)} = {json.dumps(str(local))}\n" for arm, local in paths.items()
    ))
    return path


class ModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_missing_model_fails_with_download_link(self):
        with self.assertRaises(FileNotFoundError) as error:
            resolve_model("7B-ES", self.root)
        self.assertIn("https://huggingface.co/zocrate/Qwen2.5-7B-ES-math", str(error.exception))
        self.assertIn("--config", str(error.exception))

    def test_mirror_requires_config_file(self):
        local = self.root / "Qwen2.5-7B-ES-math"
        local.mkdir()
        with self.assertRaises(FileNotFoundError):
            resolve_model("7B-ES", self.root)
        (local / "config.json").write_text("{}")
        self.assertEqual(resolve_model("7B-ES", self.root), str(local))
        self.assertEqual(
            resolve_model("zocrate/Qwen2.5-7B-ES-math", self.root), str(local)
        )

    def test_missing_custom_model_or_path_cannot_trigger_hub_fallback(self):
        for value in ("example/custom-model", str(self.root / "custom checkpoint")):
            with self.subTest(value=value):
                with self.assertRaises(FileNotFoundError):
                    resolve_model(value, self.root / "mirror")

    def test_explicit_local_directory_is_used(self):
        local = checkpoint(self.root / "custom checkpoint")
        self.assertEqual(resolve_model(str(local)), str(local))

    def test_missing_explicit_path_cannot_select_a_same_named_mirror(self):
        checkpoint(self.root / "mirror/checkpoint")
        missing = self.root / "missing/checkpoint"
        with self.assertRaises(FileNotFoundError) as error:
            resolve_model(str(missing), self.root / "mirror")
        self.assertIn(str(missing), str(error.exception))

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
            with self.assertRaises(FileNotFoundError):
                resolve_model("1.5B-base")

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

    def test_config_maps_all_six_arms_to_arbitrary_directories(self):
        paths = {arm: checkpoint(self.root / f"download {i}" / "snapshot") for i, arm in enumerate(MODELS)}
        config = model_config(self.root / "configuration/models.toml", paths)
        for arm, local in paths.items():
            with self.subTest(arm=arm):
                self.assertEqual(resolve_model(arm, config=config), str(local))

    def test_relative_paths_are_anchored_to_config_and_expand_home(self):
        relative = checkpoint(self.root / "saved checkpoints/model")
        config = model_config(self.root / "settings/models.toml", {
            "7B-base": "../saved checkpoints/model", "7B-ES": "~/model",
        })
        self.assertEqual(resolve_model("7B-base", config=config), str(relative))
        self.assertEqual(local_checkpoint_dir("7B-ES", config=config), Path.home() / "model")

    def test_explicit_config_overrides_environment_and_mirror(self):
        first = checkpoint(self.root / "first")
        second = checkpoint(self.root / "second")
        env_config = model_config(self.root / "env.toml", {"7B-base": first})
        cli_config = model_config(self.root / "cli.toml", {"7B-base": second})
        with patch.dict(os.environ, {"ES_CAPACITY_CONFIG": str(env_config), "MODELS_DIR": str(self.root / "unused")}):
            self.assertEqual(resolve_model("7B-base"), str(first))
            self.assertEqual(resolve_model("7B-base", config=cli_config), str(second))

    def test_partial_config_requires_selected_arm_instead_of_using_mirror(self):
        local = checkpoint(self.root / "one model")
        checkpoint(self.root / "Qwen2.5-7B-ES-math")
        config = model_config(self.root / "models.toml", {"7B-base": local})
        self.assertEqual(resolve_model("7B-base", config=config), str(local))
        with self.assertRaisesRegex(ModelConfigurationError, "no local path configured for 7B-ES"):
            resolve_model("7B-ES", self.root, config=config)

    def test_invalid_or_missing_config_has_actionable_error(self):
        path = self.root / "models.toml"
        with self.assertRaisesRegex(ModelConfigurationError, "cannot read model configuration"):
            resolve_model("7B-base", config=path)
        for contents, message in (
            ('[models\n', "cannot read"),
            ('[model]\n', "expected a \\[models\\]"),
            ('[models]\n"7B-bsae" = "model"\n', "unknown model arms"),
            ('[models]\n"7B-base" = 42\n', "nonempty local path string"),
            ('[models]\n"7B-base" = ""\n', "nonempty local path string"),
        ):
            with self.subTest(contents=contents):
                path.write_text(contents)
                with self.assertRaisesRegex(ModelConfigurationError, message):
                    resolve_model("7B-base", config=path)

    def test_model_commands_check_selected_arms_and_print_just_one_path(self):
        local = checkpoint(self.root / "local model")
        config = model_config(self.root / "models.toml", {"7B-base": local, "7B-ES": "missing"})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            models_main(["path", "7B-base", "--config", str(config)])
        self.assertEqual(output.getvalue(), f"{local}\n")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            models_main(["check", "--config", str(config), "--arms", "7B-base"])
        self.assertIn(f"OK 7B-base: {local}", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                models_main(["check", "--config", str(config), "--arms", "7B-base", "7B-ES"])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("MISSING", output.getvalue())
        self.assertIn("https://huggingface.co/zocrate/Qwen2.5-7B-ES-math", output.getvalue())


class RunnerConfigurationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        paths = {arm: checkpoint(self.root / arm) for arm in MODELS}
        self.config = model_config(self.root / "models.toml", paths)
        environment = patch.dict(os.environ, {"ES_CAPACITY_CONFIG": str(self.config)})
        environment.start()
        self.addCleanup(environment.stop)

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
        implementation = sys.modules[runner.run_one_arm.__module__]
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
                        patch.object(implementation, "prepare", return_value=out) as prepare,
                        patch.object(implementation, "summarize"),
                        patch.object(
                            runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)
                        ) as run,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        runner.main()
                    prepare.assert_called_once_with(out, expected)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--data-dir") + 1], str(expected))

    def test_explicit_model_config_reaches_all_worker_commands(self):
        for name in ("scripts/mmlu/run.py", "scripts/gpqa/run_sweep.py", "scripts/ethics/run.py", "scripts/ethics/run_rest.py"):
            with self.subTest(runner=name):
                runner = load_runner(name)
                implementation = sys.modules.get(runner.run_one_arm.__module__, runner)
                argv = [name, "--arms", "7B-base", "--config", str(self.config), "--out", str(self.root / "out")]
                if name.endswith("run_rest.py"):
                    argv += ["--data-dir", str(self.root)]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", argv))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    run = stack.enter_context(patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)))
                    for helper in ("ensure_test_jsonl", "write_task_yaml", "prepare", "summarize"):
                        if hasattr(implementation, helper):
                            stack.enter_context(patch.object(implementation, helper, return_value=self.root))
                    runner.main()
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--config") + 1], str(self.config))

    def test_invalid_model_fails_before_dataset_setup_or_workers(self):
        config = model_config(self.root / "missing.toml", {"7B-base": "not downloaded"})
        for name in ("scripts/mmlu/run.py", "scripts/gpqa/run_sweep.py", "scripts/ethics/run.py", "scripts/ethics/run_rest.py"):
            with self.subTest(runner=name):
                runner = load_runner(name)
                implementation = sys.modules.get(runner.run_one_arm.__module__, runner)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", [name, "--arms", "7B-base", "--config", str(config)]))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    run = stack.enter_context(patch.object(runner.subprocess, "run"))
                    helpers = [stack.enter_context(patch.object(implementation, helper)) for helper in ("ensure_test_jsonl", "prepare") if hasattr(implementation, helper)]
                    with self.assertRaises(SystemExit) as raised:
                        runner.main()
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()
                for helper in helpers:
                    helper.assert_not_called()

    def test_maths_wrapper_passes_configured_path_with_spaces_to_harness(self):
        script = self.root / "project/scripts/passk/run_eval.sh"
        script.parent.mkdir(parents=True)
        script.write_text((ROOT / "scripts/passk/run_eval.sh").read_text())
        harness = self.root / "harness"
        harness.mkdir()
        executable = harness / "run_sharded_eval.sh"
        executable.write_text('#!/bin/bash\nprintf "%s\\n" "$1" >> "$MATH_EVAL_DIR/seen-paths"\n')
        executable.chmod(0o755)
        local = checkpoint(self.root / "downloaded model with spaces")
        config = model_config(self.root / "external models.toml", {"7B-base": local})
        env = dict(os.environ, ES_CAPACITY_CONFIG=str(config), MATH_EVAL_DIR=str(harness))
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        for arguments in ([], ["--config", str(config)]):
            env["ES_CAPACITY_CONFIG"] = str(config if not arguments else self.config)
            result = subprocess.run(["bash", str(script), *arguments, "7B-base", "base", "example", "2"], cwd=self.root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((harness / "seen-paths").read_text().splitlines(), [str(local)] * 8)

    def test_weight_comparison_uses_the_same_configured_paths(self):
        # Only exercise CLI wiring; no tensor operations or checkpoint reads.
        with patch.dict(sys.modules, {"torch": SimpleNamespace(), "safetensors": SimpleNamespace(safe_open=None)}):
            runner = load_runner("scripts/weights/compare.py")
        paths = {arm: checkpoint(self.root / f"arbitrary {arm}") for arm in ("7B-base", "7B-ES", "7B-RL")}
        config = model_config(self.root / "weight-models.toml", paths)
        with (
            patch.object(sys, "argv", ["compare.py", "--scale", "7B", "--config", str(config)]),
            patch.object(sys.modules[runner.compare.__module__], "compare") as compare,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runner.main()
        self.assertEqual(compare.call_count, 2)
        self.assertEqual([call.args[:2] for call in compare.call_args_list], [
            (paths["7B-base"], paths["7B-ES"]), (paths["7B-base"], paths["7B-RL"]),
        ])

    def test_ethics_worker_uses_selected_directory(self):
        runner = load_runner("scripts/ethics/run_rest.py")
        implementation = sys.modules[runner.run_one_arm.__module__]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "output"
            argv = [
                "run_rest.py", "--_arm", "7B-base", "--data-dir", str(root),
                "--out", str(out),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(implementation, "prepare", return_value=out) as prepare,
                patch.object(implementation, "run_one_arm") as evaluate,
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
                patch.object(sys.modules[runner.run_one_arm.__module__], "prepare") as prepare,
                patch.object(runner.subprocess, "run") as run,
                contextlib.redirect_stderr(io.StringIO()) as error,
            ):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("set --data-dir or ETHICS_DATA_DIR", error.getvalue())
            prepare.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
