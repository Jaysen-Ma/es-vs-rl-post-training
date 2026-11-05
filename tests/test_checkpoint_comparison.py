"""Checkpoint comparison CLI boundaries and optional tiny CPU tensor checks."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from es_capacity.analysis import weight_comparison as comparison
from es_capacity.cli.compare_weights import main


HAS_RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


def tiny_checkpoints(root):
    import torch
    from safetensors.torch import save_file
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3) - 4,
        "model.embed_tokens.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1,
        "model.norm.weight": torch.ones(2),
        "model.layers.0.self_attn.q_proj.bias": torch.tensor([0.1, -0.1]),
    }
    trained = {name: value + (torch.arange(value.numel()).reshape(value.shape) + 1) * 0.01
               for name, value in base.items()}
    trained["lm_head.weight"] = base["model.embed_tokens.weight"] + 0.3
    left, right = root / "base", root / "trained"
    left.mkdir()
    right.mkdir()
    names = list(base)
    weight_map = {}
    for index, group in enumerate((names[:2], names[2:])):
        shard = f"shard-{index}.safetensors"
        save_file({name: base[name] for name in group}, str(left / shard))
        weight_map.update({name: shard for name in group})
    (left / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    save_file(trained, str(right / "model.safetensors"))
    for path in (left, right):
        (path / "config.json").write_text("{}")
    return left, right, base, trained


class CheckpointCliTests(unittest.TestCase):
    def test_configured_paths_and_explicit_output_reach_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("base", "es", "rl"):
                path = root / name
                path.mkdir()
                (path / "config.json").write_text("{}")
            config = root / "models.toml"
            config.write_text('[models]\n"7B-base"="base"\n"7B-ES"="es"\n"7B-RL"="rl"\n')
            with patch.object(comparison, "compare") as compare, contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(config), "--scale", "7B", "--out", str(root / "out")])
            self.assertEqual([call.args for call in compare.call_args_list], [
                (root / "base", root / "es", root / "out/7B-ES", "7B ES"),
                (root / "base", root / "rl", root / "out/7B-RL", "7B RL"),
            ])

    def test_missing_model_stops_before_comparison_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "models.toml"
            config.write_text('[models]\n"7B-base"="missing"\n')
            with patch.object(comparison, "compare") as compare, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--config", str(config), "--scale", "7B", "--out", str(root / "out")])
            self.assertEqual(raised.exception.code, 2)
            compare.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_help_does_not_import_tensor_or_model_runtime(self):
        code = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'safetensors', 'vllm', 'lm_eval', 'datasets'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockRuntime())
from es_capacity.cli.compare_weights import main
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)


@unittest.skipUnless(HAS_RUNTIME, "optional CPU checkpoint checks require PyTorch and SafeTensors")
class CheckpointNumericalTests(unittest.TestCase):
    def test_streamed_comparison_counts_grids_and_relative_l2(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right, base, trained = tiny_checkpoints(root)
            self.assertTrue(all(not value.is_cuda for value in base.values()))
            with patch.object(comparison, "CHUNK_ELEMS", 4), contextlib.redirect_stdout(io.StringIO()):
                result = comparison.compare(left, right, root / "out", "synthetic")
            matrix_names = [name for name in base if comparison.is_matrix(name)]
            delta_sq = sum(((trained[name] - base[name]).double() ** 2).sum().item() for name in matrix_names)
            base_sq = sum((base[name].double() ** 2).sum().item() for name in matrix_names)
            self.assertEqual(result["n"], 20)
            self.assertAlmostEqual(result["rel_l2"], (delta_sq / base_sq) ** 0.5)
            with (root / "out/stats.csv").open() as fh:
                stats = {(row["series"], row["grouping"], row["group"]): row for row in csv.DictReader(fh)}
            self.assertEqual(int(stats[("delta", "pooled", "all")]["n"]), 20)
            self.assertIn(("delta", "tied_head", "lm_head"), stats)
            with (root / "out/hist.csv").open() as fh:
                hist = list(csv.DictReader(fh))
            pooled = [row for row in hist if row["series"] == "delta" and row["grouping"] == "pooled"]
            self.assertEqual(sum(int(row["count"]) for row in pooled), 20)
            edges = {series: [(row["left"], row["right"]) for row in hist
                              if row["series"] == series and row["grouping"] == "pooled"
                              and row["bin"] in ("underflow", "overflow")]
                     for series in ("base", "trained")}
            self.assertEqual(edges["base"], edges["trained"])
            self.assertEqual({path.name for path in (root / "out").iterdir()},
                             {"stats.csv", "hist.csv", "per_tensor.csv"})

    def test_chunked_reader_preserves_tensor_values(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            left, _, base, _ = tiny_checkpoints(Path(tmp))
            with patch.object(comparison, "CHUNK_ELEMS", 4):
                checkpoint = comparison.Checkpoint(left)
                for name, expected in base.items():
                    actual = torch.cat(list(checkpoint.chunks(name)))
                    torch.testing.assert_close(actual, expected.reshape(-1))


if __name__ == "__main__":
    unittest.main()
