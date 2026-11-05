"""Checks for retaining validated release files without overwriting candidates."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "check_package", Path(__file__).resolve().parents[1] / "scripts/check_package.py",
)
check_package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_package)


class PackageArtifactTests(unittest.TestCase):
    def test_export_preserves_bytes_and_writes_sha256_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "build"
            dist.mkdir()
            (dist / "example.whl").write_bytes(b"abc")
            (dist / "example.tar.gz").write_bytes(b"")
            (dist / ".gitignore").write_text("*")
            output = root / "release files"
            check_package.export_artifacts(dist, output)
            self.assertEqual((output / "example.whl").read_bytes(), b"abc")
            self.assertEqual((output / "example.tar.gz").read_bytes(), b"")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"example.whl", "example.tar.gz", "SHA256SUMS"},
            )
            self.assertEqual(
                (output / "SHA256SUMS").read_text(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  example.tar.gz\n"
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  example.whl\n",
            )

    def test_existing_output_is_rejected_before_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            (existing / "keep").write_bytes(b"previous candidate")
            with patch.object(check_package, "run") as run:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        check_package.main(["--out-dir", str(existing)])
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
            self.assertEqual((existing / "keep").read_bytes(), b"previous candidate")

    def test_failed_build_does_not_export_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate"
            with patch.object(check_package, "run", side_effect=RuntimeError("build failed")):
                with self.assertRaisesRegex(RuntimeError, "build failed"):
                    check_package.main(["--out-dir", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
