"""Build and validate the distribution in isolation (Python 3.10+ and uv)."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT, quiet=False):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(
        list(map(str, args)), cwd=cwd, check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def export_artifacts(dist, output):
    """Retain the validated files without replacing an earlier candidate."""
    wheel, = dist.glob("*.whl")
    sdist, = dist.glob("*.tar.gz")
    output.mkdir(parents=True)
    checksums = []
    for artifact in sorted((wheel, sdist)):
        shutil.copy2(artifact, output / artifact.name)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {artifact.name}\n")
    (output / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        help="retain validated wheel, source archive and SHA256SUMS in a new directory",
    )
    args = parser.parse_args(argv)
    if args.out_dir is not None:
        args.out_dir = args.out_dir.absolute()
        if args.out_dir.exists() or args.out_dir.is_symlink():
            parser.error(f"output directory already exists: {args.out_dir}")

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv must be installed and available on PATH")

    with tempfile.TemporaryDirectory(prefix="es-capacity-package-") as tmp:
        work = Path(tmp)
        # uv builds the wheel from the sdist by default, checking both artifacts.
        run(uv, "build", "--out-dir", work / "dist")
        wheel, = (work / "dist").glob("*.whl")
        sdist, = (work / "dist").glob("*.tar.gz")
        with tarfile.open(sdist) as archive:
            archive.extractall(work / "source", filter="data")
        source, = (work / "source").iterdir()

        constraints = work / "constraints.txt"
        run(
            uv, "export", "--locked", "--group", "analysis", "--no-emit-project",
            "--output-file", constraints, quiet=True,
        )
        venv = work / "venv"
        run(uv, "venv", "--python", sys.executable, venv, cwd=work)
        bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
        python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
        cli = bin_dir / ("es-capacity-ethics.exe" if sys.platform == "win32" else "es-capacity-ethics")

        # The core package and command help must work without optional libraries.
        run(
            uv, "pip", "install", "--python", python,
            "--constraint", constraints, wheel, cwd=work,
        )
        run(uv, "pip", "check", "--python", python, cwd=work)
        run(python, "-I", "-c", """
from pathlib import Path
import importlib.util
import sys
import es_capacity
from es_capacity.analysis import ethics_cli
assert Path(es_capacity.__file__).is_relative_to(Path(sys.prefix))
assert importlib.util.find_spec('scipy') is None
""", cwd=work)
        run(cli, "--help", cwd=work)
        models_cli = bin_dir / ("es-capacity-models.exe" if sys.platform == "win32" else "es-capacity-models")
        run(models_cli, "--help", cwd=work)
        passk_cli = bin_dir / ("es-capacity-passk.exe" if sys.platform == "win32" else "es-capacity-passk")
        run(passk_cli, "--help", cwd=work)
        mmlu_cli = bin_dir / ("es-capacity-mmlu.exe" if sys.platform == "win32" else "es-capacity-mmlu")
        run(mmlu_cli, "--help", cwd=work)
        gpqa_cli = bin_dir / ("es-capacity-gpqa.exe" if sys.platform == "win32" else "es-capacity-gpqa")
        run(gpqa_cli, "--help", cwd=work)
        checkpoint = work / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        config = work / "models.toml"
        config.write_text('[models]\n"7B-base" = "checkpoint"\n')
        run(models_cli, "check", "--config", config, "--arms", "7B-base", cwd=work)

        # Exercise the wheel's extra while retaining the lockfile's versions.
        run(
            uv, "pip", "install", "--python", python,
            "--constraint", constraints, f"{wheel}[analysis]", cwd=work,
        )
        run(uv, "pip", "check", "--python", python, cwd=work)
        run(python, "-I", "-c", """
import importlib.util
import scipy.stats
for name in ('torch', 'vllm', 'lm_eval', 'datasets', 'transformers'):
    assert importlib.util.find_spec(name) is None, name
""", cwd=work)
        example_out = work / "passk-example"
        run(passk_cli, "--example", "--out", example_out, "--plot", cwd=work)
        for filename in ("synthetic_summary.json", "synthetic_passk.png"):
            if not (example_out / filename).is_file():
                raise RuntimeError(f"installed pass@k example did not produce {filename}")
        # Invoke the installed MMLU command from outside the checkout as well.
        mmlu_inputs = work / "mmlu-inputs"
        for arm, correct in (("7B-base", True), ("7B-ES", False)):
            directory = mmlu_inputs / arm
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(
                json.dumps({"results": {"mmlu": {"acc,none": int(correct)}},
                            "n-samples": {"mmlu": {"effective": 1}}})
            )
            (directory / "samples.jsonl").write_text(
                json.dumps({"subtask": "mmlu", "doc_id": 0, "acc": int(correct)}) + "\n"
            )
        mmlu_out = work / "mmlu-output"
        run(mmlu_cli, "--results-dir", mmlu_inputs, "--out", mmlu_out, cwd=work)
        if not (mmlu_out / "scores.csv").is_file():
            raise RuntimeError("installed MMLU command did not produce scores.csv")
        # Run the shipped tests and runner scripts against the installed wheel.
        # -I and an external cwd prevent imports from the development checkout.
        tests = source / "tests"
        if not list(tests.glob("test_*.py")):
            raise RuntimeError("source distribution contains no tests")
        run(python, "-I", "-m", "unittest", "discover", "-s", tests, "-v", cwd=work)
        if args.out_dir is not None:
            export_artifacts(work / "dist", args.out_dir)
            print(f"Validated artifacts: {args.out_dir}", flush=True)
        print("Package checks passed.", flush=True)


if __name__ == "__main__":
    main()
