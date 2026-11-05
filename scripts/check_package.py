"""Build and validate the distribution in isolation (Python 3.12 and uv)."""

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


def main():
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
        run(uv, "pip", "install", "--python", python, "--no-deps", wheel, cwd=work)
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
        # Run the shipped tests and runner scripts against the installed wheel.
        # -I and an external cwd prevent imports from the development checkout.
        tests = source / "tests"
        if not list(tests.glob("test_*.py")):
            raise RuntimeError("source distribution contains no tests")
        run(python, "-I", "-m", "unittest", "discover", "-s", tests, "-v", cwd=work)
        print("Package checks passed.", flush=True)


if __name__ == "__main__":
    main()
