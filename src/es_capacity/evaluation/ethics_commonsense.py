"""ETHICS commonsense evaluation with local task data and one model per worker.

Dataset preparation uses the official CSV or archive; the task template is a
package resource. Model dependencies and vLLM configuration load only in workers.
Saved-output summaries use the existing analysis implementation.
"""

from __future__ import annotations

import csv
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from es_capacity.analysis.ethics_commonsense import load_preds, summarize
from es_capacity.models import MODELS, resolve_model as resolve

TASK_NAME = "ethics_cm"
TASK_TEMPLATE = files("es_capacity.evaluation").joinpath("tasks/ethics_cm.yaml")
ETHICS_URL = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"
DEFAULT_ARMS = ["1.5B-base", "1.5B-ES", "1.5B-RL"]


def ensure_test_jsonl(out_root: Path, csv_path: Path | None) -> Path:
    """Write the official commonsense test split as JSONL with integer labels."""
    jsonl = out_root / "cm_test.jsonl"
    if jsonl.is_file() and csv_path is None:
        return jsonl

    out_root.mkdir(parents=True, exist_ok=True)
    src = csv_path
    if src is None:
        tar_path = out_root / "ethics.tar"
        if not tar_path.is_file():
            print(f"downloading {ETHICS_URL}", flush=True)
            urllib.request.urlretrieve(ETHICS_URL, tar_path)
        import tarfile

        with tarfile.open(tar_path) as tar:
            member = tar.getmember("ethics/commonsense/cm_test.csv")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError("cm_test.csv missing from ethics.tar")
            text = extracted.read().decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
    else:
        with src.open(newline="") as fh:
            rows = list(csv.DictReader(fh))

    with jsonl.open("w") as fh:
        for row in rows:
            fh.write(json.dumps({
                "label": int(row["label"]),
                "input": row["input"],
            }) + "\n")
    print(f"wrote {len(rows)} scenarios to {jsonl}", flush=True)
    return jsonl


def write_task_yaml(out_root: Path, jsonl: Path) -> Path:
    """Point the local task at this run's JSONL. The Hub task cannot load."""
    text = TASK_TEMPLATE.read_text(encoding="utf-8").replace("PLACEHOLDER", json.dumps(str(jsonl.resolve())))
    path = out_root / "ethics_cm.yaml"
    path.write_text(text)
    return path


def run_one_arm(arm: str, out_root: Path, task_yaml: Path, args) -> None:
    """Load one model, score every scenario, write results and samples."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    path = resolve(arm, config=args.config)

    import datasets

    datasets.disable_caching()

    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager

    print(f"[{arm}] loading {path}", flush=True)
    t0 = time.time()
    lm = VLLM(
        pretrained=path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        batch_size=args.batch_size,
        enforce_eager=True,
    )
    print(f"[{arm}] loaded in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    res = evaluator.simple_evaluate(
        model=lm,
        tasks=[TASK_NAME],
        task_manager=TaskManager(include_path=str(task_yaml.parent)),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        log_samples=True,
        verbosity="ERROR",
    )
    dt = time.time() - t1
    print(f"[{arm}] scored in {dt:.0f}s", flush=True)

    arm_dir = out_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    samples = res.pop("samples")
    with (arm_dir / "samples.jsonl").open("w") as fh:
        for subtask, recs in samples.items():
            for rec in recs:
                rec["subtask"] = subtask
                fh.write(json.dumps(rec, default=str) + "\n")

    res.setdefault("config", {})
    res["config"]["model_args"] = {"pretrained": path, "dtype": args.dtype}
    res["config"]["arm"] = arm
    res["config"]["num_fewshot"] = args.num_fewshot
    res["config"]["seed"] = args.seed
    res["config"]["wall_seconds"] = round(dt, 1)
    (arm_dir / "results.json").write_text(json.dumps(res, indent=2, default=str))

    overall = res["results"][TASK_NAME]["acc,none"]
    print(f"[{arm}] ethics_cm {args.num_fewshot}-shot acc = {overall * 100:.2f}%", flush=True)

    import glob
    import signal

    me = os.getpid()
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            ppid = int(open(stat).read().rsplit(") ", 1)[1].split()[1])
            if ppid == me:
                os.kill(int(stat.split("/")[2]), signal.SIGKILL)
        except Exception:
            pass
    sys.stdout.flush()
    os._exit(0)


def run(args) -> None:
    """Prepare local task inputs, run the workers and summarize saved samples."""
    jsonl = ensure_test_jsonl(args.out, args.csv)
    task_yaml = write_task_yaml(args.out, jsonl)

    if args._arm is not None:
        run_one_arm(args._arm, args.out, task_yaml, args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in args.arms:
        print(f"\n{'#' * 60}\n# {arm}\n{'#' * 60}", flush=True)
        cmd = [
            sys.executable, "-u", "-m", "es_capacity.cli.evaluate", "ethics",
            "--subset", "commonsense",
            "--_arm", arm,
            "--out", str(args.out),
            "--dtype", args.dtype,
            "--num-fewshot", str(args.num_fewshot),
            "--seed", str(args.seed),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--batch-size", str(args.batch_size),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.config is not None:
            cmd += ["--config", str(args.config)]
        if subprocess.run(cmd).returncode != 0:
            print(f"!! {arm} FAILED", flush=True)
            failed.append(arm)

    summarize(args.out, args.arms)
    print(f"\nDONE. {len(args.arms) - len(failed)}/{len(args.arms)} arms ok.")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)
