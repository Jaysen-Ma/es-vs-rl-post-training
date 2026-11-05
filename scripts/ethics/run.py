#!/usr/bin/env python3
"""ETHICS commonsense morality, 0-shot, one model per subprocess.

Same scoring path as scripts/mmlu/run.py: the model never generates. Each
scenario is ranked by the log-likelihood of the continuations " no" and " yes"
(the leading space is lm_eval's target delimiter). Label 0 is "no" (not wrong),
label 1 is "yes" (wrong).

The installed lm_eval task reads EleutherAI/hendrycks_ethics, whose Hub repo is
still a datasets loading script. datasets>=4 refuses those, so this writes the
official test CSV (Hendrycks' ethics.tar) to a JSONL file and evaluates a local
task yaml that points at it. The prompt text matches ethics_cm.

Writes samples.jsonl and results.json per arm and summary.csv (accuracy,
P(yes), accuracy split by gold label, exact McNemar vs. the arm's own base)
under results/ethics_results/. The other four ETHICS subsets are
scripts/ethics/run_rest.py. Default arms are the 1.5B three; pass --arms for 7B.

Usage:
    python scripts/ethics/run.py
    python scripts/ethics/run.py --arms 7B-base 7B-ES 7B-RL
    python scripts/ethics/run.py --arms 1.5B-base --limit 8
"""

from __future__ import annotations

import os  # noqa: E402

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

from es_capacity.analysis.ethics_commonsense import load_preds, summarize  # noqa: E402

from es_capacity.cli.models import add_config_argument, validate_model_args  # noqa: E402

from es_capacity.models import MODELS, resolve_model as resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASK_NAME = "ethics_cm"
TASK_TEMPLATE = Path(__file__).resolve().parent / "ethics_cm.yaml"
ETHICS_URL = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"
DEFAULT_ARMS = ["1.5B-base", "1.5B-ES", "1.5B-RL"]


def ensure_test_jsonl(out_root: Path, csv_path: Path | None) -> Path:
    """Write the official commonsense test split as JSONL with integer labels."""
    jsonl = out_root / "cm_test.jsonl"
    if jsonl.is_file() and csv_path is None:
        return jsonl

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

    out_root.mkdir(parents=True, exist_ok=True)
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
    text = TASK_TEMPLATE.read_text().replace("PLACEHOLDER", str(jsonl))
    path = out_root / "ethics_cm.yaml"
    path.write_text(text)
    return path


def run_one_arm(arm: str, out_root: Path, task_yaml: Path, args) -> None:
    """Load one model, score every scenario, write results and samples."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ethics_results")
    ap.add_argument("--csv", type=Path, default=None,
                    help="official cm_test.csv; downloaded inside --out if omitted")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--limit", type=float, default=None,
                    help="fraction or count of scenarios; for a quick check")
    ap.add_argument("--_arm", default=None, help=argparse.SUPPRESS)
    add_config_argument(ap)
    args = ap.parse_args()
    validate_model_args(ap, args)

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
            sys.executable, "-u", __file__,
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


if __name__ == "__main__":
    main()
