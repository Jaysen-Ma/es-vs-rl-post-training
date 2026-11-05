"""MMLU evaluation, with each model isolated in its own worker process.

The worker scores four answer letters by likelihood, writes the existing result
and sample files, then terminates its child processes and exits. Runtime imports
and vLLM environment settings are deferred until the worker starts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from es_capacity.models import MODELS, resolve_model as resolve

TASK = "mmlu"


def run_one_arm(arm: str, out_root: Path, args) -> None:
    """Load one model, score all 57 MMLU subjects, write results and samples."""
    # Configure vLLM only in the worker, before importing the runtime.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    path = resolve(arm, config=args.config)

    import datasets

    datasets.disable_caching()

    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM

    print(f"[{arm}] loading {path}", flush=True)
    t0 = time.time()
    lm = VLLM(
        pretrained=path,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        batch_size=args.batch_size,
        # One pass per model, so graph capture would cost more than it saves.
        enforce_eager=True,
    )
    print(f"[{arm}] loaded in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    res = evaluator.simple_evaluate(
        model=lm,
        tasks=[TASK],
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

    # Per-question records. Paired tests between arms need these, and the GPQA
    # sweep once lost them for two arms and could not recompute without a full
    # re-run. Samples are cheap; re-running is not.
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

    overall = res["results"][TASK]["acc,none"]
    print(f"[{arm}] MMLU {args.num_fewshot}-shot overall = {overall * 100:.2f}%", flush=True)

    # vLLM's engine-core child can outlive its parent while still holding the
    # GPU, which would leave nothing for the next arm. Reap it before exiting.
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
    """Run selected arms sequentially after the CLI has checked every path."""
    if args._arm is not None:
        run_one_arm(args._arm, args.out, args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in args.arms:
        print(f"\n{'#' * 60}\n# {arm}\n{'#' * 60}", flush=True)
        cmd = [
            sys.executable, "-u", "-m", "es_capacity.cli.evaluate", "mmlu",
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

    print(f"\nDONE. {len(args.arms) - len(failed)}/{len(args.arms)} arms ok.")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)
