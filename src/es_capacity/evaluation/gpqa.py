"""GPQA evaluation with one model per worker and multiple choice permutations.

Each worker loads its model once, disables dataset caching before task setup,
then evaluates all selected seeds and saves per-question records. Runtime imports
and vLLM configuration are deferred until the worker starts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from es_capacity.models import MODELS, resolve_model as resolve

TASK = "gpqa_diamond_zeroshot"


def run_one_arm(arm: str, seeds: list[int], out_root: Path, args) -> None:
    """Load one model once, evaluate it under each seed, write lm_eval-shaped output."""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    path = resolve(arm, config=args.config)

    import datasets

    # Must precede task construction so answer-order shuffles are not cached.
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
        batch_size=args.batch_size,
        enforce_eager=args.enforce_eager,
    )
    print(f"[{arm}] loaded in {time.time() - t0:.0f}s", flush=True)

    for seed in seeds:
        t1 = time.time()
        res = evaluator.simple_evaluate(
            model=lm,
            tasks=[TASK],
            # All four seeds move together so the run is described by one number.
            # random_seed is the one that reaches the choice shuffle.
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            log_samples=True,
            verbosity="ERROR",
        )

        arm_dir = out_root / f"seed{seed}" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        samples = res.pop("samples")[TASK]

        # The GPQA analyzer reads config.model_args.pretrained; passing a live LM
        # instance leaves lm_eval's own config without it, so record it here.
        res.setdefault("config", {})
        res["config"]["model_args"] = {"pretrained": path, "dtype": args.dtype}
        res["config"]["arm"] = arm
        res["config"]["seed"] = seed

        (arm_dir / f"results_{stamp}.json").write_text(
            json.dumps(res, indent=2, default=str)
        )
        with (arm_dir / f"samples_{TASK}_{stamp}.jsonl").open("w") as fh:
            for rec in samples:
                fh.write(json.dumps(rec, default=str) + "\n")

        acc = res["results"][TASK]["acc,none"]
        n = res["n-samples"][TASK]["effective"]
        print(
            f"[{arm}] seed={seed}  acc={acc * 100:5.2f}%  "
            f"({round(acc * n)}/{n})  {time.time() - t1:.0f}s",
            flush=True,
        )


def run(args) -> None:
    """Run all selected model workers, reporting failures after the sweep."""
    if args._arm is not None:
        run_one_arm(args._arm, args.seeds, args.out, args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in args.arms:
        print(f"\n{'#' * 60}\n# {arm}  ({len(args.seeds)} seeds)\n{'#' * 60}", flush=True)
        cmd = [
            sys.executable, "-m", "es_capacity.cli.evaluate", "gpqa",
            "--_arm", arm,
            "--out", str(args.out),
            "--dtype", args.dtype,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--batch-size", str(args.batch_size),
            "--seeds", *[str(s) for s in args.seeds],
        ]
        if args.config is not None:
            cmd += ["--config", str(args.config)]
        if subprocess.run(cmd).returncode != 0:
            print(f"!! {arm} FAILED", flush=True)
            failed.append(arm)

    print(f"\nSWEEP COMPLETE. {len(args.arms) - len(failed)}/{len(args.arms)} arms ok.")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)
