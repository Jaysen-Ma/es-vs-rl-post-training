#!/usr/bin/env python3
"""GPQA-diamond zero-shot across the published arms x N answer-order permutations.

This is the out-of-domain probe, NOT pass@k: lm_eval scores the 4 choices by
log-likelihood, one pass, no sampling. That determinism is the whole reason this
script exists in its current shape — re-running the same command gives a
byte-identical number, so repetition alone measures nothing. What varies here is
the *answer-option permutation*: lm_eval's GPQA task shuffles the four choices
with a bare `random.shuffle` on the global RNG, so `random_seed` decides which
slot the correct answer lands in. Sweeping the seed turns a single draw into a
distribution, which is the only way to tell a real out-of-domain gain from a
near-chance model that happens to favour one option position.

TWO THINGS THIS SCRIPT EXISTS TO GET RIGHT:

1. `datasets.disable_caching()` before any task is built. `process_docs` runs the
   shuffle inside `dataset.map()`, and the map fingerprint hashes the function
   and the dataset but never the RNG state. With caching on, every seed after the
   first silently reuses the first permutation: 10 identical results, no error,
   nothing to see. Verify with `doc_hash` afterwards (analyze_gpqa.py --check).

2. `log_samples=True` for every arm. The previous sweep lost per-question data
   for the two RL arms, which made their McNemar permanently unrecomputable once
   the box was torn down. Samples are cheap; a re-run is not.

One subprocess per arm so the GPU is fully released between models; the seeds
loop *inside* that process, reusing one loaded vLLM engine (198 questions x 4
choices is seconds of work, so startup dominates and reloading per seed would
multiply the cost of the sweep by ~10 for nothing).

Usage:
    python scripts/gpqa/run_sweep.py                       # all 6 arms, seeds 0-9
    python scripts/gpqa/run_sweep.py --arms 1.5B-base 1.5B-ES
    python scripts/gpqa/run_sweep.py --seeds 0 1 2 --out results/gpqa_results

Reduce the resulting tree with:
    python scripts/gpqa/analyze.py --sweep-root results/gpqa_results \
        --out results/gpqa --pair 1.5B-ES:1.5B-base ...
"""

from __future__ import annotations

import os  # noqa: E402

# We only ever score log-likelihoods here — nothing samples. vLLM's FlashInfer
# sampler would nonetheless JIT-compile itself at engine init, which needs
# `ninja` (absent on this box) and dies with an unhelpful "Engine core
# initialization failed". Turn it off before vllm is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from es_capacity.cli.models import add_config_argument, validate_model_args  # noqa: E402

from es_capacity.models import MODELS, resolve_model as resolve  # noqa: E402

TASK = "gpqa_diamond_zeroshot"


def run_one_arm(arm: str, seeds: list[int], out_root: Path, args) -> None:
    """Load one model once, evaluate it under each seed, write lm_eval-shaped output."""
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

        # analyze_gpqa.py reads config.model_args.pretrained; passing a live LM
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=list(MODELS), help="model arms or local checkpoint directories")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--out", type=Path, default=Path("results/gpqa_results"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    # Internal: set when this process is the per-arm worker.
    ap.add_argument("--_arm", default=None, help=argparse.SUPPRESS)
    add_config_argument(ap)
    args = ap.parse_args()
    validate_model_args(ap, args)

    if args._arm is not None:
        run_one_arm(args._arm, args.seeds, args.out, args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in args.arms:
        print(f"\n{'#' * 60}\n# {arm}  ({len(args.seeds)} seeds)\n{'#' * 60}", flush=True)
        cmd = [
            sys.executable, __file__,
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


if __name__ == "__main__":
    main()
