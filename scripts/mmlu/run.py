#!/usr/bin/env python3
"""MMLU 0-shot across the six arms, one model per subprocess.

Companion to scripts/gpqa/run_sweep.py. GPQA-diamond turned out to have no
headroom -- both bases answer it at four-choice chance -- so MMLU is the probe
that can actually show whether math-only post-training cost anything elsewhere.

Scoring is log-likelihood over the four letters: the model never generates, it
just ranks "A", "B", "C", "D". One subprocess per arm so the GPU is fully
released between models.

Three settings are not defaults and are worth understanding before changing:

  batch_size            An explicit integer, never "auto". On lm_eval's vLLM
      log-likelihood path "auto" maps to chunk size 0, and chunk size 0 in
      lm_eval/models/utils.py:chunks() never yields, so the entire task becomes
      one batch and nothing is freed until it finishes. That is 56,168 requests
      for MMLU.

  max_num_batched_tokens  Asking for a log-probability at every prompt token
      materializes logits over the whole prefill batch, sized
      max_num_batched_tokens x vocab. Qwen2.5's vocab is 151,936, so each fp32
      copy is ~1.2 GB at 2048 and ~5.0 GB at vLLM's default 8192. Note that
      gpu_memory_utilization does not bound this -- that fraction sizes the KV
      cache, and activations allocate on top of it.

  max_model_len         The longest 0-shot MMLU prompt is 1,010 tokens across
      all 57 subjects under the Qwen2.5 tokenizer, median 88, so 1,280
      truncates nothing. A few-shot run needs this re-measured: 5-shot prompts
      exceed 2,048 on the law and history subjects, and lm_eval truncates from
      the left with a warning that verbosity="ERROR" hides.

Usage:
    python scripts/mmlu/run.py                     # all six arms
    python scripts/mmlu/run.py --arms 1.5B-base
    python scripts/mmlu/run.py --limit 0.05        # quick check, 5% per subject

Reduce the resulting tree with:
    python scripts/mmlu/tables.py
"""

from __future__ import annotations

import os  # noqa: E402

# Only log-likelihoods are scored here, so the sampler is never used. vLLM's
# FlashInfer sampler nonetheless JIT-compiles itself at engine init, which needs
# a toolchain this run has no other use for.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from es_capacity.models import MODELS, resolve_model as resolve  # noqa: E402

TASK = "mmlu"


def run_one_arm(arm: str, out_root: Path, args) -> None:
    """Load one model, score all 57 MMLU subjects, write results and samples."""
    import datasets

    datasets.disable_caching()

    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM

    path = resolve(arm)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=list(MODELS))
    ap.add_argument("--out", type=Path, default=Path("results/mmlu_results"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument("--max-model-len", type=int, default=1280)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--limit", type=float, default=None,
                    help="fraction of each subject to score; for quick checks")
    ap.add_argument("--_arm", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._arm is not None:
        run_one_arm(args._arm, args.out, args)
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
        if subprocess.run(cmd).returncode != 0:
            print(f"!! {arm} FAILED", flush=True)
            failed.append(arm)

    print(f"\nDONE. {len(args.arms) - len(failed)}/{len(args.arms)} arms ok.")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
