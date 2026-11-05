"""
Given a run_tag and a set of arm labels already evaluated by
scripts/passk/run_eval.sh (which writes to results/<run_tag>/<label>/<benchmark>/),
compute for all 4 benchmarks in one invocation:
  1. pass@k curves at k = powers of 2 up to n_sampling (1, 2, 4, ..., n) for
     every arm — same k_values convention and same unbiased estimator as
     math_eval's own evaluate.py, and the same sparse-power-of-2 sampling the
     limit-of-RLVR paper's own figures use.
  2. The four-way solvable/unsolvable breakdown (Table 2 style) for every
     non-baseline arm vs. the baseline: for each question, "solvable" = at
     least one of the n_sampling completions was correct.

Each math_eval.py shard writes `{label_dir}/{benchmark}/{prefix}_s{start}_e{end}.jsonl`
with a `score` field (list[bool], one per sample) already attached by
evaluate(). This script globs all shard files for a run, dedupes by `idx`
(shards cover disjoint question ranges, so this just reassembles the full set).

Usage:
    python analyze_passk.py --run-tag 7b-sigma001-iter50 \
        --label base --label trained --label rl --baseline base --plot

Writes, per benchmark, results/<run_tag>/<benchmark>_summary.json and (with
--plot) results/<run_tag>/<benchmark>_passk.png. Two labels is a base-vs-one
comparison; three or more adds a four-way breakdown of every extra label vs.
the baseline. `--baseline` picks which `--label` is the reference (defaults
to the first `--label` given). Every non-baseline arm's k-grid must equal the
baseline's, or be a strict prefix of it (e.g. a cheap low-k check run before
committing to the full sample budget) — its curve then just stops early.
All arms must cover the same question IDs. Four-way breakdowns require equal
sample budgets; pairs with unequal budgets are omitted from that output. A
benchmark missing for any requested label is skipped with a warning rather
than failing the whole run.
"""
import argparse
import glob
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARKS = [
    ("aime24", "AIME24"),
    ("math500", "MATH500"),
    ("minerva_math", "Minerva"),
    ("olympiadbench", "OlympiadBench"),
]

# label -> (display name, color), for the plot legend. Unknown labels fall
# back to the raw label name and the next color in DEFAULT_COLORS.
DISPLAY = {
    "base": ("Base", "#1b7a72"),          # teal
    "trained": ("ES-trained", "#e8776a"),  # coral
    "rl": ("RL (SimpleRL-Zoo)", "#6a5acd"),  # slate purple
}
DEFAULT_COLORS = ["#c9a227", "#3d7fbf", "#8a8a8a", "#c95f9c"]


def load_scores(result_dir: str) -> dict:
    """Returns {idx: score_list} across all shard jsonl files in result_dir."""
    files = [f for f in glob.glob(os.path.join(result_dir, "*.jsonl"))]
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {result_dir}")
    by_idx = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                by_idx[sample["idx"]] = sample["score"]
    return by_idx


def power_of_2_ks(n: int) -> list:
    """1, 2, 4, ..., up to n (same convention as math_eval's evaluate.py and
    the limit-of-RLVR paper's own pass@k figures)."""
    ks = [1]
    power = 1
    while 2 ** power <= n:
        ks.append(2 ** power)
        power += 1
    return ks


def pass_at_k_curve(scores_by_idx: dict) -> tuple:
    """pass@k at k = powers of 2 up to N, unbiased estimator, averaged over
    questions. Returns (ks, values)."""
    lens = {len(v) for v in scores_by_idx.values()}
    if len(lens) != 1:
        raise ValueError(f"Inconsistent n_sampling across questions: {lens}")
    n = lens.pop()
    if n < 1:
        raise ValueError("Scores must contain at least one sample per question")

    def estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    num_correct = np.array([sum(v) for v in scores_by_idx.values()])
    ks = power_of_2_ks(n)
    values = [float(np.mean([estimator(n, c, k) for c in num_correct])) for k in ks]
    return ks, values


def four_way_breakdown(base: dict, trained: dict) -> dict:
    if not base or set(base) != set(trained):
        raise ValueError("Four-way breakdown requires identical, nonempty question sets")
    sample_counts = {len(v) for scores in (base, trained) for v in scores.values()}
    if len(sample_counts) != 1 or 0 in sample_counts:
        raise ValueError("Four-way breakdown requires equal, positive sample budgets")
    common_idx = sorted(base)

    counts = {"base_solved_trained_solved": 0, "base_solved_trained_failed": 0,
              "base_failed_trained_solved": 0, "base_failed_trained_failed": 0}
    for idx in common_idx:
        b = any(base[idx])
        t = any(trained[idx])
        if b and t:
            counts["base_solved_trained_solved"] += 1
        elif b and not t:
            counts["base_solved_trained_failed"] += 1
        elif not b and t:
            counts["base_failed_trained_solved"] += 1
        else:
            counts["base_failed_trained_failed"] += 1

    n = len(common_idx)
    fractions = {k: round(100 * v / n, 1) for k, v in counts.items()}
    return {"n_questions": n, "counts": counts, "fractions_pct": fractions}


def plot_passk(curves: dict, baseline: str, title: str, out_path: str):
    """Style matches the limit-of-RLVR paper's own pass@k figures: sparse
    power-of-2 k, log-x with plain-integer tick labels at exactly those k
    values, triangle markers, 0-1 y-axis labeled 'Coverage (pass@k)'. Each
    arm is plotted over its own k-range, so a shorter (prefix) curve simply
    stops early while longer ones continue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6 if len(curves) > 2 else 4.2, 3.8 if len(curves) > 2 else 3.6))

    color_pool = iter(DEFAULT_COLORS)
    all_ks = set()
    all_vals = []
    labels = [baseline] + [l for l in curves if l != baseline]
    for label in labels:
        ks, curve = curves[label]
        name, color = DISPLAY.get(label, (label, None))
        if color is None:
            color = next(color_pool, "#333333")
        ax.plot(ks, curve, label=name, marker="^", markersize=5, linewidth=1.8, color=color)
        all_ks.update(ks)
        all_vals.extend(curve)

    all_ks = sorted(all_ks)
    ax.set_xscale("log")
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])
    ax.minorticks_off()

    ymax = max(all_vals)
    ax.set_ylim(0, min(1.0, max(0.2, np.ceil(ymax * 5) / 5)))
    ax.set_xlabel("Number of Samples $k$")
    ax.set_ylabel("Coverage (pass@$k$)")
    ax.set_title(title)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8 if len(curves) > 2 else 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_benchmark(label_dirs: dict, baseline: str, title: str, out_prefix: str, plot: bool):
    scores = {label: load_scores(path) for label, path in label_dirs.items()}
    baseline_ids = set(scores[baseline])
    for label, arm_scores in scores.items():
        ids = set(arm_scores)
        if ids != baseline_ids:
            raise ValueError(
                f"{label!r} question IDs differ from baseline {baseline!r}: "
                f"{len(baseline_ids - ids)} missing, {len(ids - baseline_ids)} extra. "
                "Complete the missing shards before comparing arms."
            )
    curves = {label: pass_at_k_curve(scores[label]) for label in label_dirs}

    baseline_ks = curves[baseline][0]
    for label, (ks, _) in curves.items():
        if label == baseline or ks == baseline_ks:
            continue
        prefix_len = min(len(ks), len(baseline_ks))
        assert ks == baseline_ks[:prefix_len], (
            f"{label!r}'s k grid isn't a prefix of baseline {baseline!r}'s: "
            f"{baseline}={baseline_ks} {label}={ks}. {label!r} must use the same "
            "n_sampling as the baseline, or a smaller uniform override."
        )
        print(f"Note: {label} n_sampling ({ks[-1]}) is smaller than {baseline} ({baseline_ks[-1]}) — "
              f"its pass@k curve only covers k up to {ks[-1]}.")

    n_sampling = {label: len(next(iter(s.values()))) for label, s in scores.items()}
    breakdowns = {}
    for label in label_dirs:
        if label == baseline:
            continue
        if n_sampling[label] != n_sampling[baseline]:
            print(f"Skipping {label} vs {baseline} four-way breakdown: unequal "
                  f"sample budgets ({n_sampling[label]} vs {n_sampling[baseline]}).")
            continue
        breakdowns[label] = four_way_breakdown(scores[baseline], scores[label])

    out = {
        "baseline": baseline,
        "n_sampling": n_sampling,
        "ks": {label: curves[label][0] for label in label_dirs},
        "pass_at_k": {label: curves[label][1] for label in label_dirs},
        "four_way_breakdown": breakdowns,
    }

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_prefix}_summary.json")

    for label, breakdown in breakdowns.items():
        print(f"  {label} vs {baseline} (solvable within {n_sampling[label]} samples):")
        for k, v in breakdown["fractions_pct"].items():
            print(f"    {k}: {v}%")

    if plot:
        plot_passk(curves, baseline, title, f"{out_prefix}_passk.png")
        print(f"Wrote {out_prefix}_passk.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True, help="Run directory under results/, e.g. 7b-sigma001-iter50")
    ap.add_argument("--label", dest="labels", action="append", required=True,
                     help="Repeatable. An arm evaluated by run_eval.sh under this run_tag, e.g. base, trained, rl (2 or more required)")
    ap.add_argument("--baseline", default=None,
                     help="Which --label is the reference for the four-way breakdown and k-grid check (default: the first --label given)")
    ap.add_argument("--plot", action="store_true", help="Also save a pass@k curve PNG per benchmark (requires matplotlib)")
    args = ap.parse_args()

    if len(args.labels) < 2:
        ap.error(f"need at least 2 --label arms, got {len(args.labels)}")
    baseline = args.baseline or args.labels[0]
    if baseline not in args.labels:
        ap.error(f"--baseline {baseline!r} is not among the --label values: {args.labels}")

    run_root = os.path.join(REPO_ROOT, "results", args.run_tag)
    for benchmark, title in BENCHMARKS:
        label_dirs = {label: os.path.join(run_root, label, benchmark) for label in args.labels}
        missing = [label for label, d in label_dirs.items() if not os.path.isdir(d)]
        if missing:
            print(f"Skipping {benchmark}: no directory for {missing}")
            continue
        print(f"\n=== {title} ({benchmark}) ===")
        analyze_benchmark(label_dirs, baseline, title, os.path.join(run_root, benchmark), args.plot)


if __name__ == "__main__":
    main()
