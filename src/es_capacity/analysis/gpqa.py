"""Analysis of saved GPQA seed sweeps, independent of model execution."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

TASK = "gpqa_diamond_zeroshot"


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar (binomial sign test on the discordant pairs)."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) * 0.5**n
    return min(1.0, 2.0 * tail)


def load_arm(arm_dir: Path) -> tuple[dict, dict[int, int], dict[int, str], dict[int, int]]:
    """Return (results_json, {doc_id: correct}, {doc_id: doc_hash}, {doc_id: picked}).

    `picked` is the index of the choice with the highest log-likelihood, i.e.
    the slot the model actually went for.
    """
    results_paths = sorted(arm_dir.rglob("results_*.json"))
    if not results_paths:
        raise FileNotFoundError(f"no results_*.json under {arm_dir}")
    results_path = results_paths[-1]
    results = json.loads(results_path.read_text())
    stamp = results_path.name.removeprefix("results_").removesuffix(".json")
    samples_path = results_path.with_name(f"samples_{TASK}_{stamp}.jsonl")
    if not samples_path.is_file():
        raise FileNotFoundError(f"missing samples for selected run: {samples_path}")

    correct: dict[int, int] = {}
    hashes: dict[int, str] = {}
    picked: dict[int, int] = {}
    with samples_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            doc_id = rec["doc_id"]
            if doc_id in correct:
                raise ValueError(f"duplicate doc_id {doc_id} in {samples_path}")
            correct[doc_id] = int(rec["acc"])
            hashes[doc_id] = rec.get("doc_hash", "")
            lls = [x[0] for x in rec["filtered_resps"]]
            picked[doc_id] = max(range(len(lls)), key=lambda i: lls[i])
    expected = results["n-samples"][TASK]["effective"]
    if len(correct) != expected:
        raise ValueError(
            f"incomplete samples in {samples_path}: expected {expected}, found {len(correct)}"
        )
    return results, correct, hashes, picked


def load_sweep(root: Path) -> dict[int, dict[str, tuple]]:
    """Load results, correctness, hashes and chosen slots from seed{N}/ARM/."""
    sweep: dict[int, dict[str, tuple]] = {}
    for seed_dir in sorted(root.glob("seed*")):
        m = re.fullmatch(r"seed(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        sweep[seed] = {
            d.name: load_arm(d) for d in sorted(seed_dir.iterdir()) if d.is_dir()
        }
    if not sweep:
        raise ValueError(f"no seed*/ directories under {root}")
    return sweep


def check_invariants(sweep: dict[int, dict[str, tuple]]) -> list[str]:
    """Return a list of problems; empty means the sweep is internally valid."""
    problems: list[str] = []
    if not sweep:
        return ["no seeds in sweep"]
    reference_arms = set(sweep[min(sweep)])
    reference_ids = None

    # 1. Within a seed: all arms scored the same shuffled items.
    for seed, arms in sorted(sweep.items()):
        if set(arms) != reference_arms:
            problems.append(f"seed {seed}: model arms differ between seeds")
        for name, (_, correct, hashes, _) in arms.items():
            if not correct:
                problems.append(f"seed {seed}: {name} has no per-question samples")
            if not all(hashes.values()):
                problems.append(f"seed {seed}: {name} has missing doc_hash values")
        with_samples = [a for a, (_, c, _, _p) in arms.items() if c]
        if not with_samples:
            problems.append(f"seed {seed}: no arm has per-question samples")
            continue
        ref_name = with_samples[0]
        ref = arms[ref_name][2]
        if reference_ids is None:
            reference_ids = set(ref)
        elif set(ref) != reference_ids:
            problems.append(f"seed {seed}: question IDs differ between seeds")
        for name in with_samples[1:]:
            other = arms[name][2]
            bad = [d for d in sorted(ref.keys() | other.keys()) if other.get(d) != ref.get(d)]
            if bad:
                problems.append(
                    f"seed {seed}: {name} scored different items than {ref_name} "
                    f"at doc_ids {bad[:5]}{'...' if len(bad) > 5 else ''} "
                    f"({len(bad)} docs) — arms are not comparable"
                )

    # 2. Across seeds: the permutation actually changed.
    sigs: dict[int, tuple] = {}
    for seed, arms in sorted(sweep.items()):
        for _, (_, c, h, _p) in arms.items():
            if h:
                sigs[seed] = tuple(h[d] for d in sorted(h))
                break
    seeds = sorted(sigs)
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1 :]:
            if sigs[s1] == sigs[s2]:
                problems.append(
                    f"seeds {s1} and {s2} scored an IDENTICAL permutation — "
                    "datasets caching defeated the reseed "
                    "(call datasets.disable_caching() before building tasks); "
                    "these seeds carry no independent information"
                )
    return problems


def wilcoxon_signed_rank(diffs: list[int]) -> tuple[float, int]:
    """Two-sided Wilcoxon signed-rank over per-question count differences."""
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 1.0, 0
    from scipy.stats import wilcoxon

    return float(wilcoxon(nz).pvalue), len(nz)


def summarize(sweep: dict[int, dict[str, tuple]], pairs: list[str]) -> dict[str, list[dict]]:
    """Compute the existing CSV rows after validating the sweep and pairs."""
    problems = check_invariants(sweep)
    if problems:
        raise ValueError("; ".join(problems))
    seeds = sorted(sweep)
    arms = sorted({arm for values in sweep.values() for arm in values})
    for spec in pairs:
        arm, separator, baseline = spec.partition(":")
        if not separator or not arm or not baseline or ":" in baseline:
            raise ValueError(f"invalid pair {spec!r}; expected ARM:BASELINE")
        if arm not in arms or baseline not in arms:
            raise ValueError(f"{spec}: arm or baseline missing from sweep")
    # --- scores.csv / scores_summary.csv ---------------------------------
    score_rows = []
    by_arm: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        for arm, (results, correct, _, _p) in sorted(sweep[seed].items()):
            metrics = results["results"][TASK]
            n_total = results["n-samples"][TASK]["effective"]
            acc = metrics["acc,none"]
            by_arm[arm].append(acc)
            score_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "acc": round(acc, 6),
                    "n_correct": round(acc * n_total),
                    "n_total": n_total,
                    "acc_stderr": round(metrics["acc_stderr,none"], 6),
                    "has_per_question": bool(correct),
                    "model_path": results["config"]["model_args"]["pretrained"],
                }
            )

    summary_rows = []
    for arm in arms:
        accs = by_arm[arm]
        summary_rows.append(
            {
                "arm": arm,
                "n_seeds": len(accs),
                "acc_mean": round(statistics.fmean(accs), 6),
                "acc_std": round(statistics.stdev(accs), 6) if len(accs) > 1 else 0.0,
                "acc_min": round(min(accs), 6),
                "acc_max": round(max(accs), 6),
            }
        )

    # --- per_question.csv: how many seeds each arm got each doc right -----
    counts: dict[str, dict[int, int]] = {}
    for arm in arms:
        acc_by_doc: dict[int, int] = defaultdict(int)
        for seed in seeds:
            _, correct, _, _p = sweep[seed][arm]
            for d, v in correct.items():
                acc_by_doc[d] += v
        counts[arm] = acc_by_doc
    doc_ids = sorted(next(iter(counts.values())))
    question_rows = [
        {"doc_id": d, "n_seeds": len(seeds), **{a: counts[a][d] for a in arms}}
        for d in doc_ids
    ]

    # --- mcnemar.csv (per seed) + pairs_summary.csv (across seeds) --------
    mc_rows, pair_rows = [], []
    for spec in pairs:
        arm, _, baseline = spec.partition(":")
        deltas = []
        for seed in seeds:
            a = sweep[seed][arm][1]
            b = sweep[seed][baseline][1]
            ids = sorted(set(a) & set(b))
            gained = sum(1 for d in ids if a[d] and not b[d])
            lost = sum(1 for d in ids if b[d] and not a[d])
            delta = 100 * (sum(a[d] for d in ids) - sum(b[d] for d in ids)) / len(ids)
            deltas.append(delta)
            mc_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "baseline": baseline,
                    "acc_arm": round(sum(a[d] for d in ids) / len(ids), 6),
                    "acc_baseline": round(sum(b[d] for d in ids) / len(ids), 6),
                    "delta_pts": round(delta, 2),
                    "gained": gained,
                    "lost": lost,
                    "p_mcnemar_exact": round(mcnemar_exact(gained, lost), 4),
                }
            )
        if not deltas:
            continue
        diffs = [counts[arm][d] - counts[baseline][d] for d in doc_ids]
        p_wilcoxon, n_nonzero = wilcoxon_signed_rank(diffs)
        pair_rows.append(
            {
                "arm": arm,
                "baseline": baseline,
                "n_seeds": len(deltas),
                "delta_mean_pts": round(statistics.fmean(deltas), 2),
                "delta_std_pts": round(statistics.stdev(deltas), 2)
                if len(deltas) > 1
                else 0.0,
                "delta_min_pts": round(min(deltas), 2),
                "delta_max_pts": round(max(deltas), 2),
                "seeds_arm_better": sum(1 for d in deltas if d > 0),
                "seeds_tied": sum(1 for d in deltas if d == 0),
                "n_questions_differing": n_nonzero,
                "p_wilcoxon_per_question": round(p_wilcoxon, 4),
            }
        )

    # --- position_bias.csv ------------------------------------------------
    # GPQA places the correct answer uniformly at random, so slot frequencies
    # far from 25% are the model's own positional preference. Two arms that
    # prefer different slots will swap ranks as the permutation changes, which
    # is what makes a single-permutation comparison untrustworthy.
    bias_rows = []
    for arm in arms:
        tally = [0, 0, 0, 0]
        for seed in seeds:
            for _d, slot in sweep[seed][arm][3].items():
                tally[slot] += 1
        total = sum(tally) or 1
        pcts = [100 * t / total for t in tally]
        bias_rows.append(
            {
                "arm": arm,
                "n_decisions": total,
                **{f"pct_{c}": round(p, 2) for c, p in zip("ABCD", pcts)},
                "spread_pts": round(max(pcts) - min(pcts), 2),
                "favoured_slot": "ABCD"[max(range(4), key=lambda i: pcts[i])],
            }
        )

    tables = {
        "scores.csv": score_rows,
        "scores_summary.csv": summary_rows,
        "per_question.csv": question_rows,
        "position_bias.csv": bias_rows,
    }
    if mc_rows:
        tables["mcnemar.csv"] = mc_rows
    if pair_rows:
        tables["pairs_summary.csv"] = pair_rows
    return tables


def write_summaries(tables: dict[str, list[dict]], output: Path) -> None:
    """Write computed tables; callers select and create the output directory."""
    for name, rows in tables.items():
        with (output / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # An earlier run with --pair must not appear in a later unpaired report.
    for name in ("mcnemar.csv", "pairs_summary.csv"):
        if name not in tables:
            (output / name).unlink(missing_ok=True)


def format_console_summary(tables: dict[str, list[dict]]) -> str:
    """Render the existing console summary without writing files."""
    summary_rows = tables["scores_summary.csv"]
    bias_rows = tables["position_bias.csv"]
    pair_rows = tables.get("pairs_summary.csv", [])
    lines = []
    # --- console summary --------------------------------------------------
    lines.append(f"{'arm':<12} {'mean acc':>9}  {'std':>5}  {'min':>5}  {'max':>5}")
    for r in summary_rows:
        lines.append(
            f"{r['arm']:<12} {r['acc_mean'] * 100:8.2f}%  "
            f"{r['acc_std'] * 100:5.2f}  {r['acc_min'] * 100:5.2f}  {r['acc_max'] * 100:5.2f}"
        )
    lines.append(f"\n{'arm':<12} {'A':>7} {'B':>7} {'C':>7} {'D':>7}  {'spread':>7}")
    for r in bias_rows:
        lines.append(
            f"{r['arm']:<12} " + " ".join(f"{r['pct_' + c]:6.1f}%" for c in "ABCD")
            + f"  {r['spread_pts']:6.1f}"
        )
    if pair_rows:
        lines.append(
            f"\n{'pair':<26} {'mean delta':>11}  {'range':>15}  "
            f"{'wins':>6}  {'wilcoxon':>9}"
        )
        for r in pair_rows:
            rng = f"{r['delta_min_pts']:+.2f}..{r['delta_max_pts']:+.2f}"
            lines.append(
                f"{r['arm'] + ' vs ' + r['baseline']:<26} "
                f"{r['delta_mean_pts']:+7.2f}±{r['delta_std_pts']:<4.2f}  "
                f"{rng:>15}  {r['seeds_arm_better']:>2}/{r['n_seeds']:<3}  "
                f"{r['p_wilcoxon_per_question']:>9}"
            )
    return "\n".join(lines) + "\n"


def read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def format_tables(results_dir: Path) -> str:
    """Read reduced CSVs and render the study's Markdown tables."""
    lines = []
    summary = {r["arm"]: r for r in read(results_dir / "scores_summary.csv")}
    bias = {r["arm"]: r for r in read(results_dir / "position_bias.csv")}
    pairs_path = results_dir / "pairs_summary.csv"
    pairs = read(pairs_path) if pairs_path.exists() else []
    order = [a for a in
             ["1.5B-base", "1.5B-ES", "1.5B-RL", "7B-base", "7B-ES", "7B-RL"]
             if a in summary]
    order += sorted(set(summary) - set(order))
    if not order:
        raise ValueError(f"no arms in {results_dir / 'scores_summary.csv'}")
    n_seeds = summary[order[0]]["n_seeds"]

    lines.append(f"| Arm | mean acc over {n_seeds} permutations | sd | min | max |")
    lines.append("|---|---|---|---|---|")
    for a in order:
        r = summary[a]
        lines.append(
            f"| {a} | {float(r['acc_mean']) * 100:.2f}% | "
            f"{float(r['acc_std']) * 100:.2f} | {float(r['acc_min']) * 100:.2f}% | "
            f"{float(r['acc_max']) * 100:.2f}% |"
        )

    lines.append("")
    lines.append("| Comparison | mean delta (pts) | range | arm ahead on | Wilcoxon p |")
    lines.append("|---|---|---|---|---|")
    for r in pairs:
        lines.append(
            f"| {r['arm']} vs {r['baseline']} | "
            f"**{float(r['delta_mean_pts']):+.2f} ± {float(r['delta_std_pts']):.2f}** | "
            f"{float(r['delta_min_pts']):+.2f} … {float(r['delta_max_pts']):+.2f} | "
            f"{r['seeds_arm_better']}/{r['n_seeds']} | {r['p_wilcoxon_per_question']} |"
        )

    lines.append("")
    lines.append("| Arm | picks (A) | (B) | (C) | (D) | spread |")
    lines.append("|---|---|---|---|---|---|")
    for a in order:
        r = bias[a]
        cells = " | ".join(f"{float(r['pct_' + c]):.1f}%" for c in "ABCD")
        lines.append(f"| {a} | {cells} | {float(r['spread_pts']):.1f} pts |")
    return "\n".join(lines) + "\n"
