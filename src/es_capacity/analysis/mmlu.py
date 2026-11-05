"""Summarize saved MMLU results and paired model comparisons.

Overall accuracies come from the evaluator's results.json. Exact McNemar tests
use per-question samples, with Bonferroni correction over available trained-vs-
base comparisons. No model runtime is imported by this module.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

TASK = "mmlu"
ARM_ORDER = ["1.5B-base", "1.5B-ES", "1.5B-RL", "7B-base", "7B-ES", "7B-RL"]


def load(arm_dir: Path) -> dict:
    res = json.loads((arm_dir / "results.json").read_text())
    # n-samples is keyed only by the 57 leaf subjects, so the overall count has
    # to be summed over them. group_subtasks nests one level deep:
    # mmlu -> 4 categories -> 57 subjects.
    counts = {k: v["effective"] for k, v in res["n-samples"].items()}
    members = res.get("group_subtasks", {})

    def n_for(task: str) -> int:
        if task in counts:
            return counts[task]
        return sum(n_for(child) for child in members.get(task, []))

    cfg = res.get("config", {})
    return {
        "acc": res["results"][TASK]["acc,none"],
        "n": n_for(TASK),
        "shots": cfg.get("num_fewshot", "?"),
        "seconds": cfg.get("wall_seconds", "?"),
    }


def correctness(arm_dir: Path) -> dict[tuple[str, int], bool]:
    """Per-question correctness, keyed (subtask, doc_id), from the sample dump."""
    out = {}
    with (arm_dir / "samples.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["subtask"], r["doc_id"])
            if key in out:
                raise ValueError(f"{arm_dir}: duplicate question ID {key!r}")
            out[key] = bool(r["acc"])
    return out


def summarize(results_dir: Path) -> list[dict]:
    """Return ordered score rows without writing to the evaluation directory."""
    found = {d.name: d for d in results_dir.iterdir()
             if d.is_dir() and (d / "results.json").is_file()}
    arms = [a for a in ARM_ORDER if a in found] + sorted(set(found) - set(ARM_ORDER))
    if not arms:
        raise ValueError(f"no arms with results.json under {results_dir}")

    data = {a: load(found[a]) for a in arms}
    ok = {a: correctness(found[a]) for a in arms}
    trained = [a for a in arms
               if not a.endswith("-base") and f"{a.split('-')[0]}-base" in ok]

    rows = []
    for arm in arms:
        base = f"{arm.split('-')[0]}-base"
        gained = lost = p_raw = p_adj = ""
        if arm in trained:
            if ok[arm].keys() != ok[base].keys():
                raise ValueError(f"{arm} and {base}: question IDs do not match")
            from scipy.stats import binomtest

            gained = sum(1 for k in ok[arm] if ok[arm][k] and not ok[base][k])
            lost = sum(1 for k in ok[arm] if ok[base][k] and not ok[arm][k])
            p_raw = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
            p_adj = round(min(1.0, p_raw * len(trained)), 6)
            p_raw = round(p_raw, 6)
        rows.append({"arm": arm, "acc": round(data[arm]["acc"], 6), "n": data[arm]["n"],
                     "gained": gained, "lost": lost,
                     "mcnemar_p": p_raw, "mcnemar_p_adj": p_adj,
                     "num_fewshot": data[arm]["shots"],
                     "wall_seconds": data[arm]["seconds"]})

    return rows


def write_scores(rows: list[dict], path: Path) -> None:
    """Write the existing scores.csv schema to an explicit path."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[dict]) -> str:
    """Render the study's existing Markdown table."""
    lines = ["| Arm | MMLU 0-shot | McNemar p vs. base |", "|---|---|---|"]
    for row in rows:
        p = "—" if row["mcnemar_p_adj"] == "" else f"{row['mcnemar_p_adj']:.2f}"
        lines.append(f"| {row['arm']} | {row['acc'] * 100:.2f}% | {p} |")
    return "\n".join(lines)
