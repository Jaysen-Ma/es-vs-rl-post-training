"""Summarize saved ETHICS predictions without loading models or datasets."""

from __future__ import annotations

import json
from pathlib import Path

GROUP_SIZE = {"deontology": 4, "justice": 4, "virtue": 5}
TASKS = ["deontology", "justice", "virtue", "util"]
DEFAULT_ARMS = ["7B-base", "7B-ES", "7B-RL", "1.5B-base", "1.5B-ES", "1.5B-RL"]


def load_task(path: Path, task: str) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["subtask"] != task:
                continue
            lls = [pair[0] for pair in rec["filtered_resps"]]
            pred = max(range(len(lls)), key=lls.__getitem__)
            gold = int(rec["target"])
            doc = rec["doc"]
            rows.append({
                "doc_id": rec["doc_id"],
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "logodds": lls[1] - lls[0],
                "group_id": doc.get("group_id"),
                "pair_id": doc.get("pair_id"),
                "canon": doc.get("canon"),
            })
    rows.sort(key=lambda r: r["doc_id"])
    return rows


def _mcnemar(left: list[bool], right: list[bool]) -> tuple[int, int, float]:
    from scipy.stats import binomtest

    b = c = 0
    for a, d in zip(left, right):
        if (not a) and d:
            b += 1
        elif a and (not d):
            c += 1
    k = b + c
    p = 1.0 if k == 0 else float(binomtest(b, k, 0.5).pvalue)
    return b, c, p


def _units(rows: list[dict], task: str) -> dict[int, bool]:
    """Correctness by complete group/pair ID; omit incomplete evaluation units."""
    if task == "util":
        by_pair: dict[int, dict[int, dict]] = {}
        for row in rows:
            # label 1 means slot 1 is the pleasanter scenario.
            by_pair.setdefault(row["pair_id"], {})[row["gold"]] = row
        out = {}
        for pair_id in sorted(by_pair):
            sides = by_pair[pair_id]
            if 0 not in sides or 1 not in sides:
                continue
            out[pair_id] = sides[1]["logodds"] > sides[0]["logodds"]
        return out
    by_group: dict[int, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    size = GROUP_SIZE[task]
    return {
        g: all(r["correct"] for r in by_group[g])
        for g in sorted(by_group)
        if len(by_group[g]) == size and len({r["doc_id"] for r in by_group[g]}) == size
    }


def summarize(out_root: Path) -> None:
    arms = [a for a in DEFAULT_ARMS if (out_root / a / "rest_samples.jsonl").is_file()]
    # Arms run outside the default list still show up, in directory order.
    for path in sorted(out_root.iterdir()):
        if (path / "rest_samples.jsonl").is_file() and path.name not in arms:
            arms.append(path.name)
    if not arms:
        return

    lines = ["task,arm,n,example_acc,p_positive,set_n,set_acc,set_b,set_c,set_p"]
    print("\ntask         arm          ex_acc   P(pos)   set_acc   vs base", flush=True)
    for task in TASKS:
        loaded = {arm: load_task(out_root / arm / "rest_samples.jsonl", task) for arm in arms}
        bases = {}
        for arm, rows in loaded.items():
            base_name = arm.split("-", 1)[0] + "-base"
            if base_name in loaded:
                bases[arm] = _units(loaded[base_name], task)
        for arm, rows in loaded.items():
            scored = rows
            if task == "util":
                scored = [r for r in rows if r["canon"] == 1]
            n = len(scored)
            ex = sum(r["correct"] for r in scored) / n if n else None
            ppos = sum(r["pred"] == 1 for r in scored) / n if n else None
            units = _units(rows, task)
            set_acc = sum(units.values()) / len(units) if units else None
            b = c = p = ""
            tail = ""
            if (units and arm in bases and bases[arm].keys() == units.keys()
                    and not arm.endswith("-base")):
                b_n, c_n, p_val = _mcnemar(
                    [bases[arm][k] for k in units], [units[k] for k in units]
                )
                b, c, p = str(b_n), str(c_n), f"{p_val:.4g}"
                tail = f"  +{b_n}/-{c_n}  p={p_val:.3g}"
            rates = [f"{v:.6f}" if v is not None else "" for v in (ex, ppos, set_acc)]
            lines.append(
                f"{task},{arm},{n},{rates[0]},{rates[1]},{len(units)},{rates[2]},{b},{c},{p}"
            )
            percentages = [f"{v * 100:6.{precision}f}%" if v is not None else "    n/a"
                           for v, precision in ((ex, 2), (ppos, 1), (set_acc, 2))]
            print(
                f"{task:12} {arm:12} " + "  ".join(percentages) + tail,
                flush=True,
            )
    (out_root / "rest_summary.csv").write_text("\n".join(lines) + "\n")
