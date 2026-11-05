"""Summarize saved ETHICS predictions without loading models or datasets."""

from __future__ import annotations

import json
from pathlib import Path

from es_capacity.models import MODELS


def load_preds(path: Path) -> list[dict]:
    """One record per scenario: gold label, predicted choice, correct."""
    rows = []
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            lls = [pair[0] for pair in rec["filtered_resps"]]
            pred = max(range(len(lls)), key=lls.__getitem__)
            gold = int(rec["target"])
            rows.append({
                "doc_id": rec["doc_id"],
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
            })
    rows.sort(key=lambda r: r["doc_id"])
    return rows


def summarize(out_root: Path, arms: list[str]) -> None:
    """Accuracy, yes-rate, and McNemar against the matching base.

    Includes every arm already on disk, so a later 7B run does not drop the
    1.5B rows from summary.csv.
    """
    from scipy.stats import binomtest

    present = [p.name for p in sorted(out_root.iterdir()) if (p / "samples.jsonl").is_file()]
    ordered = [a for a in list(MODELS) + arms if a in present or a in arms]
    # Stable, scale-then-arm order, duplicates removed, disk-only arms kept.
    seen = []
    for arm in ordered:
        if arm not in seen and (out_root / arm / "samples.jsonl").is_file():
            seen.append(arm)

    preds = {}
    for arm in seen:
        preds[arm] = load_preds(out_root / arm / "samples.jsonl")
    if not preds:
        return

    def yes_rate(rows, mask):
        chosen = [r for r in rows if mask(r)]
        if not chosen:
            return float("nan"), 0
        return sum(r["pred"] == 1 for r in chosen) / len(chosen), len(chosen)

    lines = ["arm,n,acc,pred_yes,acc_when_not_wrong,acc_when_wrong,mcnemar_b,mcnemar_c,mcnemar_p"]
    print("\narm          n     acc   P(yes)  acc|not  acc|wrong   vs base", flush=True)
    for arm, rows in preds.items():
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n
        py, _ = yes_rate(rows, lambda r: True)
        acc0 = sum(r["correct"] for r in rows if r["gold"] == 0) / sum(r["gold"] == 0 for r in rows)
        acc1 = sum(r["correct"] for r in rows if r["gold"] == 1) / sum(r["gold"] == 1 for r in rows)
        base_name = arm.split("-", 1)[0] + "-base"
        b = c = p = ""
        tail = ""
        if arm != base_name and base_name in preds and len(preds[base_name]) == n:
            # b: base wrong, trained right. c: base right, trained wrong.
            b_n = c_n = 0
            for left, right in zip(preds[base_name], rows):
                if left["doc_id"] != right["doc_id"]:
                    raise RuntimeError(f"doc_id mismatch in {arm}")
                if (not left["correct"]) and right["correct"]:
                    b_n += 1
                elif left["correct"] and (not right["correct"]):
                    c_n += 1
            # Two-sided exact test on the discordant pairs.
            k = b_n + c_n
            p_val = 1.0 if k == 0 else float(binomtest(b_n, k, 0.5).pvalue)
            b, c, p = str(b_n), str(c_n), f"{p_val:.4f}"
            tail = f"  +{b_n}/-{c_n}  p={p_val:.3f}"
        lines.append(f"{arm},{n},{acc:.6f},{py:.6f},{acc0:.6f},{acc1:.6f},{b},{c},{p}")
        print(
            f"{arm:12} {n:5d}  {acc*100:5.2f}%  {py*100:5.1f}%  {acc0*100:6.2f}%  {acc1*100:7.2f}%{tail}",
            flush=True,
        )
    (out_root / "summary.csv").write_text("\n".join(lines) + "\n")
