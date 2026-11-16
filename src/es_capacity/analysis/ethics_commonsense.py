"""Summarize saved ETHICS predictions without loading models or datasets."""

from __future__ import annotations

import json
from pathlib import Path

from es_capacity.models import MODELS


def load_preds(path: Path) -> list[dict]:
    """One record per scenario: gold label, predicted choice, correct."""
    rows = []
    seen = set()
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["doc_id"] in seen:
                raise ValueError(f"{path}: duplicate question ID {rec['doc_id']!r}")
            seen.add(rec["doc_id"])
            lls = [pair[0] for pair in rec["filtered_resps"]]
            if rec["target"] not in (0, 1, "0", "1") or len(lls) != 2:
                raise ValueError(f"{path}: expected a binary label and two choice scores for question {rec['doc_id']!r}")
            gold = int(rec["target"])
            pred = max(range(len(lls)), key=lls.__getitem__)
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

    for arm, rows in preds.items():
        if not rows:
            raise ValueError(f"{arm}: samples.jsonl contains no questions")
        base_name = arm.split("-", 1)[0] + "-base"
        if arm != base_name and base_name in preds:
            baseline = {row["doc_id"]: row["gold"] for row in preds[base_name]}
            labels = {row["doc_id"]: row["gold"] for row in rows}
            if labels.keys() != baseline.keys():
                raise ValueError(f"{arm} and {base_name}: question IDs do not match")
            if labels != baseline:
                raise ValueError(f"{arm} and {base_name}: labels do not match")

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
        def class_accuracy(gold):
            chosen = [r for r in rows if r["gold"] == gold]
            return sum(r["correct"] for r in chosen) / len(chosen) if chosen else None

        acc0, acc1 = class_accuracy(0), class_accuracy(1)
        base_name = arm.split("-", 1)[0] + "-base"
        b = c = p = ""
        tail = ""
        if arm != base_name and base_name in preds:
            # b: base wrong, trained right. c: base right, trained wrong.
            b_n = c_n = 0
            for left, right in zip(preds[base_name], rows):
                if (not left["correct"]) and right["correct"]:
                    b_n += 1
                elif left["correct"] and (not right["correct"]):
                    c_n += 1
            # Two-sided exact test on the discordant pairs.
            k = b_n + c_n
            p_val = 1.0 if k == 0 else float(binomtest(b_n, k, 0.5).pvalue)
            b, c, p = str(b_n), str(c_n), f"{p_val:.4f}"
            tail = f"  +{b_n}/-{c_n}  p={p_val:.3f}"
        class_csv = [f"{value:.6f}" if value is not None else "" for value in (acc0, acc1)]
        class_display = [f"{value * 100:{width}.2f}%" if value is not None else "n/a".rjust(width + 1)
                         for value, width in ((acc0, 6), (acc1, 7))]
        lines.append(f"{arm},{n},{acc:.6f},{py:.6f},{class_csv[0]},{class_csv[1]},{b},{c},{p}")
        print(
            f"{arm:12} {n:5d}  {acc*100:5.2f}%  {py*100:5.1f}%  {class_display[0]}  {class_display[1]}{tail}",
            flush=True,
        )
    (out_root / "summary.csv").write_text("\n".join(lines) + "\n")
