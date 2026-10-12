#!/usr/bin/env python3
"""Hero figure: pass@k of base / ES / GRPO, 2x4 grid (rows 1.5B, 7B).

Reads results/{run}/{bench}_threeway_summary.json and writes
writeup/figures/hero_passk.png and hero_passk.pdf.

Usage: python scripts/plot_hero.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("Qwen2.5-1.5B", "1.5b-sigma001-iter50"), ("Qwen2.5-7B", "7b-sigma001-iter50")]
BENCHES = [("aime24", "AIME24"), ("math500", "MATH500"),
           ("minerva_math", "Minerva Math"), ("olympiadbench", "OlympiadBench")]
ARMS = [("base", "Base", "#4d4d4d", "o"),
        ("es", "ES", "#d62728", "s"),
        ("rl", "GRPO (SimpleRL-Zoo)", "#1f77b4", "^")]


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for r, (model, run) in enumerate(RUNS):
        for c, (bench, title) in enumerate(BENCHES):
            d = json.loads((ROOT / "results" / run / f"{bench}_threeway_summary.json").read_text())
            ax = axes[r, c]
            for key, label, colour, marker in ARMS:
                ks = d["ks_rl"] if key == "rl" else d["ks_base_es"]
                ys = [v * 100 for v in d[f"pass_at_k_{key}"]]
                ax.plot(ks, ys, color=colour, marker=marker, ms=4, lw=1.8, label=label)
            ax.set_xscale("log", base=2)
            ks = d["ks_base_es"]
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks], fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_title(f"{title} ({model.split('-')[1]})")
            if r == 1:
                ax.set_xlabel("k")
            if c == 0:
                ax.set_ylabel(f"{model}\npass@k (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=12)
    out = ROOT / "writeup" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hero_passk.png", dpi=150)
    fig.savefig(out / "hero_passk.pdf")
    print("wrote", out / "hero_passk.png", out / "hero_passk.pdf")


if __name__ == "__main__":
    main()
