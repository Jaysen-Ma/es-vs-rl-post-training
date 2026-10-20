#!/usr/bin/env python3
"""Figures for the weight-movement distributions, from compare.py's CSVs.

Reads only the reduced CSVs, so the figures can be reworked without touching a
checkpoint again — the reduction is minutes per pair, this is instant.

EVERYTHING IS IN RAW UNITS. The three quantities involved span roughly 2,000x
(base weights sigma ~2.8e-2, the ES change ~7.0e-4, the RL change ~1.3e-5), so
no single linear axis can hold them. The answer used throughout is a log COUNT
axis with a shared raw x-axis, which lets a narrow distribution and a wide one
sit in the same panel honestly, plus separate panels where the ranges are too
far apart for even that. There is never a second x-scale in one panel.

Usage:
    python scripts/weights/plot.py --scale 1.5B
    python scripts/weights/plot.py --scale 7B --dir results/weights
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Same arm -> colour mapping as DISPLAY in scripts/passk/analyze.py, so these
# figures read as one system with the committed pass@k plots. Copied rather
# than imported: three hex values do not justify a shared module in a repo with
# no package structure.
COLORS = {"base": "#1b7a72", "ES": "#e8776a", "RL": "#6a5acd"}
NAMES = {"base": "Base", "ES": "ES-trained", "RL": "RL (SimpleRL-Zoo)"}

N_BINS = 512
MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]


def read_stats(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out[(r["series"], r["grouping"], r["group"])] = {
                "n": int(r["n"]), "mean": float(r["mean"]), "std": float(r["std"]),
                "min": float(r["min"]), "max": float(r["max"]),
                "kurt": float(r["excess_kurtosis"]),
                "p999": float(r["p999_over_sigma"]),
                "zero": float(r["zero_frac"]),
            }
    return out


def read_hist(path: Path, stats: dict) -> dict:
    """Rebuild the full 512-bin count array per group.

    compare.py only writes non-empty bins, so the empty ones are restored here
    from the serialized underflow/overflow edges. Base and trained share
    the base grid, even when their own means and standard deviations differ.
    Empty bins remain zero here and are masked when drawing a log axis.
    """
    raw: dict = {}
    for path_row in [path]:
        with path_row.open() as fh:
            for r in csv.DictReader(fh):
                key = (r["series"], r["grouping"], r["group"])
                if r["bin"] in ("underflow", "overflow"):
                    bins = raw.setdefault(key, {})
                    bins[r["bin"]] = int(r["count"])
                    if r["bin"] == "underflow":
                        bins["lo"] = float(r["right"])
                    else:
                        bins["hi"] = float(r["left"])
                    continue
                raw.setdefault(key, {})[int(r["bin"])] = int(r["count"])
    out = {}
    for key, bins in raw.items():
        st = stats.get(key)
        if st is None or st["std"] <= 0:
            continue
        lo = bins["lo"]
        width = (bins["hi"] - lo) / N_BINS
        counts = np.zeros(N_BINS)
        for i, c in bins.items():
            if isinstance(i, int):
                counts[i] = c
        out[key] = {
            "centers": lo + (np.arange(N_BINS) + 0.5) * width,
            "counts": counts, "width": width,
            "under": bins.get("underflow", 0), "over": bins.get("overflow", 0),
            **st,
        }
    return out


def masked(counts: np.ndarray) -> np.ndarray:
    return np.where(counts > 0, counts, np.nan)


def density(h: dict) -> np.ndarray:
    """Counts divided by bin width — parameters per unit of weight change.

    NOT raw counts. Each series is binned over its own +/-6 sigma, so the ES and
    RL histograms have bin widths ~600x apart, and a raw count says "how many
    parameters fell in a window of this arm's own width". Plotting those against
    each other silently compares two different y quantities in one panel, which
    is the dual-axis mistake wearing a disguise: it made the far more
    concentrated RL change look no taller than the ES change, when in truth it
    is ~250x taller.

    Dividing by bin width fixes it. Both curves then integrate to the same
    parameter count, so a narrower distribution has to be taller, which is what
    "the weights barely moved" should look like.
    """
    return masked(h["counts"]) / h["width"]


def rebin(h: dict, factor: int) -> dict:
    """Sum adjacent bins. Exact — the CSV keeps the full 512-bin resolution."""
    if factor <= 1:
        return h
    n = (N_BINS // factor) * factor
    counts = h["counts"][:n].reshape(-1, factor).sum(axis=1)
    centers = h["centers"][:n].reshape(-1, factor).mean(axis=1)
    return {**h, "counts": counts, "centers": centers,
            "width": h["width"] * factor}


def auto_rebin(h: dict) -> dict:
    """Widen bins until the histogram stops combing.

    bfloat16 checkpoints store weights on a floating-point grid, so a
    difference of two bf16 values is a multiple of that grid's step — around
    6e-5 for a typical Qwen2.5 weight, which is several times WIDER than the
    512-bin resolution over +/-6 sigma. Binning finer than the grid puts every
    representable value in its own bin and leaves the bins between them empty,
    which renders as a comb rather than a distribution. The data is not noisy;
    the bins are simply finer than the data's own resolution.

    Detection is a SMOOTHNESS test, not an emptiness test. Emptiness only shows
    up in the smaller model: at 7B there are enough parameters that even a
    suppressed bin holds thousands, so the comb is a modulation between adjacent
    bins rather than a run of zeros. Comparing each bin to its neighbour catches
    both — a smooth histogram changes by a few percent per bin, a combed one by
    an order of magnitude.

    Returns the smallest factor that passes, so resolution is never given away
    for free.
    """
    return rebin(h, choose_factor(h))


def choose_factor(h: dict, region: float = 2.0) -> int:
    """region is the half-width, in sigma, that has to come out smooth.

    The default of 2 sigma suits the weight-CHANGE histograms, whose quantisation
    step is roughly constant because the changes themselves are all of one
    magnitude. Raw WEIGHTS need the full range instead: the bfloat16 step scales
    with the value, so it is fine near zero and coarse in the wings, and a
    criterion that only inspects the middle certifies a histogram that combs
    badly at the edges.
    """
    for factor in (1, 2, 4, 8, 16, 32):
        r = rebin(h, factor)
        core = np.abs(r["centers"] - r["mean"]) <= region * r["std"]
        if core.sum() < 12:
            return factor
        c = r["counts"][core]
        if (c == 0).mean() > 0.02:
            continue
        step = np.abs(np.diff(np.log10(c[c > 0])))
        if step.size and np.median(step) <= 0.12:
            return factor
    return 32


def common_factor(hs: list[dict], region: float = 2.0) -> int:
    """One bin width for every series drawn in the same panel.

    Choosing per series looks fine in the core and wrong in the tails: the test
    above only inspects the central +/-2 sigma, so a float32 checkpoint (smooth
    there) keeps full resolution while a bfloat16 one gets widened, and the two
    curves then disagree bin-for-bin out in the wings where the bfloat16 grid is
    coarse. Overlaid series have to share a grid to be comparable at all.
    """
    return max(choose_factor(h, region) for h in hs)


def gaussian_ref(h: dict) -> np.ndarray:
    """Density of a normal with the same n, mean and sigma — matches density()."""
    z = (h["centers"] - h["mean"]) / h["std"]
    pdf = np.exp(-0.5 * z ** 2) / (h["std"] * math.sqrt(2 * math.pi))
    return h["n"] * pdf


def style(ax) -> None:
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)


def headroom(ax, decades: float = 1.6) -> None:
    """Open space above the peak so the legend does not sit on the data.

    These are log-count histograms peaking in the middle of the panel, which is
    exactly where a legend wants to go. Raising the top of the axis is less
    disruptive than moving the legend into a tail.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 10 ** decades)


def fig_raw_weights(H: dict, scale: str, out: Path) -> None:
    """Base / ES / RL raw parameter distributions, pooled over weight matrices.

    They very nearly coincide, which is the result: neither method reshapes the
    weight distribution. The log count axis is what makes the tails legible,
    where any difference would have to show up.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    # Base is drawn first, thick and semi-transparent, so the two trained arms
    # can sit on top of it and still leave it visible. They coincide almost
    # exactly, which is the point of the figure.
    spec = [("base", ("ES", "base"), 4.5, 0.45),
            ("ES", ("ES", "trained"), 1.6, 1.0),
            ("RL", ("RL", "trained"), 1.6, 1.0)]
    present = [(arm, H[src][(series, "pooled", "all")], lw, alpha)
               for arm, (src, series), lw, alpha in spec
               if (series, "pooled", "all") in H[src]]
    # All three share one grid (compare.py bins trained on the base's range),
    # so they must also share one bin width.
    factor = common_factor([h for _, h, _, _ in present], region=5.0)
    for arm, h, lw, alpha in present:
        h = rebin(h, factor)
        ax.step(h["centers"], density(h), where="mid",
                color=COLORS[arm], linewidth=lw, alpha=alpha,
                label=f"{NAMES[arm]}   $\\sigma$ = {h['std']:.5f}")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter value (raw)")
    ax.set_ylabel("Parameters per unit weight")
    ax.set_title(f"Qwen2.5-{scale}: raw weight distribution, all weight matrices")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.text(0.5, -0.19, "residual sawtooth in the tails is the bfloat16 storage "
            "grid, whose step grows with the value — not a difference between arms",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#555555")
    style(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def label_with_shape(h: dict, arm: str) -> str:
    """Legend label carrying the two numbers that make the argument.

    Kept in the legend rather than as free-floating text: the text block
    otherwise collides with the legend box, and this way the colour that
    identifies the arm sits right beside its numbers.
    """
    return (f"{NAMES[arm]}\n   $\\sigma$ = {h['std']:.2e},  "
            f"excess kurtosis {h['kurt']:+.2f}")


def draw_both_arms(ax, es: dict, rl: dict, fontsize: float = 8.0) -> None:
    """Both arms' weight-change distributions on one shared raw x-axis.

    Shared by the two-panel local figure and the standalone one committed to
    the README, so the published figure cannot drift from the analysis one.
    """
    ax.step(es["centers"], density(es), where="mid",
            color=COLORS["ES"], linewidth=1.8, label=label_with_shape(es, "ES"))
    ax.plot(es["centers"], gaussian_ref(es), color="#8a8a8a", linewidth=1.2,
            linestyle="--", label="Gaussian, same $\\sigma$")
    ax.step(rl["centers"], density(rl), where="mid",
            color=COLORS["RL"], linewidth=1.8, label=label_with_shape(rl, "RL"))
    ax.set_yscale("log")
    ax.set_xlim(es["centers"][0], es["centers"][-1])
    ax.set_xlabel("Weight change (raw)")
    ax.set_ylabel("Parameters per unit change")
    style(ax)
    headroom(ax, 2.2)
    ax.legend(loc="upper right", frameon=True, fontsize=fontsize)


def fig_weight_change(H: dict, scale: str, out: Path) -> None:
    """Standalone version of the shared-scale panel — this one is committed.

    Two of these sit side by side in the README, so it has to stand on its own:
    its own title naming the model, and the width and tail-heaviness of each arm
    in the legend rather than in surrounding panels.
    """
    es = auto_rebin(H["ES"][("delta", "pooled", "all")])
    rl = auto_rebin(H["RL"][("delta", "pooled", "all")])
    fig, ax = plt.subplots(figsize=(5.9, 4.3))
    draw_both_arms(ax, es, rl, fontsize=8)
    ax.set_title(f"Qwen2.5-{scale}: per-parameter weight change", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_delta_pooled(H: dict, scale: str, out: Path) -> None:
    """The headline: how far each method actually moved each parameter.

    Left panel puts both arms on the ES arm's raw x-range, so the ~50x width
    difference is read straight off the axis. Right panel zooms to the RL arm's
    own range so its shape is visible at all. A Gaussian with the same sigma is
    drawn on each: the ES change lands on it, the RL change does not.
    """
    es = auto_rebin(H["ES"][("delta", "pooled", "all")])
    rl_wide = H["RL"][("delta", "pooled", "all")]
    rl = auto_rebin(rl_wide)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    draw_both_arms(axes[0], es, rl, fontsize=7.5)
    axes[0].set_title("Both arms on one raw scale")

    ax = axes[1]
    ax.step(rl["centers"], density(rl), where="mid",
            color=COLORS["RL"], linewidth=1.8, label=NAMES["RL"])
    ax.plot(rl["centers"], gaussian_ref(rl), color="#8a8a8a", linewidth=1.2,
            linestyle="--", label="Gaussian, same $\\sigma$")
    ax.set_yscale("log")
    ax.set_xlim(rl["centers"][0], rl["centers"][-1])
    ax.set_xlabel("Weight change (raw)")
    ax.set_title(f"RL arm alone, {es['std'] / rl['std']:.0f}$\\times$ narrower axis")
    style(ax)
    headroom(ax, 1.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    fig.suptitle(f"Qwen2.5-{scale}: distribution of per-parameter weight change",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_delta_by_module(H: dict, S: dict, scale: str, out: Path) -> None:
    """Weight change split by which projection the parameter belongs to.

    A facet grid with both arms in every panel does not work here: the two are
    ~50x apart in width, so one is always a spike, and every ES panel looks the
    same because the ES change genuinely is the same everywhere. So the arms
    get a panel each, with every module overlaid inside it. That makes the
    actual contrast the thing you see — the ES curves land on top of each other,
    the RL curves do not — and a third panel carries the per-module sizes, which
    are unreadable off a histogram.
    """
    mods = [m for m in MODULE_ORDER
            if ("delta", "module", m) in H["ES"] and ("delta", "module", m) in H["RL"]]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    for ax, arm, title in [(axes[0], "ES", "ES: the same width in every module"),
                           (axes[1], "RL", "RL: width varies by module")]:
        raw = [H[arm][("delta", "module", m)] for m in mods]
        factor = common_factor(raw)
        widest = None
        for h in raw:
            h = rebin(h, factor)
            ax.step(h["centers"], density(h), where="mid",
                    color=COLORS[arm], linewidth=1.3, alpha=0.55)
            if widest is None or h["std"] > widest["std"]:
                widest = h
        ax.plot(widest["centers"], gaussian_ref(widest), color="#8a8a8a",
                linewidth=1.2, linestyle="--", label="Gaussian, same $\\sigma$")
        ax.set_yscale("log")
        ax.set_xlim(widest["centers"][0], widest["centers"][-1])
        ax.set_xlabel("Weight change (raw)")
        ax.set_title(title, fontsize=10)
        handles = [plt.Line2D([], [], color=COLORS[arm], linewidth=1.3,
                              alpha=0.55, label=f"{NAMES[arm]}, one line per module"),
                   plt.Line2D([], [], color="#8a8a8a", linewidth=1.2,
                              linestyle="--", label="Gaussian, same $\\sigma$")]
        style(ax)
        headroom(ax, 1.8)
        ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=7.5)
    axes[0].set_ylabel("Parameters per unit change")
    axes[0].text(0.5, -0.22, "area under each curve is that module's parameter "
                 "count, so a narrower change is a taller curve",
                 transform=axes[0].transAxes, ha="center", fontsize=7.5,
                 color="#555555")

    ax = axes[2]
    ys = np.arange(len(mods))
    for arm, offset in (("ES", -0.15), ("RL", 0.15)):
        xs = [S[arm][("delta", "module", m)]["std"] for m in mods]
        ax.plot(xs, ys + offset, "o", markersize=7, color=COLORS[arm],
                label=NAMES[arm])
    ax.set_yticks(ys)
    ax.set_yticklabels(mods, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.margins(x=0.18)
    ax.set_xlabel("Std. dev. of weight change (raw)")
    ax.set_title("Size of the change per module", fontsize=10)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    style(ax)

    fig.suptitle(f"Qwen2.5-{scale}: weight change by module, raw units",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_delta_by_depth(S: dict, scale: str, out: Path) -> None:
    """How large the change is against layer index.

    Deliberately a line chart, not a grid of histograms: the question here is
    magnitude against position, not shape. The two lines sit ~50x apart on a
    log axis, which is the honest picture of the size difference.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm in ("ES", "RL"):
        pts = sorted((int(g), v["std"]) for (s, gr, g), v in S[arm].items()
                     if s == "delta" and gr == "depth" and g.isdigit())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8,
                color=COLORS[arm], label=NAMES[arm])
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(
        labelOnlyBase=False, minor_thresholds=(2, 0.5)))
    ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Std. dev. of weight change (raw)")
    ax.set_title(f"Qwen2.5-{scale}: size of the weight change against depth")
    ax.legend(loc="best", frameon=True, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--dir", type=Path, default=Path("results/weights"))
    args = ap.parse_args()

    S, H = {}, {}
    for arm in ("ES", "RL"):
        d = args.dir / f"{args.scale}-{arm}"
        if not (d / "stats.csv").exists():
            raise SystemExit(f"missing {d}/stats.csv — run compare.py --scale "
                             f"{args.scale} first")
        S[arm] = read_stats(d / "stats.csv")
        H[arm] = read_hist(d / "hist.csv", S[arm])

    out = args.dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_raw_weights(H, args.scale, out / f"raw_weights_{args.scale}.png")
    fig_delta_pooled(H, args.scale, out / f"delta_pooled_{args.scale}.png")
    fig_weight_change(H, args.scale, out / f"weight_change_{args.scale}.png")
    fig_delta_by_module(H, S, args.scale, out / f"delta_by_module_{args.scale}.png")
    fig_delta_by_depth(S, args.scale, out / f"delta_by_depth_{args.scale}.png")
    print(f"wrote 5 figures to {out}/ for {args.scale}")


if __name__ == "__main__":
    main()
