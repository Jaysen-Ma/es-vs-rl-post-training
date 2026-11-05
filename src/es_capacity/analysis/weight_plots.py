"""Headless rendering for saved weight distributions; imported only to plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from es_capacity.analysis.weights import (  # noqa: E402
    COLORS, NAMES, MODULE_ORDER, auto_rebin, common_factor, density, gaussian_ref, rebin,
)


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
