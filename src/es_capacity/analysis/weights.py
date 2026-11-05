"""Load and analyse saved weight statistics without opening model checkpoints."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

    z = (h["centers"] - h["mean"]) / h["std"]
    pdf = np.exp(-0.5 * z ** 2) / (h["std"] * math.sqrt(2 * math.pi))
    return h["n"] * pdf


def render_figures(results_dir: Path, scale: str, output: Path) -> list[Path]:
    """Load both saved comparisons and render the study's five PNG figures."""
    # Check every required file before importing optional libraries or writing.
    for arm in ("ES", "RL"):
        for filename in ("stats.csv", "hist.csv"):
            path = results_dir / f"{scale}-{arm}" / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing {path}; provide saved compare.py outputs")
    stats, histograms = {}, {}
    for arm in ("ES", "RL"):
        directory = results_dir / f"{scale}-{arm}"
        stats[arm] = read_stats(directory / "stats.csv")
        histograms[arm] = read_hist(directory / "hist.csv", stats[arm])
        for series in ("base", "trained", "delta"):
            if (series, "pooled", "all") not in histograms[arm]:
                raise ValueError(f"{scale}-{arm}: missing nonzero-variance pooled {series} histogram")
    if not any(("delta", "module", module) in histograms["ES"]
               and ("delta", "module", module) in histograms["RL"] for module in MODULE_ORDER):
        raise ValueError("no shared weight-change module histograms for ES and RL")

    from es_capacity.analysis import weight_plots

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, function, inputs in (
        ("raw_weights", weight_plots.fig_raw_weights, (histograms,)),
        ("delta_pooled", weight_plots.fig_delta_pooled, (histograms,)),
        ("weight_change", weight_plots.fig_weight_change, (histograms,)),
        ("delta_by_module", weight_plots.fig_delta_by_module, (histograms, stats)),
        ("delta_by_depth", weight_plots.fig_delta_by_depth, (stats,)),
    ):
        path = output / f"{name}_{scale}.png"
        function(*inputs, scale, path)
        paths.append(path)
    return paths
