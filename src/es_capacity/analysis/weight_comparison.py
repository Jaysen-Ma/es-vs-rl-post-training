"""Stream checkpoint pairs into weight statistics and fixed-grid histograms."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from es_capacity.models import MODELS

# Directory names are derived from the same model IDs as the evaluators.
SCALES = {
    scale: {arm: MODELS[f"{scale}-{arm}"].split("/")[-1]
            for arm in ("base", "ES", "RL")}
    for scale in dict.fromkeys(name.split("-")[0] for name in MODELS)
}

SERIES = ("base", "trained", "delta")
N_BINS = 512
SIGMA_SPAN = 6.0
# 16M float32 elements is 64 MB per buffer; the float64 4th-moment temporaries
# on top of that keep peak well under a gigabyte.
CHUNK_ELEMS = 16_000_000

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


# --------------------------------------------------------------------------
# checkpoint plumbing
# --------------------------------------------------------------------------

def tensor_map(d: Path) -> dict[str, str]:
    """{tensor name: shard filename} for sharded or single-file checkpoints."""
    from safetensors import safe_open

    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    with safe_open(d / "model.safetensors", framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


class Checkpoint:
    """Lazy reader that keeps one open handle per shard."""

    def __init__(self, d: Path):
        self.dir = d
        self.map = tensor_map(d)
        self._handles: dict[str, object] = {}

    def _handle(self, name: str):
        from safetensors import safe_open

        shard = self.map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.dir / shard, framework="pt")
        return self._handles[shard]

    def chunks(self, name: str):
        """Yield float32 slices of at most CHUNK_ELEMS elements.

        Large tensors are read a row-block at a time via get_slice so the full
        tensor never lands in memory; small ones are read whole, which is both
        simpler and faster.
        """
        import torch

        h = self._handle(name)
        sl = h.get_slice(name)
        shape = sl.get_shape()
        total = 1
        for s in shape:
            total *= s
        if total <= CHUNK_ELEMS or len(shape) < 2:
            yield h.get_tensor(name).reshape(-1).to(torch.float32)
            return
        row = total // shape[0]
        step = max(1, CHUNK_ELEMS // max(row, 1))
        for start in range(0, shape[0], step):
            yield sl[start:start + step].reshape(-1).to(torch.float32)


def is_matrix(name: str) -> bool:
    """Weight matrices only.

    Norms sit near 1.0 while every other parameter sits near 0, so pooling them
    turns a raw weight histogram into two disconnected spikes. Biases are
    dropped for the same reason at a smaller scale. Both are still reported, as
    their own group, so nothing vanishes silently.
    """
    return (name.endswith(".weight")
            and "layernorm" not in name
            and name != "model.norm.weight")


def module_of(name: str) -> str:
    if "embed_tokens" in name:
        return "embed_tokens"
    if name.startswith("lm_head"):
        return "lm_head"
    for m in MODULES:
        if m in name:
            return m
    return "other"


def depth_of(name: str) -> str:
    m = LAYER_RE.search(name)
    if m:
        return m.group(1)
    if "embed_tokens" in name:
        return "embed"
    if name.startswith("lm_head"):
        return "head"
    return "other"


def groups_of(name: str) -> list[tuple[str, str]]:
    """Every (grouping, group) an element of this tensor contributes to."""
    return [("pooled", "all"),
            ("module", module_of(name)),
            ("depth", depth_of(name))]


# --------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------

class Pass1:
    """count / sum / sum of squares / min / max / exact-zero count."""

    def __init__(self):
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.lo = math.inf
        self.hi = -math.inf
        self.zeros = 0

    def add(self, x: torch.Tensor) -> None:
        xd = x.double()
        self.n += x.numel()
        self.s1 += xd.sum().item()
        self.s2 += (xd * xd).sum().item()
        self.lo = min(self.lo, x.min().item())
        self.hi = max(self.hi, x.max().item())
        self.zeros += int((x == 0).sum().item())

    @property
    def mean(self) -> float:
        return self.s1 / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        var = self.s2 / self.n - self.mean ** 2
        return math.sqrt(max(var, 0.0))


class Pass2:
    """Histogram plus the 4th central moment, both against pass 1's mean.

    Bin index -1 is the underflow bin and N_BINS the overflow bin, so summed
    counts always equal the parameter count exactly.

    The binning grid is passed separately from the series' own mean and std,
    because the base and trained histograms have to land on the IDENTICAL grid
    to be overlaid. Their sigmas differ by well under a percent, and binning
    each on its own +/-6 sigma leaves the two grids slightly offset, which
    shows up as a spurious interference pattern in the tails of the overlay.
    Moments still use the series' own mean and std.
    """

    def __init__(self, mean: float, std: float,
                 grid_mean: float | None = None, grid_std: float | None = None):
        import torch

        self.mean = mean
        self.std = std
        gm = mean if grid_mean is None else grid_mean
        gs = std if grid_std is None else grid_std
        span = SIGMA_SPAN * gs if gs > 0 else 1.0
        self.lo = gm - span
        self.width = 2 * span / N_BINS
        self.counts = torch.zeros(N_BINS + 2, dtype=torch.int64)
        self.s4 = 0.0

    def add(self, x: torch.Tensor) -> None:
        import torch

        idx = torch.floor((x - self.lo) / self.width).to(torch.int64)
        idx.clamp_(-1, N_BINS)
        self.counts += torch.bincount(idx + 1, minlength=N_BINS + 2)
        d = (x - self.mean).double()
        d *= d
        self.s4 += (d * d).sum().item()

    def edges(self) -> list[tuple[float, float]]:
        return [(self.lo + i * self.width, self.lo + (i + 1) * self.width)
                for i in range(N_BINS)]

    def excess_kurtosis(self, n: int) -> float:
        if n < 2 or self.std <= 0:
            return 0.0
        return self.s4 / n / self.std ** 4 - 3.0

    def quantile(self, q: float, n: int) -> float:
        """Quantile read off the histogram, linear inside the landing bin.

        Bin width is 12 sigma / 512 = 0.023 sigma, so the p99.9 this returns is
        good to about +/-0.01 sigma — enough to separate a Gaussian's 3.09 from
        a heavy-tailed 3.6, which is the only thing it is used for.
        """
        target = q * n
        seen = 0
        for i, c in enumerate(self.counts.tolist()):
            if seen + c >= target:
                if i == 0:
                    return self.lo
                if i == N_BINS + 1:
                    return self.lo + N_BINS * self.width
                frac = (target - seen) / c if c else 0.0
                return self.lo + (i - 1 + frac) * self.width
            seen += c
        return self.lo + N_BINS * self.width


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------

def shared_names(base: Checkpoint, arm: Checkpoint) -> tuple[list[str], list[str]]:
    """Matrix tensors present in both, plus the excluded (norm/bias) names.

    Comparing only the intersection keeps the ES and RL arms at one scale
    covering the identical parameter set, which is what makes their pooled
    numbers comparable. A tied-embedding base has no lm_head.weight while an
    FSDP-saved checkpoint materialises one; that extra head is reported
    separately rather than folded into the pooled statistics.
    """
    both = [n for n in base.map if n in arm.map]
    return ([n for n in both if is_matrix(n)],
            [n for n in both if not is_matrix(n)])


def compare(base_dir: Path, arm_dir: Path, out_dir: Path, label: str) -> dict:
    base, arm = Checkpoint(base_dir), Checkpoint(arm_dir)
    names, excluded = shared_names(base, arm)
    extra = sorted(set(arm.map) - set(base.map))

    t0 = time.time()
    p1: dict = defaultdict(Pass1)
    per_tensor: dict[str, dict] = {}

    print(f"  pass 1/2  {len(names)} matrix tensors", flush=True)
    for name in names:
        acc = Pass1()  # this tensor alone, for per_tensor.csv
        base_sq = 0.0
        for b, a in zip(base.chunks(name), arm.chunks(name)):
            d = a - b
            acc.add(d)
            base_sq += (b.double() ** 2).sum().item()
            for grouping, group in groups_of(name):
                p1[("base", grouping, group)].add(b)
                p1[("trained", grouping, group)].add(a)
                p1[("delta", grouping, group)].add(d)
            del b, a, d
        per_tensor[name] = dict(
            n=acc.n, delta_std=acc.std, delta_max_abs=max(abs(acc.lo), abs(acc.hi)),
            zero_frac=acc.zeros / acc.n if acc.n else 0.0,
            rel_l2=math.sqrt(acc.s2 / base_sq) if base_sq else 0.0,
            base_sq=base_sq,
        )

    # Norms and biases, and a materialised tied head: measured, kept out of the
    # matrix groupings so they cannot distort a raw weight histogram.
    for name in excluded:
        key = "norms" if ("norm" in name) else "biases"
        for b, a in zip(base.chunks(name), arm.chunks(name)):
            p1[("base", "excluded", key)].add(b)
            p1[("trained", "excluded", key)].add(a)
            p1[("delta", "excluded", key)].add(a - b)
            del b, a
    if "lm_head.weight" in extra and "model.embed_tokens.weight" in base.map:
        for b, a in zip(base.chunks("model.embed_tokens.weight"),
                        arm.chunks("lm_head.weight")):
            p1[("base", "tied_head", "lm_head")].add(b)
            p1[("trained", "tied_head", "lm_head")].add(a)
            p1[("delta", "tied_head", "lm_head")].add(a - b)
            del b, a

    # base and trained share the base series' grid; delta gets its own, since
    # it is ~40x narrower and would otherwise land entirely in one bin.
    p2 = {}
    for key, v in p1.items():
        series, grouping, group = key
        ref = p1[("base", grouping, group)] if series in ("base", "trained") else v
        p2[key] = Pass2(v.mean, v.std, ref.mean, ref.std)

    print(f"  pass 2/2  {len(p2)} (series, grouping, group) accumulators",
          flush=True)
    for name in names:
        for b, a in zip(base.chunks(name), arm.chunks(name)):
            d = a - b
            for grouping, group in groups_of(name):
                p2[("base", grouping, group)].add(b)
                p2[("trained", grouping, group)].add(a)
                p2[("delta", grouping, group)].add(d)
            del b, a, d
    for name in excluded:
        key = "norms" if ("norm" in name) else "biases"
        for b, a in zip(base.chunks(name), arm.chunks(name)):
            p2[("base", "excluded", key)].add(b)
            p2[("trained", "excluded", key)].add(a)
            p2[("delta", "excluded", key)].add(a - b)
            del b, a
    if ("delta", "tied_head", "lm_head") in p2:
        for b, a in zip(base.chunks("model.embed_tokens.weight"),
                        arm.chunks("lm_head.weight")):
            p2[("base", "tied_head", "lm_head")].add(b)
            p2[("trained", "tied_head", "lm_head")].add(a)
            p2[("delta", "tied_head", "lm_head")].add(a - b)
            del b, a

    out_dir.mkdir(parents=True, exist_ok=True)
    write_stats(out_dir / "stats.csv", p1, p2)
    write_hist(out_dir / "hist.csv", p2)
    write_per_tensor(out_dir / "per_tensor.csv", per_tensor)

    pooled_d = p1[("delta", "pooled", "all")]
    base_sq = sum(v["base_sq"] for v in per_tensor.values())
    rel = math.sqrt(pooled_d.s2 / base_sq) if base_sq else 0.0
    print(f"  {label}: rel L2 {rel * 100:.4f}%  delta std {pooled_d.std:.3e}  "
          f"excess kurtosis {p2[('delta', 'pooled', 'all')].excess_kurtosis(pooled_d.n):+.2f}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return dict(rel_l2=rel, n=pooled_d.n)


def write_stats(path: Path, p1: dict, p2: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "grouping", "group", "n", "mean", "std", "min",
                    "max", "excess_kurtosis", "p999_over_sigma", "zero_frac"])
        for key in sorted(p1):
            series, grouping, group = key
            a, b = p1[key], p2[key]
            ratio = (b.quantile(0.999, a.n) - a.mean) / a.std if a.std > 0 else 0.0
            w.writerow([series, grouping, group, a.n,
                        f"{a.mean:.8e}", f"{a.std:.8e}",
                        f"{a.lo:.8e}", f"{a.hi:.8e}",
                        f"{b.excess_kurtosis(a.n):.4f}", f"{ratio:.4f}",
                        f"{a.zeros / a.n if a.n else 0.0:.6f}"])


def write_hist(path: Path, p2: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "grouping", "group", "bin", "left", "right", "count"])
        for key in sorted(p2):
            series, grouping, group = key
            acc = p2[key]
            counts = acc.counts.tolist()
            w.writerow([series, grouping, group, "underflow", "", f"{acc.lo:.8e}",
                        counts[0]])
            for i, (lo, hi) in enumerate(acc.edges()):
                if counts[i + 1]:
                    w.writerow([series, grouping, group, i, f"{lo:.8e}",
                                f"{hi:.8e}", counts[i + 1]])
            w.writerow([series, grouping, group, "overflow",
                        f"{acc.lo + N_BINS * acc.width:.8e}", "", counts[-1]])


def write_per_tensor(path: Path, per_tensor: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tensor", "module", "depth", "n", "rel_l2", "delta_std",
                    "delta_max_abs", "zero_frac"])
        for name, v in per_tensor.items():
            w.writerow([name, module_of(name), depth_of(name), v["n"],
                        f"{v['rel_l2']:.8e}", f"{v['delta_std']:.8e}",
                        f"{v['delta_max_abs']:.8e}", f"{v['zero_frac']:.6f}"])

