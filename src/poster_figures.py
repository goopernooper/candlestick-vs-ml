"""Poster versions of the paper figures: large text, print resolution.

Usage
-----
    python src/poster_figures.py v4

The paper figures (figures.py) are drawn for a printed page, so their labels
are 8-11 pt. On a 48 x 36 in poster each chart is printed 10-18 in wide and
read from a few feet away, which needs about 24 pt. These versions are drawn
at the size they are printed (inches = poster pixels / 96) with 22-34 pt text
and saved at 300 dpi. Titles and notes are left to the poster's captions.

Writes results/<version>/poster/:
  poster_grid.png        MCC by model and encoding        (17.7 x 9.2 in)
  poster_cost_curve.png  long/short return against cost   (17.7 x 9.2 in)
  poster_confident.png   confident days against reversal  (12.0 x 9.4 in)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import (BASELINES, BLUES, COST_MODELS, ENCODINGS, GRID, INK, MODELS, MUTED, NAMES,  # noqa: E402
                     STYLE, SURFACE, name)

ROOT = Path(__file__).resolve().parents[1]
DPI = 300
TEXT = 24        # tick labels and direct labels, in points
LABEL = 26       # axis titles
LINE = 4.5       # data line width
MARKER = 17      # marker size


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "axes.labelsize": LABEL, "xtick.color": INK, "ytick.color": INK,
                         "xtick.labelsize": TEXT, "ytick.labelsize": TEXT, "text.color": INK,
                         "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
                         "xtick.major.size": 8, "ytick.major.size": 8, "axes.labelpad": 14})
    return plt


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def poster_grid(summary: pd.DataFrame, path: Path, size=(17.7, 9.2)) -> None:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    mcc = summary["mcc"]
    values = np.array([[mcc.get(f"{m}_{e}", np.nan) for e, _ in ENCODINGS] for m, _ in MODELS])
    top = max(0.035, float(np.nanmax(values)))
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    im = ax.imshow(values, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=top, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            best = v == np.nanmax(values[i])
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=34,
                    color="#ffffff" if v / top > 0.55 else INK, fontweight="bold" if best else "normal")
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=8)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(ENCODINGS)))
    ax.set_xticklabels([label.replace(" (", "\n(") for _, label in ENCODINGS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([label for _, label in MODELS])
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    bar.ax.set_title("MCC", color=INK, fontsize=LABEL, pad=16)
    bar.set_ticks([0, 0.01, 0.02, 0.03])
    bar.ax.tick_params(labelsize=TEXT - 2, width=1.5, length=6)
    bar.outline.set_visible(False)
    ref = mcc.get("reversal", np.nan)
    if not np.isnan(ref):
        bar.ax.axhline(ref, color=INK, lw=5)
        bar.ax.text(1.1, ref, f" reversal\n {ref:.3f}", transform=bar.ax.get_yaxis_transform(), va="center",
                    fontsize=TEXT - 4, color=INK, fontweight="bold", linespacing=1.0)
    fig.tight_layout()
    _save(fig, path)


def poster_cost(backtest: pd.DataFrame, path: Path, headline: float, size=(17.7, 9.2)) -> None:
    plt = _plt()
    ls = backtest[backtest["strategy"] == "long_short"]
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    bh = ls[ls["model"] == "always_long"].sort_values("cost_bps")
    ax.plot(bh["cost_bps"], bh["ann_return"] * 100, color=INK, lw=LINE, ls="--", zorder=2)
    ends = [(float(bh["ann_return"].iloc[-1] * 100), "buy-and-hold")]
    for m in COST_MODELS:
        g = ls[ls["model"] == m].sort_values("cost_bps")
        if g.empty:
            continue
        color, marker = STYLE.get(m, ("#7a7a7a", "o"))
        ax.plot(g["cost_bps"], g["ann_return"] * 100, color=color, lw=LINE, marker=marker, ms=MARKER,
                mec=SURFACE, mew=2.5, zorder=3)
        ends.append((float(g["ann_return"].iloc[-1] * 100), m))
    ax.axhline(0, color=MUTED, lw=2, zorder=1)
    ax.axvline(headline, color=MUTED, lw=2.5, ls=":", zorder=1)
    lo, hi = ax.get_ylim()
    ax.text(headline + 0.25, hi, f"{headline:g} bps\n(headline)", va="top", fontsize=TEXT - 2, color=MUTED)
    gap = 0.085 * (hi - lo)
    placed = []
    x_end = float(ls["cost_bps"].max())
    for y, key in sorted(ends):
        y_label = max(y, placed[-1] + gap) if placed else y
        placed.append(y_label)
        ax.annotate(NAMES.get(key, key), (x_end, y), xytext=(x_end + 1.6, y_label), va="center",
                    fontsize=TEXT - 1, color=INK, bbox={"fc": SURFACE, "ec": "none", "pad": 1.5}, zorder=4,
                    arrowprops=None if abs(y_label - y) < gap / 3 else {"arrowstyle": "-", "color": MUTED, "lw": 1.5, "relpos": (0, 0.5), "shrinkA": 4})
    ax.set_ylim(lo, max(hi, placed[-1] + gap / 2))
    ax.set_xlim(-0.5, x_end + 14)
    ax.set_xticks(sorted(ls["cost_bps"].unique()))
    ax.set_xlabel("Transaction cost (basis points per side)")
    ax.set_ylabel("Long/short annual return (%)")
    ax.grid(True, color=GRID, lw=1.5)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def poster_confident(vs_reversal: pd.DataFrame, path: Path, coverage: float = 0.1, size=(12.0, 9.4)) -> None:
    plt = _plt()
    d = vs_reversal[(vs_reversal["target_coverage"] == coverage) & ~vs_reversal["model"].isin(BASELINES)]
    d = d.sort_values("diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    for i, r in d.iterrows():
        clears = r["diff_low"] > 0
        ax.plot([r["diff_low"], r["diff_high"]], [i, i], color=INK if clears else "#8a8985", lw=LINE - 1, zorder=2)
        ax.scatter([r["diff"]], [i], s=MARKER ** 2 * 0.9, zorder=3, marker="o",
                   facecolor=INK if clears else SURFACE, edgecolor=INK if clears else "#6f6e6a", linewidth=3)
    ax.axvline(0, color=MUTED, lw=2.5, zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([name(m) for m in d["model"]], fontsize=TEXT - 2)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-0.05, 0, 0.05])
    ax.set_xticklabels(["−0.05", "0", "+0.05"])
    ax.set_xlabel("MCC minus reversal's, same stock-days")
    ax.grid(True, axis="x", color=GRID, lw=1.5)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    out = folder / "poster"
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(folder / "summary.csv").set_index("model")
    backtest = pd.read_csv(folder / "backtest.csv")
    headline = float(summary["cost_bps"].iloc[0]) if "cost_bps" in summary else 5.0
    poster_grid(summary, out / "poster_grid.png")
    poster_cost(backtest, out / "poster_cost_curve.png", headline)
    poster_confident(pd.read_csv(folder / "coverage_vs_reversal.csv"), out / "poster_confident.png")
    print(f"Wrote results/{args.version}/poster/: poster_grid.png, poster_cost_curve.png, poster_confident.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
