"""Paper figures from committed results: the model x encoding grid and the cost curve.

Usage
-----
    python src/figures.py v4

Reads results/<version>/summary.csv and backtest.csv (both committed, so this
runs anywhere) and writes:

  figure_grid.png        MCC of every machine learning model on every encoding,
                         with the pattern models and the reversal baseline for
                         reference (paper Section 8.3)
  figure_cost_curve.png  long/short annual return against the cost per side,
                         for the main models, reversal and buy-and-hold
                         (paper Section 8.4)

The other figures are written by coverage.py (coverage.png,
coverage_vs_reversal.png) and regimes.py (regime_episodes.png).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import GRID, HIGHLIGHT, INK, MUTED, SURFACE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("lightgbm", "LightGBM"), ("rf", "Random forest"), ("svm", "SVM (RBF)"), ("mlp", "MLP")]
ENCODINGS = [("a", "Encoding A\n61 named patterns"), ("b", "Encoding B\n3-day candle shape"),
             ("ohlc", "Normalized prices\n3 days")]
# sequential blue, light -> dark (validated reference palette)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COST_MODELS = ["svm_b", "svm_ohlc", "lightgbm_b", "rf_b", "knn_b3", "pattern_miner", "reversal"]
NAMES = {"svm_b": "SVM, encoding B", "svm_ohlc": "SVM, normalized prices", "lightgbm_b": "LightGBM, encoding B",
         "rf_b": "Random forest, encoding B", "knn_b3": "k-NN, 3-day", "pattern_miner": "Pattern miner",
         "reversal": "Reversal (baseline)", "buy-and-hold": "Buy-and-hold"}


def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def grid_figure(summary: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style(plt)
    mcc = summary["mcc"]
    values = np.array([[mcc.get(f"{m}_{e}", np.nan) for e, _ in ENCODINGS] for m, _ in MODELS])
    top = max(0.035, float(np.nanmax(values)))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=top, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            dark = v / top > 0.55
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark else INK, fontweight="bold" if v == np.nanmax(values[i]) else "normal")
    # thin surface-coloured gaps between cells
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(ENCODINGS)))
    ax.set_xticklabels([label for _, label in ENCODINGS], fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([label for _, label in MODELS])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    bar.set_label("MCC, 2005–2022", color=INK)
    bar.outline.set_visible(False)
    ref = mcc.get("reversal", np.nan)
    if not np.isnan(ref):
        bar.ax.axhline(ref, color=INK, lw=2)
    others = [f"{name} {mcc[key]:.3f}" for key, name in
              [("pattern_miner", "pattern miner"), ("knn_b3", "k-NN 3-day"), ("knn_b5", "k-NN 5-day")] if key in mcc]
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.055, f"Bold: the model's best encoding. Black line on the scale: the reversal baseline "
             f"({ref:.3f}).", fontsize=8, color=MUTED)
    fig.text(0.01, 0.015, "Pattern-based models on encoding B: " + ", ".join(others) + ".", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def cost_figure(backtest: pd.DataFrame, path: Path, title: str, headline: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    ls = backtest[backtest["strategy"] == "long_short"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bh = ls[ls["model"] == "always_long"].sort_values("cost_bps")
    ax.plot(bh["cost_bps"], bh["ann_return"] * 100, color=INK, lw=1.6, ls="--", zorder=2)
    ends = [(float(bh["ann_return"].iloc[-1] * 100), "buy-and-hold")]
    colors = {**{m: HIGHLIGHT[m] for m in HIGHLIGHT}, "svm_ohlc": ("#6250d6", "X"), "rf_b": HIGHLIGHT["rf_b"]}
    for m in COST_MODELS:
        g = ls[ls["model"] == m].sort_values("cost_bps")
        if g.empty:
            continue
        color, marker = colors.get(m, ("#7a7a7a", "o"))
        ax.plot(g["cost_bps"], g["ann_return"] * 100, color=color, lw=2, marker=marker, ms=7,
                mec=SURFACE, mew=1.2, zorder=3)
        ends.append((float(g["ann_return"].iloc[-1] * 100), m))
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(headline, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(headline, ax.get_ylim()[1], f" headline cost\n {headline:g} bps", va="top", fontsize=8, color=MUTED)
    # labels at the right end, nudged apart
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    placed = []
    x_end = float(ls["cost_bps"].max())
    for y, name in sorted(ends):
        y_label = max(y, placed[-1] + gap) if placed else y
        placed.append(y_label)
        ax.annotate(NAMES.get(name, name), (x_end, y), xytext=(x_end + 0.6, y_label), va="center", fontsize=8.5,
                    color=INK, bbox={"fc": SURFACE, "ec": "none", "pad": 1.0}, zorder=4,
                    arrowprops=None if abs(y_label - y) < gap / 3 else {"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_xlim(-0.5, x_end + 9)
    ax.set_xticks(sorted(ls["cost_bps"].unique()))
    ax.set_xlabel("Transaction cost (basis points per side)")
    ax.set_ylabel("Long/short annual return (%)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.01, "Compounded annual return of the equal-weighted long/short portfolio, 2005–2022. "
             "A line crosses zero at the model's break-even cost.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    summary = pd.read_csv(folder / "summary.csv").set_index("model")
    backtest = pd.read_csv(folder / "backtest.csv")
    from features import load_config
    headline = load_config()["backtest"]["cost_bps_per_side"]
    grid_figure(summary, folder / "figure_grid.png", "Predictive accuracy by model and encoding")
    cost_figure(backtest, folder / "figure_cost_curve.png", "Returns fall with trading costs", headline)
    print(f"Wrote results/{args.version}/figure_grid.png and figure_cost_curve.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
