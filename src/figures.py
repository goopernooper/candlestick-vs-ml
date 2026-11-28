"""Paper figures from committed results.

Usage
-----
    python src/figures.py v4

Reads committed results files in results/<version>/ (so this runs anywhere,
without the raw data) and writes the five paper figures:

  figure_grid.png          Figure 1: MCC of every machine learning model on every
                           encoding (Section 9.3)
  figure_cost_curve.png    Figure 2: long/short annual return against the cost
                           per side (Section 9.4)
  figure_coverage.png      Figure 3: MCC against the share of stock-days acted on
                           (Section 9.5; from coverage.csv)
  figure_confident.png     Figure 4: each model against reversal on its 10% most
                           confident stock-days (Section 9.5; from
                           coverage_vs_reversal.csv)
  figure_episodes.png      Figure 5: high-volatility MCC by episode and with each
                           episode left out (Section 9.6; from regime_episodes*.csv)

coverage.py and regimes.py write working versions of Figures 3-5 when their
analyses run; the versions here use readable model names and the paper's
colours.
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
# one colour and marker per model, the same in every figure
STYLE = {**HIGHLIGHT, "svm_ohlc": ("#6250d6", "X")}
FAMILY = {"lightgbm": "LightGBM", "rf": "Random forest", "svm": "SVM", "mlp": "MLP"}
ENCODING = {"a": "encoding A", "b": "encoding B", "ohlc": "normalized prices"}
FOCUS = ["svm_ohlc", "svm_b", "lightgbm_b", "rf_b", "knn_b3", "pattern_miner"]
BASELINES = {"majority", "always_long", "previous_day", "random", "reversal"}


def name(model: str) -> str:
    """Readable name for any model key, e.g. 'rf_ohlc' -> 'Random forest, normalized prices'."""
    if model in NAMES:
        return NAMES[model]
    fixed = {"knn_b5": "k-NN, 5-day", "lightgbm_today": "LightGBM, today's return only"}
    if model in fixed:
        return fixed[model]
    family, _, enc = model.partition("_")
    return f"{FAMILY.get(family, family)}, {ENCODING.get(enc, enc)}" if enc else model


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


def _clean(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def coverage_figure(coverage: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(9.2, 4.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cov = coverage[~coverage["model"].isin(BASELINES)]
    for m, g in cov.groupby("model"):
        if m not in FOCUS:
            g = g.sort_values("coverage")
            ax.plot(g["coverage"] * 100, g["mcc"], color="#b9b8b4", lw=1, zorder=1)
    handles = []
    order = sorted(FOCUS, key=lambda m: -float(cov[(cov["model"] == m)].sort_values("coverage")["mcc"].iloc[0]))
    for m in order:
        g = cov[cov["model"] == m].sort_values("coverage")
        color, marker = STYLE[m]
        (h,) = ax.plot(g["coverage"] * 100, g["mcc"], color=color, lw=2, marker=marker, ms=7, mec=SURFACE,
                       mew=1.2, zorder=3, label=name(m))
        handles.append(h)
    rev = coverage[coverage["model"] == "reversal"]
    if not rev.empty:
        color, marker = STYLE["reversal"]
        h = ax.scatter([100], [float(rev["mcc"].iloc[0])], color=color, marker=marker, s=80, zorder=4,
                       edgecolor=SURFACE, linewidth=1, label="Reversal (acts on every day)")
        handles.append(h)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Share of stock-days acted on (most confident first)")
    ax.set_ylabel("MCC on the stock-days acted on")
    _clean(ax)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.01, "Test years 2005–2022. Confidence cutoffs set on each fold's validation year and applied "
             "unchanged to its test year. Grey: the other models.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def confident_figure(vs_reversal: pd.DataFrame, path: Path, title: str, coverage: float = 0.1) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    d = vs_reversal[(vs_reversal["target_coverage"] == coverage) & ~vs_reversal["model"].isin(BASELINES)]
    d = d.sort_values("diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(d) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, r in d.iterrows():
        clears = r["diff_low"] > 0
        ax.plot([r["diff_low"], r["diff_high"]], [i, i], color=INK if clears else "#9a9995", lw=1.6, zorder=2)
        ax.scatter([r["diff"]], [i], s=46, zorder=3, marker="o",
                   facecolor=INK if clears else SURFACE, edgecolor=INK if clears else "#7a7975", linewidth=1.3)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([name(m) for m in d["model"]], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("MCC minus reversal's MCC on the same stock-days")
    _clean(ax)
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.01, f"Each model's {coverage:.0%} most confident stock-days, 2005–2022. Lines: 95% "
             "stationary-bootstrap intervals over dates. Filled: interval above zero.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


EPISODE_MODELS = ["svm_b", "lightgbm_b", "rf_b", "knn_b3", "pattern_miner", "reversal"]


def episodes_figure(episodes: pd.DataFrame, leave_one_out: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.9), sharey=True, facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [5, 5]})
    names = list(dict.fromkeys(episodes["episode"]))
    dropped = ["none"] + [e for e in names if e != "other" and e in set(leave_one_out["dropped"])]
    width = 0.12
    offsets = (np.arange(len(EPISODE_MODELS)) - (len(EPISODE_MODELS) - 1) / 2) * width
    handles = []
    for k, m in enumerate(EPISODE_MODELS):
        color, marker = STYLE[m]
        e = episodes[episodes["model"] == m].set_index("episode").reindex(names)
        h = left.scatter(np.arange(len(names)) + offsets[k], e["mcc"], color=color, marker=marker, s=46,
                         edgecolor=SURFACE, linewidth=0.8, zorder=3, label=name(m))
        handles.append(h)
        o = leave_one_out[leave_one_out["model"] == m].set_index("dropped").reindex(dropped)
        right.scatter(np.arange(len(dropped)) + offsets[k], o["mcc"], color=color, marker=marker, s=46,
                      edgecolor=SURFACE, linewidth=0.8, zorder=3)
    first = episodes.drop_duplicates("episode").set_index("episode")
    pretty = lambda e: e.replace("-", "–") if e != "other" else "Other days"
    left.set_xticks(range(len(names)))
    left.set_xticklabels([f"{pretty(e)}\n{int(first.loc[e, 'episode_days'])} days\n"
                          f"({first.loc[e, 'share_of_high_days']:.0%})" for e in names], fontsize=8.5)
    right.set_xticks(range(len(dropped)))
    right.set_xticklabels(["All high-\nvolatility days" if d == "none" else f"Without\n{pretty(d)}"
                           for d in dropped], fontsize=8.5)
    left.set_title("By episode", loc="left", fontsize=10)
    right.set_title("With one episode left out", loc="left", fontsize=10)
    left.set_ylabel("MCC on high-volatility days")
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        _clean(ax)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
    right.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.01, "High volatility: SPY's 21-day volatility in the top third of all earlier days. "
             "Test years 2005–2022.", fontsize=8, color=MUTED)
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
    written = ["figure_grid.png", "figure_cost_curve.png"]
    if (folder / "coverage.csv").exists():
        coverage_figure(pd.read_csv(folder / "coverage.csv"), folder / "figure_coverage.png",
                        "Accuracy rises on the days a model is most confident about")
        written.append("figure_coverage.png")
    if (folder / "coverage_vs_reversal.csv").exists():
        confident_figure(pd.read_csv(folder / "coverage_vs_reversal.csv"), folder / "figure_confident.png",
                         "Confident days: each model against reversal on the same stock-days")
        written.append("figure_confident.png")
    if (folder / "regime_episodes.csv").exists() and (folder / "regime_episodes_leave_one_out.csv").exists():
        episodes_figure(pd.read_csv(folder / "regime_episodes.csv"),
                        pd.read_csv(folder / "regime_episodes_leave_one_out.csv"), folder / "figure_episodes.png",
                        "High-volatility predictability by episode")
        written.append("figure_episodes.png")
    print(f"Wrote results/{args.version}/: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
