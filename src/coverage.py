"""Coverage curves: is any model better on the days it is most confident about?

Usage
-----
    python src/coverage.py v4

Implements ANALYSIS_PLAN.md section 10. A model's confidence on a stock-day is
|P(up) - 0.5|. A trader could act only on the days a model is most sure of.
If a model's edge is concentrated there, its accuracy and MCC should rise as
coverage (the share of stock-days it acts on) falls. The synthetic checks of
paper Section 4.3 showed why this matters: a pattern that is right 70% of the
time on 1% of days is invisible in the pooled MCC.

The cutoffs must not be chosen on the test year, so for each fold:

  1. Validation predictions. The model, with that fold's settings, is fitted on
     the training window minus its last year and predicts that last year (the
     validation year, the same split as tuning, splits.Fold.tuning). These are
     out-of-sample predictions the model could have made at the time.
  2. Cutoffs. For each target coverage (100%, 50%, 25%, 10%), the cutoff is the
     confidence that the top target share of validation stock-days reach.
  3. Test. The cutoffs are applied unchanged to the test-year predictions of
     the main run (results/<version>/predictions.parquet): the model acts on
     every test stock-day whose confidence is at or above the cutoff.

Coverage actually achieved differs from the target when the confidence
distribution shifts between years, and when many days share one confidence
value (the pattern miner and k-NN). Models that give every day the same
confidence (majority, always_long, previous_day, reversal) cannot rank days
and stay at 100% coverage.

For each model and target, pooled over the 18 test years: the coverage
achieved, the stock-days acted on, accuracy, balanced accuracy and MCC on the
labelled ones, and the share predicted up.

The deciding comparison. A model's confident days may simply be the days on
which short-term reversal works best (after big moves, in volatile markets).
So for each model and target, the reversal baseline is scored on exactly the
stock-days the model acts on (benchmark_mcc), and the difference in MCC gets a
95% stationary-bootstrap interval over dates (mean block 20 days, as in
bootstrap_ci.py). An interval above 0 means the model's candles add something
beyond reversal on the days it chose.

Writes results/<version>/coverage.csv and coverage.png, coverage_vs_reversal.csv
and coverage_vs_reversal.png (the comparison), and keeps the
validation predictions in results/<version>/predictions_validation.parquet
(not committed, like predictions.parquet; delete it to recompute).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import classification_metrics  # noqa: E402
from features import load_config  # noqa: E402
from splits import Fold, walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [1.0, 0.5, 0.25, 0.10]
HIGHLIGHT = {   # model -> (color, marker): the paper figure's labelled lines
    "lightgbm_b": ("#2a78d6", "o"), "svm_b": ("#eb6834", "s"), "rf_b": ("#1baf7a", "D"),
    "pattern_miner": ("#eda100", "^"), "knn_b3": ("#e87ba4", "v"), "reversal": ("#008300", "P"),
}
BENCHMARK = "reversal"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def validation_folds(folds) -> list[Fold]:
    """For each fold, a Fold that trains on its fit set and tests on its
    validation year, but keeps the outer test year as its name, so each model
    is built with that fold's settings and seed."""
    out = []
    for f in folds:
        inner = f.tuning()
        out.append(Fold(test_year=f.test_year, train=inner.train, purged=inner.purged,
                        test=inner.test, horizon=f.horizon))
    return out


def confidence(p) -> np.ndarray:
    return np.abs(np.asarray(p, dtype=float) - 0.5)


def cutoff(val_conf: np.ndarray, target: float) -> float:
    """Confidence that the top `target` share of validation days reach.

    A model that gives every validation day the same confidence cannot rank
    days at all (majority, always_long, previous_day, reversal), so it acts on
    every day whatever the target."""
    val_conf = np.asarray(val_conf, dtype=float)
    if target >= 1 or np.ptp(val_conf) == 0:
        return -np.inf
    return float(np.quantile(val_conf, 1 - target))


def selections(val: pd.DataFrame, test: pd.DataFrame, targets=TARGETS) -> dict:
    """{(model, target): the test rows the model acts on}, cutoffs from validation years."""
    out = {}
    for model, t in test.groupby("model", sort=True):
        v = val[val["model"] == model]
        for target in targets:
            picked = []
            for year, ty in t.groupby("test_year"):
                vy = v[v["test_year"] == year]
                if vy.empty:
                    raise ValueError(f"{model}: no validation predictions for the {year} fold")
                c = cutoff(confidence(vy["p_up"]), target)
                picked.append(ty[confidence(ty["p_up"]) >= c])
            out[(model, target)] = pd.concat(picked)
    return out


def same_days(selected: pd.DataFrame, test: pd.DataFrame, other: str) -> pd.DataFrame:
    """Model `other`'s test predictions on the same stock-days as `selected`."""
    o = test[test["model"] == other]
    o = o.set_index([o.index, "ticker"])
    keys = pd.MultiIndex.from_arrays([selected.index, selected["ticker"]])
    picked = o.reindex(keys)
    if picked["pred"].isna().any():
        raise ValueError(f"{other} has no prediction on some selected stock-days")
    return picked.reset_index(level="ticker")


def coverage_table(val: pd.DataFrame, test: pd.DataFrame, targets=TARGETS,
                   benchmark: str = BENCHMARK) -> pd.DataFrame:
    """val and test: prediction tables (model, test_year, p_up, pred, up; test
    also indexed by date with a ticker column, for the benchmark comparison).

    With the benchmark's predictions present, adds benchmark_mcc: the
    benchmark's MCC on exactly the stock-days the model acts on, and
    mcc_minus_benchmark: whether the model beats it on those days."""
    sel = selections(val, test, targets)
    has_bench = benchmark in set(test["model"]) and "ticker" in test.columns
    rows = []
    for (model, target), s in sel.items():
        t_all = (test["model"] == model).sum()
        m = classification_metrics(s["up"], s["pred"]) if len(s) else {}
        row = {"model": model, "target_coverage": target,
               "coverage": len(s) / t_all, "stock_days": len(s),
               "labelled_days": m.get("days", 0), "accuracy": m.get("accuracy", np.nan),
               "balanced_accuracy": m.get("balanced_accuracy", np.nan),
               "mcc": m.get("mcc", np.nan), "predicted_up_share": m.get("predicted_up_share", np.nan)}
        if has_bench and len(s):
            b = same_days(s, test, benchmark)
            bm = classification_metrics(b["up"], b["pred"])["mcc"]
            row.update({"benchmark": benchmark, "benchmark_mcc": bm, "mcc_minus_benchmark": row["mcc"] - bm})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_vs_benchmark(val: pd.DataFrame, test: pd.DataFrame, targets=TARGETS,
                           benchmark: str = BENCHMARK, reps: int | None = None,
                           seed: int | None = None, chunk: int = 250) -> pd.DataFrame:
    """95% stationary-bootstrap intervals (whole dates, mean block 20 days, as
    in bootstrap_ci.py) for each model's MCC on its selected days, the
    benchmark's MCC on the same days, and their difference. Every model and
    target uses the same resampled dates in a replication."""
    from bootstrap_ci import metrics_from_counts, stationary_indices
    cfg = load_config()
    reps = cfg["significance"]["reps"] if reps is None else reps
    block = cfg["significance"]["block_size"]
    seed = cfg["seed"] if seed is None else seed
    dates = pd.Index(sorted(test.index.unique()))
    pos = pd.Series(np.arange(len(dates)), index=dates)

    def counts(frame):
        lab = frame[frame["up"].notna()]
        y, p = lab["up"].astype(int).to_numpy(), lab["pred"].astype(int).to_numpy()
        cells = np.stack([(p == 1) & (y == 1), (p == 1) & (y == 0), (p == 0) & (y == 0),
                          (p == 0) & (y == 1)], axis=1).astype(float)
        out = np.zeros((len(dates), 4))
        np.add.at(out, pos.reindex(lab.index).to_numpy(), cells)
        return out

    keys, blocks = [], []
    for (model, target), s in selections(val, test, targets).items():
        if model == benchmark or not len(s):
            continue
        keys.append((model, target))
        blocks += [counts(s), counts(same_days(s, test, benchmark))]
    C = np.concatenate(blocks, axis=1)                       # T x (keys x 2 x 4)
    k = len(keys)
    point = metrics_from_counts(C.sum(axis=0).reshape(k, 2, 4))["mcc"]
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        idx = stationary_indices(len(dates), r, block, rng)
        w = np.zeros((r, len(dates)))
        np.add.at(w, (np.repeat(np.arange(r), len(dates)), idx.ravel()), 1.0)
        draws.append(metrics_from_counts((w @ C).reshape(r, k, 2, 4))["mcc"])
    d = np.concatenate(draws)                                 # reps x keys x 2
    rows = []
    for j, (model, target) in enumerate(keys):
        diff = d[:, j, 0] - d[:, j, 1]
        rows.append({"model": model, "target_coverage": target, "mcc": point[j, 0],
                     "mcc_low": np.percentile(d[:, j, 0], 2.5), "mcc_high": np.percentile(d[:, j, 0], 97.5),
                     "benchmark": benchmark, "benchmark_mcc": point[j, 1],
                     "diff": point[j, 0] - point[j, 1],
                     "diff_low": np.percentile(diff, 2.5), "diff_high": np.percentile(diff, 97.5),
                     "share_of_draws_diff_above_0": float((diff > 0).mean()), "reps": reps})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, metric, label in [(axes[0], "accuracy", "Accuracy"), (axes[1], "mcc", "MCC")]:
        ax.set_facecolor(SURFACE)
        for model, g in table.groupby("model"):
            g = g.sort_values("coverage")
            if model in HIGHLIGHT:
                continue
            ax.plot(g["coverage"] * 100, g[metric], color="#b8b7b1", lw=1, zorder=1)
        ends = []
        for model, (color, marker) in HIGHLIGHT.items():
            g = table[table["model"] == model].sort_values("coverage")
            if g.empty:
                continue
            ax.plot(g["coverage"] * 100, g[metric], color=color, lw=2, marker=marker, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=model)
            ends.append((g.iloc[0][metric], g.iloc[0]["coverage"] * 100, model))
        # label each line at its lowest-coverage end, where the lines spread apart,
        # nudged vertically so labels never overlap
        lo, hi = ax.get_ylim()
        gap = 0.055 * (hi - lo)
        placed = []
        for y, x, model in sorted(ends):
            y_label = max(y, placed[-1] + gap) if placed else y
            placed.append(y_label)
            ax.annotate(model, (x, y), xytext=(x / 1.12, y_label), textcoords="data", ha="right",
                        va="center", fontsize=8.5, color=INK,
                        arrowprops=None if abs(y_label - y) < gap / 3 else
                        {"arrowstyle": "-", "color": MUTED, "lw": 0.6})
        if metric == "mcc":
            ax.axhline(0, color=MUTED, lw=1, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100])
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.set_xlim(3.5, 130)
        ax.set_xlabel("Coverage achieved (share of stock-days acted on)")
        ax.set_ylabel(label)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.18, 1))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.005, "Grey: the other models. Cutoffs chosen on each fold's validation year, "
             "applied unchanged to its test year.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_vs_benchmark(boot: pd.DataFrame, path: Path, title: str) -> None:
    """MCC minus the benchmark's MCC on the same stock-days, with 95% intervals,
    against target coverage, for the highlighted models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [m for m in HIGHLIGHT if m in set(boot["model"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    targets = sorted(boot["target_coverage"].unique())
    width = 0.12
    for i, model in enumerate(shown):
        color, marker = HIGHLIGHT[model]
        g = boot[boot["model"] == model].set_index("target_coverage").loc[targets]
        x = np.arange(len(targets)) + (i - (len(shown) - 1) / 2) * width
        ax.errorbar(x, g["diff"], yerr=[g["diff"] - g["diff_low"], g["diff_high"] - g["diff"]],
                    fmt=marker, ms=7, color=color, mec=SURFACE, mew=1.2, elinewidth=1.6, capsize=0,
                    label=model, zorder=3)
    ax.axhline(0, color=MUTED, lw=1, zorder=2)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"{t:.0%}" for t in targets])
    ax.set_xlabel("Target coverage (share of stock-days acted on)")
    ax.set_ylabel(f"MCC minus {BENCHMARK}'s MCC on the same days")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.005, "Bars: 95% stationary-bootstrap intervals over dates (mean block 20 days). "
             "Above 0: better than the benchmark on the days the model chose.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def validation_predictions(version: str, models) -> pd.DataFrame:
    """Fit every model on each fold's fit set and predict its validation year,
    with the settings the main run used."""
    import manifest
    from dataset import build_table
    from run import experiments
    from tune import tuned_settings, with_settings
    if not manifest.check(quiet=True):
        raise SystemExit("Stopping: data/raw does not match data/manifest.json.")
    info = json.loads((ROOT / "results" / version / "run_info.json").read_text())
    specs = {name: sp for name, sp in experiments().items() if name in models}
    feature_sets = sorted({fs for sp in specs.values() for fs in sp["feature_sets"]})
    table = build_table(feature_sets=feature_sets)
    folds = walk_forward(table.index)
    for name in info.get("tuned", []):
        if name in specs:
            settings = tuned_settings(table, folds, name, specs[name], log=lambda _: None)
            specs[name] = with_settings(specs[name], settings)
    from run import run_experiments
    return run_experiments(table, validation_folds(folds), specs,
                           progress=lambda f: print(f.test_year, end=" ", flush=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    test_path, val_path = folder / "predictions.parquet", folder / "predictions_validation.parquet"
    if not test_path.exists():
        print(f"{test_path} not found (predictions aren't committed; they exist on the computer that ran "
              f"`python src/run.py --version {args.version}`).")
        return 1
    test = pd.read_parquet(test_path)
    models = sorted(test["model"].unique())
    if val_path.exists():
        val = pd.read_parquet(val_path)
        print(f"Using saved validation predictions ({val_path.name}).")
    else:
        print(f"Fitting {len(models)} models on each fold's fit set to predict its validation year "
              "(about as long as the main run's 18 folds, with cached tuning):", flush=True)
        val = validation_predictions(args.version, models)
        val.to_parquet(val_path)
        print()
    table = coverage_table(val, test)
    table.to_csv(folder / "coverage.csv", index=False)
    plot(table, folder / "coverage.png", f"Coverage curves, results {args.version}, test years "
         f"{int(test['test_year'].min())}-{int(test['test_year'].max())}")

    print(f"Bootstrapping MCC against {BENCHMARK} on the same stock-days "
          f"({load_config()['significance']['reps']:,} replications)...", flush=True)
    boot = bootstrap_vs_benchmark(val, test)
    boot.to_csv(folder / f"coverage_vs_{BENCHMARK}.csv", index=False)
    plot_vs_benchmark(boot, folder / f"coverage_vs_{BENCHMARK}.png",
                      f"Confident days: each model against {BENCHMARK} on the same stock-days, results {args.version}")

    wide = table.pivot(index="model", columns="target_coverage", values=["coverage", "mcc", "accuracy"])
    order = wide[("mcc", 1.0)].sort_values(ascending=False).index
    print("\nMCC (and coverage achieved) by target coverage; cutoffs set on validation years:\n")
    print(f"  {'model':<15}" + "".join(f"{f'{t:.0%} target':>22}" for t in TARGETS))
    for m in order:
        cells = "".join(f"{wide.loc[m, ('mcc', t)]:>+12.4f} ({wide.loc[m, ('coverage', t)]:>5.1%})"
                        for t in TARGETS)
        print(f"  {m:<15}{cells}")
    b10 = boot[boot["target_coverage"] == min(TARGETS)].sort_values("diff", ascending=False)
    print(f"\nOn the {min(TARGETS):.0%} most confident days: the model's MCC, {BENCHMARK}'s MCC on the same "
          "stock-days, and the difference with its 95% interval:\n")
    print(f"  {'model':<15}{'model MCC':>11}{BENCHMARK + ' MCC':>15}{'difference':>13}   95% interval")
    for r in b10.itertuples():
        flag = "  above 0" if r.diff_low > 0 else ("  below 0" if r.diff_high < 0 else "")
        print(f"  {r.model:<15}{r.mcc:>+11.4f}{r.benchmark_mcc:>+15.4f}{r.diff:>+13.4f}   "
              f"[{r.diff_low:+.4f}, {r.diff_high:+.4f}]{flag}")
    print(f"\nWrote results/{args.version}/coverage.csv, coverage.png, coverage_vs_{BENCHMARK}.csv "
          f"and coverage_vs_{BENCHMARK}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
