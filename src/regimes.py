"""Volatility regimes: do the results change between calm and turbulent markets?

Usage
-----
    python src/regimes.py v4            # 10,000 bootstrap replications for the MCS
    python src/regimes.py v4 --reps 1000

Implements ANALYSIS_PLAN.md section 11 (D-014).

Regime of day t (the day a prediction is made, at its close):
  spy   SPY's realized volatility over the 21 trading days ending on day t
        (standard deviation of daily close-to-close returns, x sqrt(252)),
        sorted into low / medium / high by terciles.
  vix   the VIX close on day t, sorted the same way (robustness check).
The tercile cutoffs for day t use only values through day t - 1, on an
expanding window from 2002-01-02, so the regime is something a trader could
have known at the time. A day is left unassigned until 252 earlier values
exist (only in 2002-2003, before the first test year).

Within each regime, every model is re-evaluated with the same metrics and tests
as the main results:
  metrics   accuracy, balanced accuracy and MCC on the regime's labelled
            stock-days; long/short Sharpe ratio and mean annual return at
            5 bps on the regime's days (positions carry across regimes, so
            each day's cost is the one actually paid that day)
  tests     Diebold-Mariano against the strongest baseline over all test days
            (highest pooled MCC: reversal in practice) on the classification
            loss, and against always_long on the economic loss; the 90% Model
            Confidence Set on both losses (significance.py, losses.py). The
            benchmark is the same in every regime, so regimes compare like
            with like; picking each regime's best baseline would also let a
            lucky coin flip become the benchmark.
The high-volatility SPY regime is also broken down by episode (2008-09, 2011,
2020, 2022, other): each episode's share of the high-volatility days, every
model's metrics on that episode alone (and its MCC minus the benchmark's on the
same days), and the high-volatility results with each episode left out. A
result that moves a lot when one episode is dropped rests on that episode.

Writes results/<version>/regime_days.csv, regime_metrics.csv,
regime_tests.csv, regime_mcs.csv, regime_episodes.csv,
regime_episodes_leave_one_out.csv and regime_episodes.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import classification_metrics, portfolio_returns, stock_returns  # noqa: E402
from features import load_config  # noqa: E402
from losses import classification_losses, economic_losses  # noqa: E402
from significance import dm_test, mcs, strongest_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 21
MIN_HISTORY = 252
LABELS = ["low", "medium", "high"]
EPISODES = {"2008-09": (2008, 2009), "2011": (2011, 2011), "2020": (2020, 2020), "2022": (2022, 2022)}


def realized_volatility(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """Annualized standard deviation of the daily returns ending on each day."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def expanding_terciles(x: pd.Series, start=None, min_history: int = MIN_HISTORY) -> pd.Series:
    """low / medium / high for each day, against the 1/3 and 2/3 quantiles of
    the values strictly before that day (from `start` on)."""
    x = x.dropna()
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    values = x.to_numpy()
    out = np.full(len(values), None, dtype=object)
    for i in range(min_history, len(values)):
        lo, hi = np.quantile(values[:i], [1 / 3, 2 / 3])
        out[i] = "low" if values[i] <= lo else ("high" if values[i] > hi else "medium")
    return pd.Series(out, index=x.index, name="regime")


def regime_days(spy: pd.DataFrame, vix: pd.DataFrame, start=None) -> pd.DataFrame:
    """One row per day: SPY volatility, VIX, and the regime under each definition."""
    start = load_config()["splits"]["train_start"] if start is None else start
    vol = realized_volatility(spy["Close"])
    level = vix["Close"]
    days = pd.DataFrame({"spy_vol": vol, "vix": level})
    days["regime_spy"] = expanding_terciles(vol, start)
    days["regime_vix"] = expanding_terciles(level, start)
    days.index.name = "date"
    return days[days.index >= pd.Timestamp(start)]


def episode(date: pd.Timestamp) -> str:
    for name, (y0, y1) in EPISODES.items():
        if y0 <= date.year <= y1:
            return name
    return "other"


def metrics_on(preds: pd.DataFrame, net: dict, dates) -> pd.DataFrame:
    """Per model: classification metrics on `dates`, and Sharpe / mean annual
    return of the long/short net returns on those dates."""
    days_per_year = load_config()["backtest"]["trading_days_per_year"]
    dates = pd.Index(dates)
    rows = []
    for model, g in preds[preds.index.isin(dates)].groupby("model", sort=True):
        m = classification_metrics(g["up"], g["pred"])
        r = net[model].reindex(dates).dropna()
        sd = r.std(ddof=1)
        rows.append({"model": model, "days": int(r.size), "stock_days": m["days"],
                     "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                     "mcc": m["mcc"], "predicted_up_share": m["predicted_up_share"],
                     "ls_ann_mean_return": float(r.mean() * days_per_year),
                     "ls_sharpe": float(r.mean() / sd * np.sqrt(days_per_year)) if sd > 1e-12 else np.nan})
    return pd.DataFrame(rows)


def analyse(preds: pd.DataFrame, days: pd.DataFrame, reps: int | None = None) -> dict:
    cfg = load_config()
    sig, seed = cfg["significance"], cfg["seed"]
    reps = sig["reps"] if reps is None else reps
    cost = cfg["backtest"]["cost_bps_per_side"]
    net = {m: portfolio_returns(stock_returns(g, "long_short", cost)) for m, g in preds.groupby("model")}
    closs, eloss = classification_losses(preds), economic_losses(preds, cost_bps=cost)
    test_days = pd.Index(sorted(preds.index.unique()))
    econ = sig["economic_benchmark"]
    overall = metrics_on(preds, net, test_days).set_index("model")
    strongest = strongest_baseline(overall, sig["baselines"])

    metric_rows, test_rows, mcs_rows, episode_rows = [], [], [], []
    for definition in ["spy", "vix"]:
        reg = days[f"regime_{definition}"].reindex(test_days)
        for regime in LABELS:
            dates = reg.index[reg == regime]
            if len(dates) == 0:
                continue
            met = metrics_on(preds, net, dates).assign(definition=definition, regime=regime)
            metric_rows.append(met)
            for loss, table, bench in [("classification", closs, strongest), ("economic", eloss, econ)]:
                sub = table.reindex(dates)
                for model in sub.columns:
                    if model == bench:
                        continue
                    pair = sub[[model, bench]].dropna()
                    test_rows.append({"definition": definition, "regime": regime, "loss": loss,
                                      "model": model, "benchmark": bench,
                                      **dm_test(pair[model] - pair[bench])})
                m = mcs(sub, sig["mcs_size"], reps, sig["block_size"], seed)
                mcs_rows.append(m.rename_axis("model").reset_index().assign(
                    definition=definition, regime=regime, loss=loss))
        if definition == "spy":
            ep, loo = episode_breakdown(preds, net, reg.index[reg == "high"], strongest)
    return {"metrics": pd.concat(metric_rows, ignore_index=True),
            "tests": pd.DataFrame(test_rows),
            "mcs": pd.concat(mcs_rows, ignore_index=True),
            "episodes": ep, "leave_one_out": loo}


def episode_breakdown(preds: pd.DataFrame, net: dict, high, benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much do a few events drive the high-volatility results?

    episodes       metrics on each episode's high-volatility days, with the
                   episode's share of all high-volatility days and each model's
                   MCC minus the benchmark's on the same days
    leave_one_out  metrics on all high-volatility days except one episode's: a
                   result that changes a lot when one episode is dropped rests
                   on that episode
    """
    high = pd.Index(high)
    names = pd.Series([episode(d) for d in high], index=high)
    rows, loo = [], []
    for name in [*EPISODES, "other"]:
        dates = names.index[names == name]
        if len(dates) == 0:
            continue
        m = metrics_on(preds, net, dates)
        bench = m.set_index("model").loc[benchmark, "mcc"] if benchmark in set(m["model"]) else np.nan
        rows.append(m.assign(episode=name, episode_days=len(dates), share_of_high_days=len(dates) / len(high),
                             mcc_minus_benchmark=m["mcc"] - bench, benchmark=benchmark))
        if name != "other":
            loo.append(metrics_on(preds, net, names.index[names != name]).assign(
                dropped=name, days_left=int((names != name).sum())))
    loo.append(metrics_on(preds, net, high).assign(dropped="none", days_left=len(high)))
    return pd.concat(rows, ignore_index=True), pd.concat(loo, ignore_index=True)


def plot_episodes(ep: pd.DataFrame, loo: pd.DataFrame, path: Path, benchmark: str) -> None:
    """Left: MCC on each episode's high-volatility days. Right: MCC on all
    high-volatility days with one episode left out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from coverage import GRID, HIGHLIGHT, INK, MUTED, SURFACE
    shown = [m for m in HIGHLIGHT if m in set(ep["model"])]
    order = [e for e in [*EPISODES, "other"] if e in set(ep["episode"])]
    days = ep.drop_duplicates("episode").set_index("episode")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor=SURFACE, sharey=True)
    for ax, frame, col, cats, labels, xlabel in [
        (axes[0], ep, "episode", order,
         [f"{e}\n{int(days.loc[e, 'episode_days'])} days ({days.loc[e, 'share_of_high_days']:.0%})" for e in order],
         "Episode (high-volatility days in it, share of all high-volatility days)"),
        (axes[1], loo, "dropped", ["none", *[e for e in order if e != "other"]],
         ["all high-vol\ndays", *[f"without\n{e}" for e in order if e != "other"]],
         "High-volatility days with one episode left out")]:
        ax.set_facecolor(SURFACE)
        width = 0.12
        for i, model in enumerate(shown):
            color, marker = HIGHLIGHT[model]
            g = frame[frame["model"] == model].set_index(col).reindex(cats)
            x = np.arange(len(cats)) + (i - (len(shown) - 1) / 2) * width
            ax.plot(x, g["mcc"], linestyle="none", marker=marker, ms=7, color=color, mec=SURFACE, mew=1.2,
                    label=model, zorder=3)
        ax.axhline(0, color=MUTED, lw=1, zorder=2)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("MCC", color=INK)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("High-volatility results by episode", x=0.01, ha="left", fontsize=11.5)
    fig.text(0.01, 0.005, f"High volatility: SPY's 21-day volatility in the top tercile of all earlier days. "
             f"Benchmark for comparisons: {benchmark}.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    parser.add_argument("--reps", type=int, help="bootstrap replications (default: config.yaml)")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    path = folder / "predictions.parquet"
    if not path.exists():
        print(f"{path} not found (predictions aren't committed; they exist on the computer that ran "
              f"`python src/run.py --version {args.version}`).")
        return 1
    import data
    import manifest
    if not manifest.check(quiet=True):
        print("Stopping: data/raw does not match data/manifest.json.")
        return 1
    preds = pd.read_parquet(path)
    days = regime_days(data.load_raw("SPY"), data.load_raw("^VIX"))
    days.to_csv(folder / "regime_days.csv")
    print(f"Evaluating {preds['model'].nunique()} models in each regime "
          f"({args.reps or load_config()['significance']['reps']:,} replications per Model Confidence Set)...",
          flush=True)
    out = analyse(preds, days, reps=args.reps)
    for key, name in [("metrics", "regime_metrics"), ("tests", "regime_tests"), ("mcs", "regime_mcs"),
                      ("episodes", "regime_episodes"), ("leave_one_out", "regime_episodes_leave_one_out")]:
        out[key].to_csv(folder / f"{name}.csv", index=False)

    test_days = pd.Index(sorted(preds.index.unique()))
    reg = days["regime_spy"].reindex(test_days)
    print("\nTest days per SPY-volatility regime: "
          + ", ".join(f"{r} {int((reg == r).sum())}" for r in LABELS))
    met = out["metrics"][out["metrics"]["definition"] == "spy"]
    t = out["tests"]
    alpha = load_config()["significance"]["alpha"]
    mc = out["mcs"]

    def verdict(row) -> str:
        if row is None or np.isnan(row["p_value"]):
            return f"{'(benchmark)':<17}"
        mark = ("better" if row["dm_stat"] < 0 else "worse") if row["p_value"] < alpha else ""
        return f"{'DM ' + format(row['dm_stat'], '+6.2f') + ' ' + mark:<17}"

    for loss, label, values in [("classification", "MCC", "mcc"), ("economic", "long/short Sharpe", "ls_sharpe")]:
        tl = t[(t["definition"] == "spy") & (t["loss"] == loss)].set_index(["model", "regime"])
        bench = tl["benchmark"].iloc[0]
        inside = mc[(mc["definition"] == "spy") & (mc["loss"] == loss)].set_index(["model", "regime"])["in_mcs"]
        wide = met.pivot(index="model", columns="regime", values=values)[LABELS]
        print(f"\n{label} by SPY-volatility regime. DM: Diebold-Mariano statistic against {bench} on the "
              f"{loss} loss,\nnegative = lower loss than {bench}; 'better' / 'worse' = significant at "
              f"{alpha:.0%}; * = in the 90% Model Confidence Set.\n")
        print(f"  {'model':<15}" + "".join(f"{r:<28}" for r in LABELS))
        for model in wide.sort_values("high", ascending=False).index:
            cells = ""
            for r in LABELS:
                row = tl.loc[(model, r)] if (model, r) in tl.index else None
                star = "*" if bool(inside.get((model, r), False)) else " "
                cells += f"{wide.loc[model, r]:>+8.4f}{star} {verdict(row)} "
            print(f"  {model:<15}{cells}")
    ep_all, loo = out["episodes"], out["leave_one_out"]
    order = [e for e in [*EPISODES, "other"] if e in set(ep_all["episode"])]
    counts = ep_all.drop_duplicates("episode").set_index("episode").loc[order]
    print("\nHigh-volatility days by episode: "
          + ", ".join(f"{e} {int(r.episode_days)} ({r.share_of_high_days:.0%})" for e, r in counts.iterrows()))
    ep = ep_all.pivot(index="model", columns="episode", values="mcc")[order]
    print("\nMCC on each episode's high-volatility days:\n")
    print(ep.sort_values(order[0], ascending=False).to_string(float_format=lambda x: f"{x:+.4f}"))
    lw = loo.pivot(index="model", columns="dropped", values="mcc")
    lw = lw[["none", *[e for e in order if e != "other"]]].rename(
        columns={"none": "all high", **{e: f"w/o {e}" for e in order}})
    print("\nMCC on all high-volatility days, and with one episode left out:\n")
    print(lw.sort_values("all high", ascending=False).to_string(float_format=lambda x: f"{x:+.4f}"))
    plot_episodes(ep_all, loo, folder / "regime_episodes.png", ep_all["benchmark"].iloc[0])
    print(f"\nWrote results/{args.version}/regime_days.csv, regime_metrics.csv, regime_tests.csv, "
          "regime_mcs.csv, regime_episodes.csv, regime_episodes_leave_one_out.csv, regime_episodes.png "
          "(VIX regimes included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
