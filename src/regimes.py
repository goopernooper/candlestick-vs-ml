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
2020, 2022, other), because a few episodes dominate it.

Writes results/<version>/regime_days.csv, regime_metrics.csv,
regime_tests.csv, regime_mcs.csv and regime_episodes.csv.
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
            high = reg.index[reg == "high"]
            for name in [*EPISODES, "other"]:
                dates = [d for d in high if episode(d) == name]
                if dates:
                    episode_rows.append(metrics_on(preds, net, dates).assign(episode=name))
    return {"metrics": pd.concat(metric_rows, ignore_index=True),
            "tests": pd.DataFrame(test_rows),
            "mcs": pd.concat(mcs_rows, ignore_index=True),
            "episodes": pd.concat(episode_rows, ignore_index=True)}


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
                      ("episodes", "regime_episodes")]:
        out[key].to_csv(folder / f"{name}.csv", index=False)

    test_days = pd.Index(sorted(preds.index.unique()))
    reg = days["regime_spy"].reindex(test_days)
    print("\nTest days per SPY-volatility regime: "
          + ", ".join(f"{r} {int((reg == r).sum())}" for r in LABELS))
    met = out["metrics"][out["metrics"]["definition"] == "spy"]
    wide = met.pivot(index="model", columns="regime", values="mcc")[LABELS]
    t = out["tests"]
    c = t[(t["definition"] == "spy") & (t["loss"] == "classification")]
    bench = c.groupby("regime")["benchmark"].first()
    p = c.pivot(index="model", columns="regime", values="p_value")
    mc = out["mcs"]
    inside = mc[(mc["definition"] == "spy") & (mc["loss"] == "classification")].pivot(
        index="model", columns="regime", values="in_mcs")
    print(f"\nMCC by SPY-volatility regime (p: Diebold-Mariano against {bench.iloc[0]} on the classification "
          "loss; * = in the 90% Model Confidence Set):\n")
    print(f"  {'model':<15}" + "".join(f"{r:>30}" for r in LABELS))
    for model in wide.sort_values("high", ascending=False).index:
        cells = ""
        for r in LABELS:
            pv = p.loc[model, r] if model in p.index and r in p.columns else np.nan
            star = "*" if model in inside.index and bool(inside.loc[model, r]) else " "
            pv_text = f"p={pv:.2f}" if not np.isnan(pv) else "   -  "
            cells += f"{wide.loc[model, r]:>+14.4f}{star} {pv_text:>8}      "
        print(f"  {model:<15}{cells}")
    ep = out["episodes"].pivot(index="model", columns="episode", values="mcc")
    print("\nHigh-volatility days by episode, MCC:\n")
    print(ep[[e for e in [*EPISODES, "other"] if e in ep.columns]].sort_values("2008-09", ascending=False)
          .to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\nWrote results/{args.version}/regime_days.csv, regime_metrics.csv, regime_tests.csv, "
          "regime_mcs.csv, regime_episodes.csv (VIX regimes included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
