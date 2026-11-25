"""Robustness: trade at the next day's open instead of today's close.

Usage
-----
    python src/robustness_open.py v4

The main backtest enters each position at the close of day t, the moment the
signal is computed, which a real trader could only approximate. Here the same
predictions (results/<version>/predictions.parquet, nothing is retrained) are
traded at the next open instead: a position decided at the close of day t is
entered at the open of day t + 1 and held to the close of day t + 1, so the
return is Close(t+1) / Open(t+1) - 1. The overnight move from Close(t) to
Open(t+1) is missed. Costs, strategies and the cost grid are as in the main
backtest (ANALYSIS_PLAN.md section 12, item 2).

Writes results/<version>/robustness_next_open_backtest.csv (every model,
strategy and cost), robustness_next_open_break_even.csv and
robustness_next_open_exposure.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from break_even import break_even_table  # noqa: E402
from evaluate import backtest  # noqa: E402
from exposure import exposure_table  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def next_open_returns(prices: pd.DataFrame) -> pd.Series:
    """For each day t: Close(t+1) / Open(t+1) - 1."""
    return (prices["Close"] / prices["Open"] - 1).shift(-1)


def with_next_open(preds: pd.DataFrame, loader) -> pd.DataFrame:
    """The predictions with fwd_return replaced by the next day's open-to-close return."""
    lookup = pd.concat({t: next_open_returns(loader(t)) for t in preds["ticker"].unique()},
                       names=["ticker", "date"])
    keys = pd.MultiIndex.from_arrays([preds["ticker"], preds.index], names=["ticker", "date"])
    return preds.assign(fwd_return=lookup.reindex(keys).to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
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
    moved = with_next_open(preds, data.load_raw)
    missing = int(moved["fwd_return"].isna().sum() - preds["fwd_return"].isna().sum())
    bt = backtest(moved)
    bt.to_csv(folder / "robustness_next_open_backtest.csv", index=False)
    be = break_even_table(moved)
    be.to_csv(folder / "robustness_next_open_break_even.csv", index=False)
    cost = load_config()["backtest"]["cost_bps_per_side"]
    ex = exposure_table(moved, cost_bps=cost)
    ex.to_csv(folder / "robustness_next_open_exposure.csv", index=False)

    main_bt = pd.read_csv(folder / "backtest.csv")
    key = ["model", "strategy", "cost_bps"]
    both = main_bt.merge(bt, on=key, suffixes=("_close", "_open"))
    ls = both[(both["strategy"] == "long_short") & (both["cost_bps"] == cost)].set_index("model")
    be_ls = be[be["strategy"] == "long_short"].set_index("model")["break_even_bps"]
    alpha = ex[ex["strategy"] == "long_short"].set_index("model")["alpha_t"]
    print(f"Long/short at {cost} bps per side: entering at today's close (main) vs the next day's open"
          + (f" ({missing} stock-days without a next open)" if missing else "") + ":\n")
    print(f"  {'model':<15}{'Sharpe close':>13}{'Sharpe open':>13}{'return close':>14}{'return open':>13}"
          f"{'break-even open':>17}{'alpha t open':>14}")
    for m in ls.sort_values("sharpe_close", ascending=False).index:
        r = ls.loc[m]
        b = be_ls.get(m)
        b_text = "no limit" if b == float("inf") else ("never" if pd.isna(b) else f"{b:.1f} bps")
        print(f"  {m:<15}{r['sharpe_close']:>13.2f}{r['sharpe_open']:>13.2f}{r['ann_return_close']:>14.1%}"
              f"{r['ann_return_open']:>13.1%}{b_text:>17}{alpha.get(m, float('nan')):>14.2f}")
    print(f"\nWrote results/{args.version}/robustness_next_open_backtest.csv, "
          "robustness_next_open_break_even.csv, robustness_next_open_exposure.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
