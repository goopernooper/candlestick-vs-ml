"""Break-even transaction cost: the cost per side at which a strategy stops making money.

Usage
-----
    python src/break_even.py v4

run.py also writes this for every new run.

The cost sweep (0, 1, 2, 5, 10 and 20 bps per side) is in backtest.csv. This
finds, for each model and strategy, the exact cost at which the compounded
annual net return reaches zero (ANALYSIS_PLAN.md section 7).

Each day's portfolio return is linear in the cost c (in bps per side):

    net_t(c) = gross_t - turnover_t * c / 10,000

where gross_t and turnover_t are averaged across that day's stocks. The
compounded return over the test period is zero when sum_t log(1 + net_t(c)) = 0.
That sum falls as c rises, so the break-even cost is its single root, found
numerically.

  break_even_bps    the cost per side at which the annual return is zero; NaN
                    if the strategy loses money even at zero cost ("never"),
                    inf if it never trades after its first position
                    (always_long: costs cannot make buy-and-hold lose money here)
  gross_ann_return  annual return at zero cost
  turnover_per_year average units traded per stock per year

Writes results/<version>/break_even.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import portfolio_returns, stock_returns  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_BPS = 10_000.0


def break_even(preds_one_model: pd.DataFrame, strategy: str) -> dict:
    """Break-even cost and context for one model's predictions."""
    daily = stock_returns(preds_one_model, strategy, cost_bps=0)
    gross = portfolio_returns(daily, "gross").to_numpy()
    turnover = portfolio_returns(daily, "turnover").to_numpy()
    days_per_year = load_config()["backtest"]["trading_days_per_year"]
    n = len(gross)

    def log_growth(c):
        net = gross - turnover * c / 10_000
        if (net <= -1).any():
            return -np.inf
        return np.log1p(net).sum()

    g0 = log_growth(0.0)
    out = {"days": n, "gross_ann_return": float(np.expm1(g0 * days_per_year / n)),
           "turnover_per_year": float(turnover.mean() * days_per_year)}
    if g0 <= 0:
        return {**out, "break_even_bps": np.nan}
    # turnover after the first day (the first position is a trade from cash)
    if turnover[1:].sum() == 0:
        return {**out, "break_even_bps": np.inf}
    hi = 1.0
    while log_growth(hi) > 0 and hi < MAX_BPS:
        hi *= 2
    if log_growth(hi) > 0:
        return {**out, "break_even_bps": np.inf}
    return {**out, "break_even_bps": float(brentq(log_growth, 0.0, hi, xtol=1e-6))}


def break_even_table(preds: pd.DataFrame, strategies=None) -> pd.DataFrame:
    strategies = load_config()["backtest"]["strategies"] if strategies is None else list(strategies)
    rows = [{"model": model, "strategy": s, **break_even(g, s)}
            for model, g in preds.groupby("model", sort=True) for s in strategies]
    return pd.DataFrame(rows)


def describe(bps: float) -> str:
    if np.isnan(bps):
        return "never profitable"
    if np.isinf(bps):
        return "no limit"
    return f"{bps:5.1f} bps"


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
    table = break_even_table(pd.read_parquet(path))
    table.to_csv(folder / "break_even.csv", index=False)
    wide = table.pivot(index="model", columns="strategy", values="break_even_bps")
    gross = table[table["strategy"] == "long_short"].set_index("model")["gross_ann_return"]
    turn = table[table["strategy"] == "long_short"].set_index("model")["turnover_per_year"]
    order = wide["long_short"].fillna(-1).sort_values(ascending=False).index
    print("Break-even cost per side (the cost at which the compounded annual net return is zero):\n")
    print(f"  {'model':<15} {'long/short':>16} {'long/flat':>16}   gross l/s return   turnover/yr")
    for m in order:
        print(f"  {m:<15} {describe(wide.loc[m, 'long_short']):>16} {describe(wide.loc[m, 'long_flat']):>16}"
              f"   {gross[m]:>+15.1%}   {turn[m]:>11.0f}")
    print(f"\nThe headline cost is {load_config()['backtest']['cost_bps_per_side']} bps per side; "
          "the sweep at 0-20 bps is in backtest.csv.")
    print(f"Wrote results/{args.version}/break_even.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
