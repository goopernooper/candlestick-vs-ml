"""How much of each strategy's return is just the market? Beta and alpha
against buy-and-hold.

Usage
-----
    python src/exposure.py v3          # from results/v3/predictions.parquet

run.py also does this for every new run and writes results/<version>/exposure.csv.

A long/short strategy that predicts "up" on most days is net long, so in a
rising market part of its return is simply market exposure, which buy-and-hold
earns for free. For each model and strategy, the daily portfolio return (net
of costs at the headline cost) is regressed on the market's daily return, the
equal-weight average next-day return of the 22 stocks (buy-and-hold, before costs):

    r_model = alpha + beta * r_market + e

  beta            market exposure: 1 = moves like buy-and-hold, 0 = unrelated
  alpha_ann       the return not explained by the market, per year (alpha x 252)
  alpha_t         alpha / its Newey-West standard error (lag floor(4 (T/100)^(2/9)),
                  as for the Diebold-Mariano tests in ANALYSIS_PLAN.md section 8)
  info_ratio      alpha / sd(e), annualized: the Sharpe ratio of the part of the
                  strategy that isn't market exposure
  net_exposure    average position across stocks and days (+1 = fully long,
                  0 = as much short as long)

A positive alpha_t above about 2 means the strategy earned more than its market
exposure explains. This is a descriptive check (DECISIONS.md D-025); the formal
comparisons between models are the tests in ANALYSIS_PLAN.md section 8.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import portfolio_returns, stock_returns  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def newey_west_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def alpha_beta(r, market, lags: int | None = None, days_per_year: int = 252) -> dict:
    """OLS of r on the market, with a Newey-West (Bartlett) standard error for alpha."""
    df = pd.concat([pd.Series(r, dtype=float), pd.Series(market, dtype=float)], axis=1).dropna()
    y, m = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    n = len(y)
    if n < 3:
        raise ValueError("fewer than 3 days where both series have a value (do their dates line up?)")
    X = np.column_stack([np.ones(n), m])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    e = y - X @ coef
    lags = newey_west_lag(n) if lags is None else lags
    u = X * e[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    sd_e = e.std(ddof=2)
    exact = sd_e <= 1e-12          # the strategy is the market (e.g. always long, no costs)
    return {"days": n, "beta": float(coef[1]), "alpha_ann": float(coef[0] * days_per_year),
            "alpha_t": np.nan if exact or se_alpha == 0 else float(coef[0] / se_alpha),
            "info_ratio": np.nan if exact else float(coef[0] / sd_e * np.sqrt(days_per_year)),
            "nw_lags": lags}


def market_returns(preds: pd.DataFrame) -> pd.Series:
    """Buy-and-hold: the equal-weight average next-day return across stocks, each day."""
    one = preds[preds["model"] == preds["model"].iloc[0]]
    one = one[one["fwd_return"].notna()]
    return one.groupby(level=0)["fwd_return"].mean().sort_index()


def exposure_table(preds: pd.DataFrame, strategies=None, cost_bps: float | None = None) -> pd.DataFrame:
    cfg = load_config()["backtest"]
    strategies = cfg["strategies"] if strategies is None else list(strategies)
    cost_bps = cfg["cost_bps_per_side"] if cost_bps is None else cost_bps
    market = market_returns(preds)
    rows = []
    for model, g in preds.groupby("model", sort=True):
        for strategy in strategies:
            daily = stock_returns(g, strategy, cost_bps)
            r = portfolio_returns(daily)
            rows.append({"model": model, "strategy": strategy, "cost_bps": cost_bps,
                         "net_exposure": float(daily["position"].mean()),
                         **alpha_beta(r, market, days_per_year=cfg["trading_days_per_year"])})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v3")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    path = folder / "predictions.parquet"
    if not path.exists():
        print(f"{path} not found (predictions aren't committed; they exist on the computer that ran "
              f"`python src/run.py --version {args.version}`).")
        return 1
    table = exposure_table(pd.read_parquet(path))
    table.to_csv(folder / "exposure.csv", index=False)
    ls = table[table["strategy"] == "long_short"].sort_values("alpha_t", ascending=False)
    cols = ["model", "net_exposure", "beta", "alpha_ann", "alpha_t", "info_ratio"]
    print(f"Long/short at {table['cost_bps'].iloc[0]:g} bps per side, regressed on buy-and-hold:\n")
    print(ls[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {(folder / 'exposure.csv').relative_to(ROOT)} (both strategies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
