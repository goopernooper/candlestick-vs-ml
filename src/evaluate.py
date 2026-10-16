"""Scoring predictions: classification metrics now, the trading backtest next.

Every evaluation starts from a predictions table with one row per stock per
test day and these columns:

    date (index)  the day the prediction is made, at that day's close
    ticker        which stock
    test_year     which walk-forward fold the day belongs to
    model         which model made the prediction
    p_up          the model's probability that the next return is up
    pred          1 (up) or 0 (down)
    up            what actually happened: 1, 0, or <NA> in the dead zone
    fwd_return    the actual next-day return (used by the backtest)

Classification metrics skip dead-zone days (up is <NA>); the backtest will not.

The metrics:
  accuracy           share of days called correctly
  balanced_accuracy  average of the hit rate on up days and on down days, so
                     always guessing "up" scores 0.5, not the share of up days
  mcc                Matthews correlation coefficient, from -1 (always wrong)
                     through 0 (no better than chance or a constant guess) to +1
                     (always right). The headline classification metric, because
                     it can't be inflated by guessing the more common direction
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import check_not_holdout  # noqa: E402

METRIC_COLUMNS = ["days", "dead_zone_days", "accuracy", "balanced_accuracy", "mcc",
                  "actual_up_share", "predicted_up_share"]


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    """(true positives, false positives, true negatives, false negatives), with up = positive."""
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    return tp, fp, tn, fn


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, balanced accuracy and MCC for days with a label (dead-zone days removed first)."""
    y_true = pd.array(y_true, dtype="Int8")
    y_pred = np.asarray(y_pred)
    labelled = ~np.asarray(y_true.isna())
    tp, fp, tn, fn = confusion(y_true[labelled], y_pred[labelled])
    n = tp + fp + tn + fn
    if n == 0:
        return {"days": 0, "dead_zone_days": int((~labelled).sum()), "accuracy": np.nan,
                "balanced_accuracy": np.nan, "mcc": np.nan,
                "actual_up_share": np.nan, "predicted_up_share": np.nan}
    hit_up = tp / (tp + fn) if tp + fn else np.nan        # recall on up days
    hit_down = tn / (tn + fp) if tn + fp else np.nan      # recall on down days
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "days": n,
        "dead_zone_days": int((~labelled).sum()),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": np.nanmean([hit_up, hit_down]),
        # A constant prediction makes the denominator zero; by convention MCC is then 0.
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
        "actual_up_share": (tp + fn) / n,
        "predicted_up_share": (tp + fp) / n,
    }


def summarize(preds: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """Classification metrics for each group of the predictions table.

    by=("model",)               one row per model, all test days pooled
    by=("model", "test_year")   one row per model per fold
    by=("model", "ticker")      one row per model per stock
    """
    check_not_holdout(preds.index)
    by = list(by)
    rows = []
    for key, g in preds.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **classification_metrics(g["up"], g["pred"])})
    return pd.DataFrame(rows, columns=by + METRIC_COLUMNS)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
from features import load_config  # noqa: E402

PERFORMANCE_COLUMNS = ["days", "ann_return", "ann_volatility", "sharpe", "max_drawdown",
                       "turnover_per_year", "cost_per_year"]


def positions(pred, strategy: str) -> np.ndarray:
    """Position held from day t's close to day t+1's close, per unit of capital."""
    pred = np.asarray(pred, dtype=int)
    if strategy == "long_short":
        return np.where(pred == 1, 1.0, -1.0)
    if strategy == "long_flat":
        return np.where(pred == 1, 1.0, 0.0)
    raise ValueError(f"unknown strategy {strategy!r}")


def stock_returns(preds: pd.DataFrame, strategy: str, cost_bps: float) -> pd.DataFrame:
    """Daily results for each stock, for one model's predictions.

    Each stock's positions run continuously across the test years (every day
    from 2005 on is a test day of exactly one fold). The first position is
    paid for as a trade from cash. Days without a next-day return (the last
    day of the data) are dropped.
    """
    df = preds[preds["fwd_return"].notna()].copy()
    df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    df = df.sort_values(["ticker", "date"])
    df["position"] = positions(df["pred"], strategy)
    previous = df.groupby("ticker")["position"].shift(1).fillna(0.0)
    df["turnover"] = (df["position"] - previous).abs()
    df["gross"] = df["position"] * df["fwd_return"]
    df["cost"] = df["turnover"] * cost_bps / 10_000
    df["net"] = df["gross"] - df["cost"]
    return df.set_index("date")


def portfolio_returns(stock_daily: pd.DataFrame, column: str = "net") -> pd.Series:
    """Equal-weight portfolio: each day's return is the average across stocks."""
    return stock_daily.groupby(level="date")[column].mean().sort_index()


def performance(daily: pd.Series, days_per_year: int | None = None) -> dict:
    """Annualized return (compounded), volatility, Sharpe ratio and maximum drawdown.

    The Sharpe ratio here is mean / standard deviation of daily returns times
    sqrt(days_per_year), with no risk-free rate subtracted.
    """
    days_per_year = days_per_year or load_config()["backtest"]["trading_days_per_year"]
    r = pd.Series(daily, dtype=float).dropna()
    n = len(r)
    if n == 0:
        return {"days": 0, "ann_return": np.nan, "ann_volatility": np.nan,
                "sharpe": np.nan, "max_drawdown": np.nan}
    growth = (1 + r).cumprod()
    sd = r.std(ddof=1) if n > 1 else 0.0
    return {
        "days": n,
        "ann_return": growth.iloc[-1] ** (days_per_year / n) - 1,
        "ann_volatility": sd * np.sqrt(days_per_year),
        # Rounding leaves a constant series with a tiny nonzero spread; treat it as zero.
        "sharpe": r.mean() / sd * np.sqrt(days_per_year) if sd > 1e-12 else np.nan,
        "max_drawdown": float((growth / growth.cummax() - 1).min()),   # worst fall from a peak
    }


def backtest(preds: pd.DataFrame, strategies=None, costs_bps=None) -> pd.DataFrame:
    """One row per model x strategy x cost level, with the performance figures.

    turnover_per_year  average units traded per stock per year (1 = one full
                       position bought or sold); cost_per_year is what that
                       turnover cost, as a share of capital.
    """
    check_not_holdout(preds.index)
    cfg = load_config()["backtest"]
    strategies = cfg["strategies"] if strategies is None else list(strategies)
    costs_bps = [cfg["cost_bps_per_side"]] if costs_bps is None else list(costs_bps)
    per_year = cfg["trading_days_per_year"]
    rows = []
    for model, g in preds.groupby("model", sort=True):
        for strategy in strategies:
            for bps in costs_bps:
                daily = stock_returns(g, strategy, bps)
                perf = performance(portfolio_returns(daily), per_year)
                rows.append({"model": model, "strategy": strategy, "cost_bps": bps, **perf,
                             "turnover_per_year": daily["turnover"].groupby(level="date").mean().mean() * per_year,
                             "cost_per_year": portfolio_returns(daily, "cost").mean() * per_year})
    return pd.DataFrame(rows, columns=["model", "strategy", "cost_bps"] + PERFORMANCE_COLUMNS)
