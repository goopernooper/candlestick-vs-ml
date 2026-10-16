"""Prediction targets: forward return, direction, and the dead zone.

Usage
-----
    python src/labels.py     # summary of the labels for the 22 selected stocks

These are the only values in the project that use future prices, by design:
the label dated t describes what happens after day t's close. That is why the
walk-forward splitter needs a purge gap of `horizon` days between training and
test data (week 3).

Columns returned by make_labels(), one row per day t:
  fwd_return   Close[t + horizon] / Close[t] - 1. Used by the backtest on every day.
  vol          standard deviation of the last vol_window daily returns, ending at t.
               Known at day t's close.
  threshold    dead_zone_k * vol * sqrt(horizon): how big a move must be to count.
  dead_zone    True when |fwd_return| < threshold.
  up           1 if fwd_return > 0, 0 if below 0. Missing (<NA>) in the dead zone
               and on the last `horizon` days, where there is no future price yet.
               This is the column models train on and classification metrics use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_config  # noqa: E402

LABEL_COLUMNS = ["fwd_return", "vol", "threshold", "dead_zone", "up"]


def make_labels(df: pd.DataFrame, horizon: int | None = None, dead_zone_k: float | None = None,
                vol_window: int | None = None) -> pd.DataFrame:
    """Labels for one stock. Settings default to the `labels` section of config.yaml."""
    if horizon is None or dead_zone_k is None or vol_window is None:
        cfg = load_config()["labels"]
        horizon = cfg["horizon"] if horizon is None else horizon
        dead_zone_k = cfg["dead_zone_k"] if dead_zone_k is None else dead_zone_k
        vol_window = cfg["vol_window"] if vol_window is None else vol_window

    close = df["Close"]
    fwd = close.shift(-horizon) / close - 1                     # looks `horizon` days ahead
    daily = close / close.shift(1) - 1
    vol = daily.rolling(vol_window).std()                       # ends at day t: no look-ahead
    # A move over `horizon` days is naturally about sqrt(horizon) times bigger.
    threshold = dead_zone_k * vol * np.sqrt(horizon)

    known = fwd.notna() & threshold.notna()
    dead = known & (fwd.abs() < threshold)
    usable = known & ~dead
    up = pd.array(np.where(usable, (fwd > 0).astype(int), 0), dtype="Int8")
    up[~usable.to_numpy()] = pd.NA

    return pd.DataFrame({"fwd_return": fwd, "vol": vol, "threshold": threshold,
                         "dead_zone": dead, "up": up}, index=df.index)


def main() -> int:
    import data
    u = pd.read_csv(data.UNIVERSE_CSV)
    stocks = u.loc[u["role"] == "selected", "ticker"].tolist()
    lab = pd.concat([make_labels(data.load_raw(t)).loc["2002-01-02":"2025-12-31"].assign(ticker=t)
                     for t in stocks])
    cfg = load_config()["labels"]
    labelled = lab["up"].notna()
    print(f"Labels for {len(stocks)} stocks, 2002-2025, horizon {cfg['horizon']} day(s), "
          f"dead zone {cfg['dead_zone_k']} x {cfg['vol_window']}-day volatility.\n")
    print(f"Stock-days:                 {len(lab):,}")
    print(f"In the dead zone:           {lab['dead_zone'].mean():.1%}  (dropped from training and metrics)")
    print(f"No future price yet:        {lab['fwd_return'].isna().sum()}  (the last day of each stock)")
    print(f"Labelled days:              {labelled.sum():,}")
    print(f"Share of labelled days up:  {lab.loc[labelled, 'up'].astype(float).mean():.1%}  "
          "(the majority-class baseline's accuracy)")
    by_year = lab.groupby(lab.index.year)["dead_zone"].mean()
    print(f"Dead-zone share by year:    {by_year.min():.1%} to {by_year.max():.1%} "
          "(should be fairly steady, since the threshold scales with volatility)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
