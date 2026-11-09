"""Do Caginalp & Laurent's (1998) candlestick reversal patterns still work?

Usage
-----
    python src/replicate_caginalp.py

Caginalp & Laurent reported that eight three-day reversal patterns predicted
S&P 500 stocks in 1992-1996, with nearly 1% profit over a short holding period
and very high statistical significance. This script tests the same eight
patterns on our 22 stocks over 2002-2022, using TA-Lib's definitions (the same
columns as encoding A). It is a descriptive replication added after results v3
(DECISIONS.md D-026); it does not feed into any model.

  bullish: three white soldiers, three inside up, three outside up, morning star
  bearish: three black crows, three inside down, three outside down, evening star

Trend condition. A reversal pattern only makes sense after a trend in the other
direction. Caginalp & Laurent required one; TA-Lib does not. As described in
secondary sources, their downtrend is a 3-day moving average of closes that
falls over six days (an uptrend rises). Each pattern is tested three ways:
  none     TA-Lib's definition as it is
  net      the 3-day average at the close before the pattern is below (above)
           its value six days earlier
  strict   the 3-day average fell (rose) on each of those six days
Check the exact rule against the paper; it is not reproduced from it here.

Holding periods (the pattern ends on day t):
  next_day   close t to close t+1: the target our models predict
  cl_style   buy at the open of t+1, sell at the average of the closes of
             t+1, t+2 and t+3: close to Caginalp & Laurent's design

Statistics for each pattern, trend rule, holding period and period
(2002-2012, 2013-2022, 2002-2022):
  n             signals
  success       share of signals followed by a move in the predicted direction
  base          the same share over all stock-days (the success rate of always
                predicting that direction)
  z_binomial    a binomial test of success against base, counting every signal
                as independent: the kind of test behind the original paper's
                very large significance
  mean_bps      average return in the predicted direction, in basis points
  excess_bps    the same after subtracting that day's average return of all
                22 stocks (removes the market's move)
  t_dates       excess return averaged within each date, then a t-statistic
                across dates with Newey-West standard errors: signals on the
                same day move together, so they aren't independent (D-005)
  net_bps       mean_bps minus a round trip at the headline cost (2 x 5 bps)

Only signals whose exit is before the locked holdout (2023) are used.
Writes results/caginalp_replication.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import encoding_a, load_config  # noqa: E402
from splits import holdout_locked, holdout_start  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# name -> (encoding A column, direction: +1 bullish, -1 bearish)
PATTERNS = {
    "three_white_soldiers": ("a_3whitesoldiers", 1),
    "three_inside_up": ("a_3inside", 1),
    "three_outside_up": ("a_3outside", 1),
    "morning_star": ("a_morningstar", 1),
    "three_black_crows": ("a_3blackcrows", -1),
    "three_inside_down": ("a_3inside", -1),
    "three_outside_down": ("a_3outside", -1),
    "evening_star": ("a_eveningstar", -1),
}
TREND_DAYS = 6
PERIODS = {"2002-2012": (2002, 2012), "2013-2022": (2013, 2022), "2002-2022": (2002, 2022)}
HORIZON_LAGS = {"next_day": 0, "cl_style": 2}      # overlap between consecutive days' holdings


def stock_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """One stock: pattern signals, trend flags and forward returns, one row per day.

    Trend flags look only backward (they end at the close before the pattern's
    first day, t-3). Forward returns look ahead by design: they are what the
    pattern is supposed to predict, and exit_date records the last price used.
    """
    a = encoding_a(prices)
    c, o = prices["Close"], prices["Open"]
    ma = c.rolling(3).mean().shift(3)                       # 3-day average ending at t-3
    change = ma - ma.shift(TREND_DAYS)
    daily = ma.diff()
    falling_each = (daily < 0).astype(float).rolling(TREND_DAYS).sum() == TREND_DAYS
    rising_each = (daily > 0).astype(float).rolling(TREND_DAYS).sum() == TREND_DAYS
    out = pd.DataFrame(index=prices.index)
    for name, (col, direction) in PATTERNS.items():
        out[name] = (a[col] == direction)
    out["down_net"], out["up_net"] = change < 0, change > 0
    out["down_strict"], out["up_strict"] = falling_each, rising_each
    out["next_day"] = c.shift(-1) / c - 1
    out["cl_style"] = (c.shift(-1) + c.shift(-2) + c.shift(-3)) / 3 / o.shift(-1) - 1
    dates = pd.Series(prices.index, index=prices.index)
    out["exit_next_day"] = dates.shift(-1)
    out["exit_cl_style"] = dates.shift(-3)
    return out


def nw_t(series: pd.Series, lags: int) -> float:
    """t-statistic of a mean, with Newey-West (Bartlett) standard errors."""
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def evaluate(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """The statistics above for every pattern, trend rule, holding period and period.

    frame: all stocks stacked, indexed by date, with a ticker column."""
    rows = []
    groups = {**{n: [n] for n in PATTERNS},
              "all_bullish": [n for n, (_, d) in PATTERNS.items() if d > 0],
              "all_bearish": [n for n, (_, d) in PATTERNS.items() if d < 0]}
    year = frame.index.year
    for horizon, lags in HORIZON_LAGS.items():
        market = frame.groupby(level=0)[horizon].mean()
        excess = frame[horizon] - market.reindex(frame.index).to_numpy()
        for period, (y0, y1) in PERIODS.items():
            in_period = (year >= y0) & (year <= y1) & frame[horizon].notna().to_numpy()
            for group, names in groups.items():
                direction = PATTERNS[names[0]][1]
                fired = frame[names].any(axis=1).to_numpy()
                trend = {"none": np.ones(len(frame), bool),
                         "net": frame["down_net" if direction > 0 else "up_net"].to_numpy(),
                         "strict": frame["down_strict" if direction > 0 else "up_strict"].to_numpy()}
                signed_all = direction * frame[horizon]
                base = float((signed_all[in_period] > 0).mean())
                for rule, ok in trend.items():
                    sel = in_period & fired & ok
                    n = int(sel.sum())
                    signed = signed_all[sel]
                    signed_excess = direction * excess[sel]
                    success = float((signed > 0).mean()) if n else np.nan
                    z = ((success - base) * np.sqrt(n) / np.sqrt(base * (1 - base))) if n else np.nan
                    by_date = signed_excess.groupby(level=0).mean()
                    rows.append({"pattern": group, "direction": "bullish" if direction > 0 else "bearish",
                                 "trend_rule": rule, "horizon": horizon, "period": period, "n": n,
                                 "signal_days": int(by_date.size), "success": success, "base": base,
                                 "z_binomial": z,
                                 "mean_bps": float(signed.mean() * 1e4) if n else np.nan,
                                 "excess_bps": float(signed_excess.mean() * 1e4) if n else np.nan,
                                 "t_dates": nw_t(by_date, lags),
                                 "net_bps": float(signed.mean() * 1e4 - 2 * cost_bps) if n else np.nan})
    return pd.DataFrame(rows)


def build(stocks, loader) -> pd.DataFrame:
    frames = []
    for t in stocks:
        f = stock_frame(loader(t))
        frames.append(f.assign(ticker=t))
    frame = pd.concat(frames).sort_index()
    start = pd.Timestamp(load_config()["splits"]["train_start"])
    frame = frame[frame.index >= start]
    if holdout_locked():                      # nothing whose exit reaches the holdout
        last = holdout_start()
        for h in HORIZON_LAGS:
            frame.loc[~(frame[f"exit_{h}"] < last), h] = np.nan
        frame = frame[frame.index < last]
    return frame


def main() -> int:
    import data
    import dataset
    import manifest
    if not manifest.check(quiet=True):
        print("Stopping: data/raw does not match data/manifest.json.")
        return 1
    cost = load_config()["backtest"]["cost_bps_per_side"]
    frame = build(dataset.selected_stocks(), data.load_raw)
    table = evaluate(frame, cost)
    path = ROOT / "results" / "caginalp_replication.csv"
    table.to_csv(path, index=False)

    cols = ["pattern", "n", "success", "base", "z_binomial", "mean_bps", "excess_bps", "t_dates", "net_bps"]
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    for rule in ["none", "strict"]:
        view = table[(table["horizon"] == "cl_style") & (table["period"] == "2002-2022")
                     & (table["trend_rule"] == rule)]
        print(f"\nCaginalp & Laurent's patterns, 2002-2022, buy at next open, sell at the average of "
              f"the next 3 closes; trend rule: {rule}\n")
        print(view[cols].to_string(index=False, float_format=fmt))
    pooled = table[(table["pattern"].str.startswith("all_")) & (table["trend_rule"] == "strict")]
    print("\nAll bullish / all bearish patterns (strict trend rule), by period and holding period:\n")
    print(pooled[["pattern", "horizon", "period", "n", "success", "base", "excess_bps", "t_dates", "net_bps"]]
          .to_string(index=False, float_format=fmt))
    print("\nz_binomial treats every signal as independent (as in the original test); t_dates averages "
          "signals within each day first.\n"
          f"Wrote {path.relative_to(ROOT)} (every pattern x trend rule x holding period x period)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
