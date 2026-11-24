"""Do Caginalp & Laurent's (1998) candlestick reversal patterns still work?

Usage
-----
    python src/replicate_caginalp.py

Caginalp & Laurent tested eight three-day reversal patterns on S&P 500 stocks,
1992-1996, and reported that prices reversed after them far more often than
chance (Z = 36 for the bullish patterns), with about 0.9% average profit per
bullish trade before costs. This script repeats their test on our 22 stocks
over 2002-2022. It is a descriptive replication added after results v3
(DECISIONS.md D-026) and does not feed into any model.

Everything below follows the paper (Sections 2.2 and 3), checked against it
on 2026-11-09. The pattern's three days are t-2, t-1, t (their t*+1..t*+3).

Trend (their Definition 3.1). M(d) is the average of the closes of d-2, d-1, d.
Day d is in a downtrend if M(d-6) > M(d-5) > ... > M(d), with at most one of
the six inequalities violated; an uptrend is the mirror image. Every pattern
requires its first day to be in the opposite trend.

Patterns (o, c = open, close of days 1, 2, 3). No conditions on sizes:
  three white soldiers  three up days, each closing higher, each opening
                        inside the previous day's body
  three inside up       down day; day 2's body inside day 1's body; up day
                        closing above day 1's open
  three outside up      down day; up day whose body engulfs it; up day closing
                        higher still
  morning star          down day; day 2's body below day 1's close; up day
                        closing above the middle of day 1's body
and their mirror images: three black crows, three inside down, three outside
down, evening star. Two formulas are printed with evident typos (the star
midpoint and three inside down's last condition); we follow the text.

For comparison, the same test is run with TA-Lib's definitions of the same
patterns (encoding A's columns), which add conditions on candle sizes, with the
same trend rule ("talib"), and with no trend rule ("talib_notrend", which is
what encoding A gives the models).

Outcomes:
  cl        their test: the close of day t against the average of the next
            three closes. Success for a bullish pattern: that average is at or
            above the close (their statement A1); the return is their r_b, buying
            at the close of day t and selling a third at each of the next three
            closes. Bearish patterns mirror this (B1, r_s).
  next_day  close t to close t+1: the target our models predict.

Statistics for each definition, pattern, outcome and period:
  n            signals
  success      share of signals that succeeded
  base         their p0: the success share over all stock-days in the same
               trend (three days earlier, as in the paper), pattern or not
  z_binomial   their Z: (success - base) x sqrt(n) / sqrt(base (1 - base)),
               which counts every signal as independent
  mean_bps     average return in the predicted direction, basis points
  excess_bps   the same after subtracting that day's average over all 22 stocks
  t_dates      excess return averaged within each date, then a t-statistic
               across dates with Newey-West standard errors: signals on the
               same day move together and holding periods overlap (D-005)
  net_bps      mean_bps minus a round trip at the headline cost (2 x 5 bps)

Differences from the paper that remain: 22 large stocks instead of about 500;
split- and dividend-adjusted prices (an ex-dividend day can flip a comparison
between consecutive days by a fraction of a percent); a later period. Only
signals whose last price is before the locked holdout (2023) are used.
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

# name -> (direction: +1 bullish / -1 bearish, encoding A column for the TA-Lib version)
PATTERNS = {
    "three_white_soldiers": (1, "a_3whitesoldiers"),
    "three_inside_up": (1, "a_3inside"),
    "three_outside_up": (1, "a_3outside"),
    "morning_star": (1, "a_morningstar"),
    "three_black_crows": (-1, "a_3blackcrows"),
    "three_inside_down": (-1, "a_3inside"),
    "three_outside_down": (-1, "a_3outside"),
    "evening_star": (-1, "a_eveningstar"),
}
DEFINITIONS = ["paper", "talib", "talib_notrend"]
TREND_STEPS = 6
PERIODS = {"2002-2012": (2002, 2012), "2013-2022": (2013, 2022), "2002-2022": (2002, 2022)}
OUTCOME_LAGS = {"cl": 2, "next_day": 0}          # days of overlap between consecutive holdings


def trends(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(downtrend, uptrend) flags for each day, by the paper's Definition 3.1."""
    step = close.rolling(3).mean().diff()
    known = step.notna().astype(float).rolling(TREND_STEPS).sum() == TREND_STEPS
    falls = (step < 0).astype(float).rolling(TREND_STEPS).sum()
    rises = (step > 0).astype(float).rolling(TREND_STEPS).sum()
    return known & (falls >= TREND_STEPS - 1), known & (rises >= TREND_STEPS - 1)


def paper_shapes(o: pd.Series, c: pd.Series) -> pd.DataFrame:
    """The eight patterns' candle conditions (without the trend), dated on the
    pattern's last day. Bearish patterns are the bullish ones turned upside down."""
    def bullish(o, c):
        o1, o2, o3, c1, c2, c3 = o.shift(2), o.shift(1), o, c.shift(2), c.shift(1), c
        soldiers = ((c1 > o1) & (c2 > o2) & (c3 > o3) & (c3 > c2) & (c2 > c1)
                    & (c1 > o2) & (o2 > o1) & (c2 > o3) & (o3 > o2))
        inside = ((o1 > c1) & (o1 >= o2) & (o2 > c1) & (o1 > c2) & (c2 >= c1)
                  & ~((o1 == o2) & (c2 == c1)) & (c3 > o3) & (c3 > o1))
        outside = ((o1 > c1) & (c2 >= o1) & (c1 >= o2) & ((c2 - o2).abs() > (c1 - o1).abs())
                   & (c3 > o3) & (c3 > c2))
        star = ((o1 > c1) & (o2 != c2) & (c1 > c2) & (c1 > o2) & (c3 > o3) & (c3 > (o1 + c1) / 2))
        return soldiers, inside, outside, star
    up, down = bullish(o, c), bullish(-o, -c)
    names = list(PATTERNS)
    return pd.DataFrame(dict(zip(names, [*up, *down])), index=o.index)


def stock_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """One stock, one row per day: signals under each definition, the trend
    flags for the base rate, and the outcomes.

    Signals and trends only look backward from the pattern's last day t.
    Outcomes look ahead by design, and exit_* records the last price they use.
    """
    o, c = prices["Open"], prices["Close"]
    down, up = trends(c)
    shapes, a = paper_shapes(o, c), encoding_a(prices)
    out = pd.DataFrame(index=prices.index)
    for name, (direction, col) in PATTERNS.items():
        first_day_trend = (down if direction > 0 else up).shift(2, fill_value=False)
        out[f"paper__{name}"] = shapes[name] & first_day_trend
        out[f"talib_notrend__{name}"] = a[col] == direction
        out[f"talib__{name}"] = out[f"talib_notrend__{name}"] & first_day_trend
    # the paper's base rate conditions on the trend at t*, three days before the pattern's last day
    out["base_down"], out["base_up"] = down.shift(3, fill_value=False), up.shift(3, fill_value=False)
    out["cl"] = (c.shift(-1) + c.shift(-2) + c.shift(-3)) / 3 / c - 1
    out["next_day"] = c.shift(-1) / c - 1
    dates = pd.Series(prices.index, index=prices.index)
    out["exit_cl"], out["exit_next_day"] = dates.shift(-3), dates.shift(-1)
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
    """The statistics in the module docstring. frame: all stocks stacked,
    indexed by date, with the columns of stock_frame."""
    groups = {**{n: [n] for n in PATTERNS},
              "all_bullish": [n for n, (d, _) in PATTERNS.items() if d > 0],
              "all_bearish": [n for n, (d, _) in PATTERNS.items() if d < 0]}
    year = frame.index.year
    rows = []
    for outcome, lags in OUTCOME_LAGS.items():
        ret = frame[outcome]
        excess = ret - ret.groupby(level=0).mean().reindex(frame.index).to_numpy()
        for period, (y0, y1) in PERIODS.items():
            in_period = (year >= y0) & (year <= y1) & ret.notna().to_numpy()
            for definition in DEFINITIONS:
                for group, names in groups.items():
                    direction = PATTERNS[names[0]][0]
                    signed, signed_excess = direction * ret, direction * excess
                    if definition == "talib_notrend":
                        base_days = in_period
                    else:
                        base_days = in_period & frame["base_down" if direction > 0 else "base_up"].to_numpy()
                    base = float((signed[base_days] >= 0).mean()) if base_days.any() else np.nan
                    sel = in_period & frame[[f"{definition}__{n}" for n in names]].any(axis=1).to_numpy()
                    n = int(sel.sum())
                    success = float((signed[sel] >= 0).mean()) if n else np.nan
                    ok = n and 0 < base < 1
                    by_date = signed_excess[sel].groupby(level=0).mean()
                    rows.append({
                        "definition": definition, "pattern": group,
                        "direction": "bullish" if direction > 0 else "bearish",
                        "outcome": outcome, "period": period, "n": n, "signal_days": int(by_date.size),
                        "success": success, "base": base,
                        "z_binomial": (success - base) * np.sqrt(n) / np.sqrt(base * (1 - base)) if ok else np.nan,
                        "mean_bps": float(signed[sel].mean() * 1e4) if n else np.nan,
                        "excess_bps": float(signed_excess[sel].mean() * 1e4) if n else np.nan,
                        "t_dates": nw_t(by_date, lags),
                        "net_bps": float(signed[sel].mean() * 1e4 - 2 * cost_bps) if n else np.nan})
    return pd.DataFrame(rows)


def build(stocks, loader) -> pd.DataFrame:
    frame = pd.concat([stock_frame(loader(t)).assign(ticker=t) for t in stocks]).sort_index()
    frame = frame[frame.index >= pd.Timestamp(load_config()["splits"]["train_start"])]
    if holdout_locked():                      # nothing whose last price is in the holdout
        last = holdout_start()
        for outcome in OUTCOME_LAGS:
            frame.loc[~(frame[f"exit_{outcome}"] < last), outcome] = np.nan
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
    table = evaluate(build(dataset.selected_stocks(), data.load_raw), cost)
    path = ROOT / "results" / "caginalp_replication.csv"
    table.to_csv(path, index=False)

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    cols = ["pattern", "n", "success", "base", "z_binomial", "mean_bps", "excess_bps", "t_dates", "net_bps"]
    full = table[(table["outcome"] == "cl") & (table["period"] == "2002-2022")]
    print("Caginalp & Laurent's test (close of the pattern's last day vs the average of the next three "
          "closes), 2002-2022.\nIn 1992-1996 they reported success 0.71 vs base 0.45 (Z = 36) for the "
          "bullish patterns, 0.67 vs 0.53 (Z = 27) for\nthe bearish ones, and average returns of 90 and "
          "27 bps before costs.")
    for definition, label in [("paper", "their definitions"),
                              ("talib", "TA-Lib's definitions with their trend rule"),
                              ("talib_notrend", "TA-Lib's definitions, no trend rule (encoding A)")]:
        print(f"\n{label}:\n")
        print(full[full["definition"] == definition][cols].to_string(index=False, float_format=fmt))
    pooled = table[table["pattern"].str.startswith("all_") & (table["definition"] == "paper")]
    print("\nTheir definitions, all bullish / all bearish patterns, by outcome and period:\n")
    print(pooled[["pattern", "outcome", "period", "n", "success", "base", "z_binomial", "excess_bps",
                  "t_dates", "net_bps"]].to_string(index=False, float_format=fmt))
    print("\nz_binomial counts every signal as independent (their test); t_dates averages signals within "
          f"each day first.\nWrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
