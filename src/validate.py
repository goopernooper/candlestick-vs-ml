"""Check the raw data snapshot for errors before anything is built on it.

Usage
-----
    python src/validate.py

Checks every ticker in data/raw/:
  1. High >= max(Open, Close)              (failure if broken)
  2. Low  <= min(Open, Close)              (failure if broken)
  3. No missing prices                     (failure)
  4. No zero or negative prices            (failure)
  5. No duplicate dates                    (failure)
  6. Same trading calendar as SPY          (failure: a missing or extra day)
  7. Zero-range days, where High == Low    (counted, not a failure)
  8. Zero-volume days for stocks           (counted, not a failure)

Writes two files, both small enough to commit:
  data/validation_report.csv   one row per ticker with every count
  data/validation_details.csv  one row per problem day, for investigating

Every failure needs an explanation (a real event, or a data error and what you
did about it) before the data is used. Record those in DECISIONS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import manifest  # noqa: E402

REPORT = data.ROOT / "data" / "validation_report.csv"
DETAILS = data.ROOT / "data" / "validation_details.csv"
OHLC = ["Open", "High", "Low", "Close"]
CALENDAR = "SPY"
# Adjusted prices are rescaled originals, so tiny rounding differences appear.
# Gaps under 0.0001% of the price count as rounding, not as broken candles.
REL_TOL = 1e-6
FAILURES = ["missing_prices", "nonpositive_prices", "duplicate_dates",
            "high_below_body", "low_above_body", "missing_vs_spy", "extra_vs_spy"]


def check_ticker(ticker: str, df: pd.DataFrame, spy_dates: pd.DatetimeIndex,
                 details: list) -> dict:
    """Run every check on one ticker. Appends problem days to `details`."""
    def flag(check: str, mask, note=None) -> int:
        hits = np.asarray(mask, dtype=bool)
        for i in np.flatnonzero(hits):
            details.append({"ticker": ticker, "check": check, "date": df.index[i].date(),
                            "detail": note(df.iloc[i]) if note else ""})
        return int(hits.sum())

    o, h, l, c = (df[col] for col in OHLC)
    body_top, body_bottom = np.maximum(o, c), np.minimum(o, c)
    tol = REL_TOL * body_top.abs()
    ohlc_str = lambda r: f"O={r.Open:.4f} H={r.High:.4f} L={r.Low:.4f} C={r.Close:.4f}"

    result = {
        "ticker": ticker,
        "rows": len(df),
        "first_date": df.index.min().date() if len(df) else None,
        "last_date": df.index.max().date() if len(df) else None,
        "missing_prices": flag("missing_prices", df[OHLC].isna().any(axis=1)),
        "nonpositive_prices": flag("nonpositive_prices", (df[OHLC] <= 0).any(axis=1), ohlc_str),
        "duplicate_dates": flag("duplicate_dates", pd.Series(df.index.duplicated(), index=df.index)),
        "high_below_body": flag("high_below_body", (body_top - h) > tol, ohlc_str),
        "low_above_body": flag("low_above_body", (l - body_bottom) > tol, ohlc_str),
    }

    # Calendar: compare against SPY over the span this ticker covers.
    dates = df.index.unique()
    span = spy_dates[(spy_dates >= dates.min()) & (spy_dates <= dates.max())]
    missing = span.difference(dates)
    extra = dates.difference(spy_dates)
    for d in missing:
        details.append({"ticker": ticker, "check": "missing_vs_spy", "date": d.date(),
                        "detail": "SPY traded; this ticker has no row"})
    for d in extra:
        details.append({"ticker": ticker, "check": "extra_vs_spy", "date": d.date(),
                        "detail": "row on a day SPY did not trade"})
    result["missing_vs_spy"] = len(missing)
    result["extra_vs_spy"] = len(extra)
    result["starts_after_spy"] = bool(dates.min() > spy_dates.min())

    # Information only: these are real market events, but downstream code must handle them.
    result["zero_range_days"] = flag("zero_range_days", (h - l).abs() <= tol, ohlc_str)
    is_index = ticker.startswith("^")  # ^VIX has no volume
    result["zero_volume_days"] = (0 if is_index else
                                  flag("zero_volume_days", df["Volume"] == 0))
    result["failures"] = sum(result[k] for k in FAILURES)
    return result


def main() -> int:
    if not manifest.check(quiet=True):
        print("WARNING: data/raw/ does not match data/manifest.json. "
              "Validating anyway, but find out why.\n")

    tickers = [t for t in data.universe_tickers() if data.raw_path(t).exists()]
    absent = [t for t in data.universe_tickers() if t not in tickers]
    spy_dates = pd.DatetimeIndex(data.load_raw(CALENDAR).index.unique()).sort_values()

    details: list[dict] = []
    rows = [check_ticker(t, data.load_raw(t), spy_dates, details) for t in tickers]
    report = pd.DataFrame(rows)
    report.to_csv(REPORT, index=False)
    pd.DataFrame(details, columns=["ticker", "check", "date", "detail"]).to_csv(DETAILS, index=False)

    shown = ["ticker", "rows", "first_date"] + FAILURES + ["zero_range_days", "zero_volume_days"]
    print(report[shown].rename(columns={
        "missing_prices": "nan", "nonpositive_prices": "<=0", "duplicate_dates": "dup",
        "high_below_body": "hi<body", "low_above_body": "lo>body",
        "missing_vs_spy": "miss_cal", "extra_vs_spy": "extra_cal",
        "zero_range_days": "0range", "zero_volume_days": "0vol",
    }).to_string(index=False))

    bad = report[report["failures"] > 0]
    late = report[report["starts_after_spy"]]
    print()
    if absent:
        print(f"Not downloaded: {', '.join(absent)}")
    if len(late):
        print(f"Start later than SPY: {', '.join(late['ticker'])} (check this is expected)")
    if len(bad):
        print(f"{len(bad)} ticker(s) with failures: {', '.join(bad['ticker'])}. "
              f"See {DETAILS.relative_to(data.ROOT)} for the exact days.")
    else:
        print(f"No failures across {len(report)} tickers.")
    print(f"Zero-range days in total: {int(report['zero_range_days'].sum())}")
    print(f"Wrote {REPORT.relative_to(data.ROOT)} and {DETAILS.relative_to(data.ROOT)}")
    return 1 if len(bad) or absent else 0


if __name__ == "__main__":
    sys.exit(main())
