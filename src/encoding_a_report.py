"""Count how often each TA-Lib candlestick pattern fires on the study stocks.

Usage
-----
    python src/encoding_a_report.py

Runs encoding A on the 22 selected stocks over the study period (2002-2025) and
writes results/encoding_a_frequency.csv with one row per pattern:

  fires            days the pattern appeared, across all 22 stocks
  pct_of_days      fires as a share of all stock-days
  bullish/bearish  how many of those were +1 and -1
  direction        "both", "+1 only", "-1 only" or "never fires". A pattern that is only
                   ever +1 is either non-directional (TA-Lib reports a doji as +100) or
                   bullish by definition (a hammer); check which before reading +1 as bullish
  confirmed        days TA-Lib returned +/-200 instead of +/-100 (Hikkake patterns only)
  stocks           how many of the 22 stocks it fired in at least once
  fires_2002_2004  fires in the first three years, roughly the first training window;
                   patterns that barely appear there give the models little to learn from
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from features import PATTERNS, encoding_a, pattern_column  # noqa: E402

OUT = data.ROOT / "results" / "encoding_a_frequency.csv"
STUDY_START, STUDY_END = "2002-01-02", "2025-12-31"
FIRST_WINDOW_END = "2004-12-31"


def selected_stocks() -> list[str]:
    u = pd.read_csv(data.UNIVERSE_CSV)
    return u.loc[u["role"] == "selected", "ticker"].tolist()


def main() -> int:
    stocks = selected_stocks()
    frames = []
    for t in stocks:
        df = data.load_raw(t)
        # Compute on the full history (including the 2001 warm-up), then keep the study period.
        frames.append(encoding_a(df, raw=True).loc[STUDY_START:STUDY_END].assign(ticker=t))
    raw = pd.concat(frames)  # all stocks stacked, so dates repeat
    total_days = len(raw)
    in_first_window = raw.index <= pd.Timestamp(FIRST_WINDOW_END)

    rows = []
    for name in PATTERNS:
        col = pattern_column(name)
        v = raw[col]
        fired = v != 0
        bull, bear = int((v > 0).sum()), int((v < 0).sum())
        rows.append({
            "pattern": col,
            "talib_function": name,
            "fires": int(fired.sum()),
            "pct_of_days": round(100 * fired.mean(), 3),
            "bullish": bull,
            "bearish": bear,
            "direction": ("never fires" if bull + bear == 0 else "both" if bull and bear
                          else "+1 only" if bull else "-1 only"),
            "confirmed": int((v.abs() == 200).sum()),
            "stocks": int(raw.loc[fired, "ticker"].nunique()),
            "fires_2002_2004": int(fired[in_first_window].sum()),
        })
    report = pd.DataFrame(rows).sort_values("fires", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT, index=False)

    print(report.drop(columns=["talib_function"]).to_string(index=False))
    rare = report[report["fires"] < 100]
    print(f"\n{len(stocks)} stocks, {total_days:,} stock-days, {len(PATTERNS)} patterns.")
    print(f"Days with at least one pattern: "
          f"{100 * (raw[[pattern_column(p) for p in PATTERNS]] != 0).any(axis=1).mean():.1f}%")
    print(f"{(report['direction'] == '+1 only').sum()} patterns only ever output +1 "
          "(non-directional like the doji, or bullish by definition like the hammer).")
    print(f"{len(rare)} patterns fire fewer than 100 times in total: "
          + (", ".join(rare["pattern"]) if len(rare) else "none"))
    print(f"Wrote {OUT.relative_to(data.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
