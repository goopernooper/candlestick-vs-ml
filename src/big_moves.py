"""Flag every daily move larger than 20% so each one can be confirmed by hand.

Usage
-----
    python src/big_moves.py

A move is flagged when either the close-to-close return or the overnight gap
(open vs. the previous close) is larger than 20% in either direction. ^VIX is
skipped: moves that large are routine for a volatility index.

Writes data/big_moves.csv, one row per flagged day, with context to help judge it:
  spy_return        SPY's return that day (a big SPY move suggests a market-wide event)
  next_return       the following day's return (a bad tick usually reverses at once)
  volume_ratio      volume vs. the prior 20-day average (real news brings heavy volume)
  split / dividend  corporate actions Yahoo recorded that day
  suspicion         an automatic hint: "split-sized", "reverses next day",
                    "undoes previous day", "gap only, check the open", "market-wide", or blank
  explanation       left blank for you to fill in after checking the news

Your explanations are kept when you rerun the script. The task is done when every
row has one. If a row turns out to be a data error, note the fix in DECISIONS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

OUT = data.ROOT / "data" / "big_moves.csv"
THRESHOLD = 0.20
SKIP = {"^VIX"}
# Price ratios a forgotten split would leave behind, e.g. 2-for-1 halves the price.
SPLIT_RATIOS = [1 / 2, 1 / 3, 1 / 4, 2 / 3, 1 / 5, 1 / 10, 2, 3, 4, 5, 8, 10, 32]
COLUMNS = ["ticker", "date", "prev_close", "open", "close", "return", "gap",
           "next_return", "spy_return", "volume_ratio", "split", "dividend",
           "suspicion", "explanation"]


def round_trip(a: float, b: float) -> bool:
    """True if two big consecutive moves cancel out (net change within 5%)."""
    return bool(np.isfinite(a) and np.isfinite(b)
                and min(abs(a), abs(b)) > THRESHOLD / 2 and np.sign(a) == -np.sign(b)
                and abs((1 + a) * (1 + b) - 1) < 0.05)


def suspicion(ratio: float, ret: float, gap: float, prv: float, nxt: float,
              spy: float, split: float) -> str:
    hints = []
    if split or any(abs(ratio / r - 1) < 0.02 for r in SPLIT_RATIOS):
        hints.append("split-sized")
    if round_trip(ret, nxt):
        hints.append("reverses next day")
    if abs(ret) > THRESHOLD and round_trip(prv, ret):
        hints.append("undoes previous day")
    if abs(gap) > THRESHOLD and abs(ret) < THRESHOLD / 2:
        hints.append("gap only, check the open")
    if np.isfinite(spy) and abs(spy) >= 0.05 and np.sign(spy) == np.sign(ret):
        hints.append("market-wide")
    return "; ".join(hints)


def flag_ticker(ticker: str, spy_ret: pd.Series) -> list[dict]:
    df = data.load_raw(ticker)
    prev = df["Close"].shift(1)
    ret = df["Close"] / prev - 1
    gap = df["Open"] / prev - 1
    nxt = ret.shift(-1)
    prv = ret.shift(1)
    vol_ratio = df["Volume"] / df["Volume"].shift(1).rolling(20).mean()
    hits = (ret.abs() > THRESHOLD) | (gap.abs() > THRESHOLD)

    rows = []
    for date in df.index[hits.fillna(False).to_numpy()]:
        r, g = ret[date], gap[date]
        main = r if abs(r) >= abs(g) else g
        rows.append({
            "ticker": ticker, "date": date.date().isoformat(),
            "prev_close": round(prev[date], 4), "open": round(df.at[date, "Open"], 4),
            "close": round(df.at[date, "Close"], 4),
            "return": round(r, 4), "gap": round(g, 4),
            "next_return": round(nxt[date], 4) if pd.notna(nxt[date]) else np.nan,
            "spy_return": round(spy_ret.get(date, np.nan), 4),
            "volume_ratio": round(vol_ratio[date], 1) if pd.notna(vol_ratio[date]) else np.nan,
            "split": df.at[date, "Stock Splits"], "dividend": df.at[date, "Dividends"],
            "suspicion": suspicion(1 + main, r, g, prv[date], nxt[date],
                                   spy_ret.get(date, np.nan), df.at[date, "Stock Splits"]),
            "explanation": "",
        })
    return rows


def main() -> int:
    spy = data.load_raw("SPY")["Close"]
    spy_ret = spy / spy.shift(1) - 1
    tickers = [t for t in data.universe_tickers()
               if t not in SKIP and data.raw_path(t).exists()]

    rows = [row for t in tickers for row in flag_ticker(t, spy_ret)]
    out = pd.DataFrame(rows, columns=COLUMNS)

    # Keep explanations you already wrote.
    if OUT.exists() and len(out):
        old = pd.read_csv(OUT, dtype={"explanation": str}, keep_default_na=False)
        notes = old.set_index(["ticker", "date"])["explanation"]
        keys = list(zip(out["ticker"], out["date"]))
        out["explanation"] = [notes.get(k, "") for k in keys]

    out = out.sort_values(["date", "ticker"])
    out.to_csv(OUT, index=False)

    if out.empty:
        print(f"No moves larger than {THRESHOLD:.0%} in {len(tickers)} tickers.")
        return 0
    print(f"{len(out)} moves larger than {THRESHOLD:.0%} across {out['ticker'].nunique()} "
          f"of {len(tickers)} tickers:\n")
    show = out.drop(columns=["prev_close", "open", "close", "dividend", "explanation"])
    print(show.to_string(index=False))
    print("\nBy ticker: " + ", ".join(f"{t} {n}" for t, n in out["ticker"].value_counts().items()))
    for hint in ["split-sized", "reverses next day", "gap only"]:
        n = out["suspicion"].str.contains(hint).sum()
        if n:
            print(f"Check first: {n} row(s) marked '{hint}'")
    done = (out["explanation"].str.strip() != "").sum()
    print(f"\n{done} of {len(out)} explained. Wrote {OUT.relative_to(data.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
