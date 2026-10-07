"""Download and cache raw daily price data.

Usage
-----
    python src/data.py            # download anything not already cached
    python src/data.py --force    # re-download everything (replaces the snapshot)

What it downloads
-----------------
Every ticker in data/universe.csv (the 22 selected stocks, their 22
alternates, and SPY) plus ^VIX. Alternates are included so that if a
selected stock fails validation, its replacement is already in the
snapshot and nothing has to be re-downloaded.

Prices are split- and dividend-adjusted (auto_adjust=True, set explicitly).
Dividends and stock splits are kept as extra columns so the validation step
can check large moves against them.

Rule for the rest of the project
--------------------------------
Experiments read prices with load_raw() and never call yfinance. Yahoo can
revise its history, so re-downloading mid-project can silently change results.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "data" / "universe.csv"
RAW_DIR = ROOT / "data" / "raw"

# Start three months before 2002 so rolling features (14-day ATR, 20-day
# volatility) are warmed up by January 2002. Decimalization finished in
# April 2001, so this buffer contains no fractional-tick prices.
START = "2001-10-01"
END = "2026-01-01"  # yfinance's end date is exclusive: the last day is 2025-12-31
EXTRA_TICKERS = ["^VIX"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
ACTION_COLUMNS = ["Dividends", "Stock Splits"]
MAX_TRIES = 3


def universe_tickers() -> list[str]:
    """Selected stocks, alternates, SPY, then ^VIX."""
    tickers = pd.read_csv(UNIVERSE_CSV)["ticker"].tolist()
    return tickers + [t for t in EXTRA_TICKERS if t not in tickers]


def raw_path(ticker: str) -> Path:
    """File for one ticker, e.g. data/raw/AAPL.parquet or data/raw/VIX.parquet."""
    return RAW_DIR / f"{ticker.replace('^', '')}.parquet"


def fetch(ticker: str) -> pd.DataFrame:
    """Download one ticker's adjusted daily history from Yahoo."""
    df = yf.Ticker(ticker).history(start=START, end=END, interval="1d",
                                   auto_adjust=True, actions=True)
    if df is None or df.empty:
        raise ValueError("no rows returned")
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    for c in ACTION_COLUMNS:  # indexes like ^VIX may come back without these
        if c not in df.columns:
            df[c] = 0.0
    # Yahoo stamps daily bars at midnight New York time; keep just the date.
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    # Keep the data raw: no filling or de-duplicating. Validation reports problems.
    return df[PRICE_COLUMNS + ACTION_COLUMNS].sort_index()


def download(force: bool = False) -> int:
    """Download every ticker that isn't cached yet. Returns a process exit code."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = universe_tickers()
    if force:
        print("--force: re-downloading everything and replacing the snapshot.")
    print(f"{len(tickers)} tickers, {START} to 2025-12-31\n")

    failed = []
    for ticker in tickers:
        path = raw_path(ticker)
        if path.exists() and not force:
            print(f"  cached  {ticker}")
            continue
        df = None
        for attempt in range(1, MAX_TRIES + 1):
            try:
                df = fetch(ticker)
                break
            except Exception as exc:
                if attempt == MAX_TRIES:
                    print(f"  FAILED  {ticker}: {exc}")
                    failed.append(ticker)
                else:
                    time.sleep(5 * attempt)
        if df is None:
            continue
        df.to_parquet(path)
        print(f"  saved   {ticker:<6} {len(df):>5} rows  "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        time.sleep(1)  # be gentle with Yahoo

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}. "
              "Run the script again to retry just those.")
        return 1
    print(f"\nAll {len(tickers)} tickers are in {RAW_DIR}")
    return 0


def load_raw(ticker: str) -> pd.DataFrame:
    """Read one ticker from the local snapshot. Never downloads."""
    path = raw_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the snapshot once with `python src/data.py`; "
            "experiments never download.")
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download everything, replacing the snapshot")
    sys.exit(download(force=parser.parse_args().force))


if __name__ == "__main__":
    main()
