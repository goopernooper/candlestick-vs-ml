"""Build the stock universe from a fixed, written rule.

Rule
----
For each of the 11 GICS sectors, select the two largest current S&P 500
companies by market capitalization that have daily price data on
2002-01-02, the first trading day of 2002. The next two eligible companies
in each sector are recorded as alternates. If a selected stock later fails
the data validation checks, it is replaced by the first alternate in the
same sector. SPY is added as the market proxy.

Because selection uses today's S&P 500 members, the universe is subject to
survivorship bias; the paper discusses this as a limitation.

Usage
-----
    python src/universe.py

Run it once, then commit data/universe.csv and the constituents snapshot.
Market caps change daily, so rerunning later can change the selection.
Only rerun deliberately, and log it in DECISIONS.md if you do.
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

PER_SECTOR = 2
ALTERNATES = 2
CHECK_DATE = "2002-01-02"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATCH = 100


def sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 members with GICS sectors, one row per company."""
    headers = {"User-Agent": "candlestick-vs-ml student research project"}
    html = requests.get(WIKI_URL, headers=headers, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    df = next(t for t in tables if {"Symbol", "GICS Sector", "CIK"} <= set(t.columns))
    df = df.rename(columns={"Symbol": "wiki_symbol", "Security": "name", "GICS Sector": "sector"})
    # Yahoo writes share classes with a dash: BRK.B -> BRK-B
    df["ticker"] = df["wiki_symbol"].str.replace(".", "-", regex=False)
    # Keep one share class per company (e.g. GOOGL/GOOG)
    df = df.drop_duplicates(subset="CIK", keep="first")
    return df[["ticker", "name", "sector", "CIK"]].reset_index(drop=True)


def tickers_with_data_on(tickers: list[str], day: str) -> set[str]:
    """Tickers that have a closing price on `day`."""
    target = pd.Timestamp(day)
    end = (target + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=day, end=end, auto_adjust=True,
                     progress=False, threads=True)
    close = px["Close"]
    if isinstance(close, pd.Series):  # a single ticker comes back as a Series
        close = close.to_frame(tickers[0])
    row = close.reindex([target]).iloc[0]
    return set(row.index[row.notna()])


def market_cap(ticker: str) -> float:
    """Current market cap from Yahoo, or NaN if unavailable."""
    info = yf.Ticker(ticker).fast_info
    try:
        return float(info.market_cap)
    except Exception:
        return float(info["marketCap"])


def main() -> None:
    snapshot = date.today().isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    spx = sp500_constituents()
    spx.to_csv(DATA_DIR / f"sp500_constituents_{snapshot}.csv", index=False)
    print(f"{len(spx)} S&P 500 companies across {spx['sector'].nunique()} sectors")

    tickers = spx["ticker"].tolist()
    eligible: set[str] = set()
    for i in range(0, len(tickers), BATCH):
        eligible |= tickers_with_data_on(tickers[i:i + BATCH], CHECK_DATE)
        time.sleep(2)
    spx["has_2002_data"] = spx["ticker"].isin(eligible)
    print(f"{len(eligible)} of them have data on {CHECK_DATE}")

    cand = spx[spx["has_2002_data"]].copy()
    caps = {}
    for i, t in enumerate(cand["ticker"], 1):
        try:
            caps[t] = market_cap(t)
        except Exception as exc:
            print(f"  no market cap for {t}: {exc}")
            caps[t] = float("nan")
        if i % 25 == 0:
            print(f"  market caps: {i}/{len(cand)}")
            time.sleep(1)
    cand["market_cap"] = cand["ticker"].map(caps)
    cand = cand.dropna(subset=["market_cap"])

    cand = cand.sort_values(["sector", "market_cap"], ascending=[True, False])
    cand["sector_rank"] = cand.groupby("sector").cumcount() + 1
    keep = cand[cand["sector_rank"] <= PER_SECTOR + ALTERNATES].copy()
    keep["role"] = keep["sector_rank"].map(
        lambda r: "selected" if r <= PER_SECTOR else "alternate")

    short = keep[keep["role"] == "selected"].groupby("sector").size()
    for sector, n in short[short < PER_SECTOR].items():
        print(f"WARNING: only {n} eligible stock(s) in {sector}")

    spy = pd.DataFrame([{
        "ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "sector": "Market proxy",
        "market_cap": float("nan"), "sector_rank": 0, "role": "market_proxy",
    }])
    cols = ["ticker", "name", "sector", "market_cap", "sector_rank", "role"]
    out = pd.concat([keep[cols], spy], ignore_index=True)
    out["snapshot_date"] = snapshot
    out.to_csv(DATA_DIR / "universe.csv", index=False)

    selected = out[out["role"] != "alternate"]
    print(f"\nSelected {len(selected) - 1} stocks plus SPY (snapshot {snapshot}):")
    for _, r in keep.iterrows():
        cap = f"${r.market_cap / 1e9:,.0f}B"
        print(f"  {r.sector:<24} {r.role:<9} {r.ticker:<6} {r['name']:<34} {cap:>8}")
    print(f"\nWrote {DATA_DIR / 'universe.csv'}")


if __name__ == "__main__":
    main()
