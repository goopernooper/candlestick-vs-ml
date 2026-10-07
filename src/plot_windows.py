"""Plot candlestick charts of random windows for an eyeball check against a charting site.

Usage
-----
    python src/plot_windows.py                         # 10 random windows (same 10 every time)
    python src/plot_windows.py --around BRK-B:2008-10-10   # also chart a window centred on a date

Picks windows of 60 trading days from the selected stocks and SPY, using a fixed
random seed so the same windows are drawn each run. Saves one PNG per window and a
checklist to results/sanity_charts/.

How to compare
--------------
Open the same ticker and dates on a charting site (TradingView, Yahoo, or similar)
and compare the SHAPE: the pattern of up and down days, the gaps, the long wicks.
Don't compare dollar levels. These prices are adjusted for dividends as well as
splits, and most sites adjust only for splits, so older prices here sit lower.
Record the result in checklist.csv: "yes", or "no" plus a note.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot  # noqa: E402
import mplfinance as mpf
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

OUT_DIR = data.ROOT / "results" / "sanity_charts"
WINDOW = 60
N_WINDOWS = 10
SEED = 20261007
FIRST_DAY = pd.Timestamp("2002-01-02")  # the study period, not the warm-up months


def selected_tickers() -> list[str]:
    u = pd.read_csv(data.UNIVERSE_CSV)
    return u.loc[u["role"].isin(["selected", "market_proxy"]), "ticker"].tolist()


def random_windows(tickers: list[str]) -> list[tuple[str, int]]:
    """(ticker, start position) pairs, drawn with a fixed seed."""
    rng = np.random.default_rng(SEED)
    picks = []
    chosen = rng.choice(tickers, size=N_WINDOWS, replace=len(tickers) < N_WINDOWS)
    for t in chosen:
        df = data.load_raw(t)
        first = int(df.index.searchsorted(FIRST_DAY))
        picks.append((str(t), int(rng.integers(first, len(df) - WINDOW + 1))))
    return picks


def centred_window(spec: str) -> tuple[str, int]:
    """'TICKER:YYYY-MM-DD' -> (ticker, start position) with that date in the middle."""
    ticker, day = spec.split(":", 1)
    df = data.load_raw(ticker)
    pos = int(df.index.searchsorted(pd.Timestamp(day)))
    start = min(max(pos - WINDOW // 2, 0), len(df) - WINDOW)
    return ticker, start


def plot(ticker: str, start: int, n: int) -> dict:
    df = data.load_raw(ticker).iloc[start:start + WINDOW]
    first, last = df.index[0].date(), df.index[-1].date()
    name = f"{n:02d}_{ticker.replace('^', '')}_{first}_{last}.png"
    change = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    fig, _ = mpf.plot(
        df, type="candle", style="yahoo", volume=not ticker.startswith("^"),
        ylabel="Adjusted price", figsize=(11, 6), tight_layout=True, returnfig=True,
    )
    fig.suptitle(f"{ticker}  {first} to {last}  ({change:+.1%} close to close)",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.savefig(OUT_DIR / name, dpi=110, bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    return {"chart": name, "ticker": ticker, "start": first, "end": last,
            "change": f"{change:+.1%}", "matches": "", "notes": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--around", action="append", default=[], metavar="TICKER:DATE",
                        help="also chart a window centred on this date (repeatable)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = random_windows(selected_tickers()) + [centred_window(s) for s in args.around]
    rows = [plot(t, s, i) for i, (t, s) in enumerate(windows, 1)]

    checklist = OUT_DIR / "checklist.csv"
    new = pd.DataFrame(rows)
    if checklist.exists():  # keep answers you already filled in
        old = pd.read_csv(checklist, dtype=str, keep_default_na=False).set_index("chart")
        for col in ["matches", "notes"]:
            new[col] = [old[col].get(c, "") for c in new["chart"]]
    new.to_csv(checklist, index=False)

    print(new[["chart", "change", "matches"]].to_string(index=False))
    print(f"\nSaved {len(rows)} charts and checklist.csv in {OUT_DIR.relative_to(data.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
