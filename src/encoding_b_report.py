"""Check symbolic encoding B's bin edges on 2002-2004 data only.

Usage
-----
    python src/encoding_b_report.py

The edges in config.yaml were picked in advance. This checks that none of the
levels they create is nearly empty in the first three years (roughly the first
training window). Only 2002-2004 is looked at, so no test-period data can
influence the edges.

Prints, for the 22 selected stocks:
  - percentiles of the continuous body and wick values (in ATR units)
  - the share of days in each body level and each wick level
  - how often each of the 20 symbols occurs
and writes the symbol table to results/encoding_b_edges_check.csv.

A level with under 5% of days, or a symbol with under 30 occurrences, is flagged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from features import (BODY_LEVELS, N_SYMBOLS, encoding_b_continuous,  # noqa: E402
                      encoding_b_symbolic, load_config, symbol_label)

OUT = data.ROOT / "results" / "encoding_b_edges_check.csv"
MIN_LEVEL_SHARE = 5.0   # percent of days
MIN_SYMBOL_COUNT = 30


def selected_stocks() -> list[str]:
    u = pd.read_csv(data.UNIVERSE_CSV)
    return u.loc[u["role"] == "selected", "ticker"].tolist()


def main() -> int:
    cfg = load_config()["encoding_b"]
    start, end = cfg["checked_on"]["start"], cfg["checked_on"]["end"]
    cont, sym = [], []
    for t in selected_stocks():
        df = data.load_raw(t)
        cont.append(encoding_b_continuous(df).loc[start:end])
        sym.append(encoding_b_symbolic(df).loc[start:end])
    cont, sym = pd.concat(cont), pd.concat(sym)
    n = len(sym)
    print(f"Checking edges on {start} to {end}: {n:,} stock-days. "
          f"Body edges {cfg['body_edges']}, wick edge {cfg['wick_edge']}.\n")

    pct = cont[["b_body", "b_upper", "b_lower"]].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T.round(2)
    pct.columns = ["p5", "p25", "median", "p75", "p95"]
    print("Continuous values (ATR units):")
    print(pct.to_string(), "\n")

    flags = []
    body_share = sym["b_body_level"].value_counts(normalize=True).reindex(range(5), fill_value=0) * 100
    print("Body levels (% of days):")
    for lvl, share in body_share.items():
        mark = "  <-- nearly empty" if share < MIN_LEVEL_SHARE else ""
        print(f"  {BODY_LEVELS[lvl]:>3}  {share:5.1f}%{mark}")
        if mark:
            flags.append(f"body level {BODY_LEVELS[lvl]}")
    for col, name in [("b_upper_long", "upper"), ("b_lower_long", "lower")]:
        long_share = 100 * sym[col].mean()
        print(f"Long {name} wick: {long_share:.1f}% of days, short: {100 - long_share:.1f}%")
        for label, share in [("long", long_share), ("short", 100 - long_share)]:
            if share < MIN_LEVEL_SHARE:
                flags.append(f"{label} {name} wick")

    counts = sym["b_symbol"].value_counts().reindex(range(N_SYMBOLS), fill_value=0)
    table = pd.DataFrame({
        "symbol": range(N_SYMBOLS),
        "label": [symbol_label(s) for s in range(N_SYMBOLS)],
        "days": counts.to_numpy(),
        "pct_of_days": (100 * counts / n).round(2).to_numpy(),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print("\nSymbols (body, upper wick, lower wick; capital U/L = long wick):")
    print(table.to_string(index=False))

    rare = table[table["days"] < MIN_SYMBOL_COUNT]
    print()
    if flags:
        print("Nearly empty levels: " + ", ".join(flags) + ". Consider adjusting the edges "
              "(and log the change in DECISIONS.md).")
    else:
        print(f"No body or wick level is nearly empty (all at least {MIN_LEVEL_SHARE:.0f}% of days).")
    if len(rare):
        print(f"Rare symbols (under {MIN_SYMBOL_COUNT} days): " + ", ".join(rare["label"]))
    print(f"Wrote {OUT.relative_to(data.ROOT)}")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
