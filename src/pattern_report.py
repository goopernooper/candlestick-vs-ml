"""How sparse are the symbol contexts? Counts on the first fold's training data only.

Usage
-----
    python src/pattern_report.py

Counts every context of 1 to 3 consecutive symbols in the training window of the
first walk-forward fold (2002-2004, the data the 2005 model learns from), pooled
across the 22 stocks. Only training data is used, so nothing here peeks at any
test year; the shrinkage and minimum-count settings for the next tasks can be
chosen from it.

Writes results/pattern_context_counts.csv: every context with n, ups, its raw
up_rate, and p_up, the estimate after shrinkage toward the base rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from dataset import build_table  # noqa: E402
from features import symbol_window_columns  # noqa: E402
from features import load_config  # noqa: E402
from models.pattern import context_table, count_contexts, shrink  # noqa: E402
from splits import walk_forward  # noqa: E402

OUT = data.ROOT / "results" / "pattern_context_counts.csv"
THRESHOLDS = [30, 100]


def main() -> int:
    table = build_table(feature_sets=["encoding_b_symbol_window"])
    fold = walk_forward(table.index)[0]
    train = fold.select(table, "train")
    train = train[train["up"].notna()]
    X, y = train[symbol_window_columns()], train["up"].astype(int)
    base = float(y.mean())
    m = load_config()["models"]["pattern_miner"]["shrinkage_m"]
    counts = shrink(count_contexts(X, y), base, m)
    t = context_table(counts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)

    print(f"Training data of the first fold: {fold.train[0].date()} to {fold.train[-1].date()}, "
          f"{len(y):,} labelled stock-days, up share {base:.3f}.\n")
    rows = []
    for length, c in counts.items():
        possible = 20 ** length
        row = {"length": length, "possible": possible, "seen": len(c),
               "median_n": int(c["n"].median()), "max_n": int(c["n"].max())}
        for k in THRESHOLDS:
            enough = c["n"] >= k
            row[f"contexts_n>={k}"] = int(enough.sum())
            row[f"days_covered_n>={k}"] = f"{c.loc[enough, 'n'].sum() / c['n'].sum():.0%}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nRaw up rates swing far more for rare contexts. After shrinkage (m = {m}, toward "
          f"the base rate {base:.3f}) they stay close to the base rate unless seen often:")
    for length, c in counts.items():
        raw = c["ups"] / c["n"]
        for lo, hi in [(1, 9), (10, 99), (100, 10**9)]:
            sel = (c["n"] >= lo) & (c["n"] <= hi)
            if sel.any():
                label = f"n {lo}-{hi}" if hi < 10**9 else f"n {lo}+"
                print(f"  length {length}, {label:<10}: {int(sel.sum()):>5} contexts | raw "
                      f"{raw[sel].min():.2f} to {raw[sel].max():.2f} | shrunk "
                      f"{c.loc[sel, 'p_up'].min():.3f} to {c.loc[sel, 'p_up'].max():.3f}")
    print(f"\nWrote {OUT.relative_to(data.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
