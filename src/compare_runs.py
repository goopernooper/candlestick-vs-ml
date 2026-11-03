"""Compare two results versions model by model.

Usage
-----
    python src/compare_runs.py v1 v2

For every model in either run, shows pooled MCC, accuracy and the long/short
Sharpe ratio side by side, and marks each model:
  same      in both runs with identical numbers (the expected outcome for a
            model whose code and settings didn't change: the runs are
            deterministic)
  CHANGED   in both runs with different numbers
  new       only in the second run
  dropped   only in the first run

Returns exit code 1 if any model is CHANGED, so an unintended change can't go
unnoticed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHOWN = ["mcc", "accuracy", "ls_sharpe"]
COMPARED = ["days", "accuracy", "balanced_accuracy", "mcc", "ls_ann_return", "ls_sharpe",
            "lf_ann_return", "lf_sharpe"]


def load_summary(version: str, results_dir: Path = ROOT / "results") -> pd.DataFrame:
    path = results_dir / version / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python src/run.py --version {version}` first")
    return pd.read_csv(path).set_index("model")


def compare(old: pd.DataFrame, new: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """One row per model with both runs' headline numbers and a status."""
    rows = []
    for model in list(old.index) + [m for m in new.index if m not in old.index]:
        a = old.loc[model] if model in old.index else None
        b = new.loc[model] if model in new.index else None
        if a is None:
            status = "new"
        elif b is None:
            status = "dropped"
        else:
            cols = [c for c in COMPARED if c in old.columns and c in new.columns]
            same = np.allclose(a[cols].astype(float), b[cols].astype(float), atol=tol, rtol=0, equal_nan=True)
            status = "same" if same else "CHANGED"
        row = {"model": model, "status": status}
        for c in SHOWN:
            row[f"{c}_old"] = np.nan if a is None else a[c]
            row[f"{c}_new"] = np.nan if b is None else b[c]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old", help="earlier version, e.g. v1")
    parser.add_argument("new", help="later version, e.g. v2")
    args = parser.parse_args()
    old, new = load_summary(args.old), load_summary(args.new)
    table = compare(old, new)
    print(f"{args.old} (commit {old['git_commit'].iloc[0]}) vs "
          f"{args.new} (commit {new['git_commit'].iloc[0]}):\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}", na_rep="-"))
    changed = table.loc[table["status"] == "CHANGED", "model"].tolist()
    if changed:
        print(f"\nCHANGED: {', '.join(changed)}. If their code or settings didn't change, find out why "
              "before using either run.")
        return 1
    print("\nEvery model in both runs gave identical results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
