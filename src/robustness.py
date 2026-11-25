"""Robustness checks side by side (ANALYSIS_PLAN.md section 12).

Usage
-----
    python src/robustness.py v4 v4_dz0 v4_dz2 v4_perstock

Compares the main results (the first version) with each robustness run, model
by model: pooled MCC, long/short Sharpe ratio at 5 bps, and the section 9
verdicts against the strongest baseline and buy-and-hold. Each version's
significance tests must have been run first (python src/significance.py
<version>). The next-day-open check has its own script, robustness_open.py.

Writes results/<first version>/robustness_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def setting(info: dict, default_k: float) -> str:
    """A short name for a run's setting, from its run_info.json."""
    parts = []
    if info.get("per_stock"):
        parts.append("per-stock")
    k = info.get("dead_zone_k", default_k)
    if k != default_k:
        parts.append(f"dead zone {k:g}x")
    return ", ".join(parts) or "main"


def describe(version: str) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from features import load_config
    info = json.loads((ROOT / "results" / version / "run_info.json").read_text())
    return f"{version} ({setting(info, load_config()['labels']['dead_zone_k'])})"


def load(version: str) -> pd.DataFrame:
    folder = ROOT / "results" / version
    s = pd.read_csv(folder / "summary.csv").set_index("model")[["mcc", "ls_sharpe"]]
    path = folder / "significance_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python src/significance.py {version}` first")
    v = pd.read_csv(path).set_index("model")
    dm_col = next(c for c in v.columns if c.startswith("class_dm_vs_"))
    return s.join(v[[dm_col, "class_p", "beats_classification", "beats_economic", "genuine_value"]]
                  .rename(columns={dm_col: "class_dm"}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("versions", nargs="+", help="main version first, then the robustness runs")
    args = parser.parse_args()
    tables = {v: load(v) for v in args.versions}
    rows = []
    for v, t in tables.items():
        for model, r in t.iterrows():
            rows.append({"version": v, "setting": describe(v), "model": model, **r.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / args.versions[0] / "robustness_summary.csv", index=False)

    models = [m for m in tables[args.versions[0]].index if all(m in t.index for t in tables.values())]
    print("MCC (DM statistic against the strongest baseline on the classification loss; negative = "
          "lower loss; 'b' = beats it under section 9):\n")
    print(f"  {'model':<15}" + "".join(f"{describe(v)[:26]:>27}" for v in args.versions))
    order = tables[args.versions[0]].loc[models, "mcc"].sort_values(ascending=False).index
    for m in order:
        cells = ""
        for v in args.versions:
            r = tables[v].loc[m]
            dm = "" if pd.isna(r["class_dm"]) else f"{r['class_dm']:+.2f}"
            flag = "b" if r["beats_classification"] is True or r["beats_classification"] == "True" else " "
            cells += f"{r['mcc']:>+12.4f} ({dm:>6}) {flag}     "
        print(f"  {m:<15}{cells}")
    print("\nLong/short Sharpe at 5 bps ('e' = beats buy-and-hold under section 9):\n")
    print(f"  {'model':<15}" + "".join(f"{describe(v)[:26]:>27}" for v in args.versions))
    for m in order:
        cells = ""
        for v in args.versions:
            r = tables[v].loc[m]
            flag = "e" if r["beats_economic"] is True or r["beats_economic"] == "True" else " "
            cells += f"{r['ls_sharpe']:>+21.2f} {flag}     "
        print(f"  {m:<15}{cells}")
    gv = {v: t.index[t["genuine_value"].astype(str) == "True"].tolist() for v, t in tables.items()}
    print("\nGenuine predictive value (section 9): "
          + "; ".join(f"{describe(v)}: {', '.join(x) if x else 'none'}" for v, x in gv.items()))
    print(f"\nWrote results/{args.versions[0]}/robustness_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
