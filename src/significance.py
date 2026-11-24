"""Statistical tests: does any model really beat the baselines?

Usage
-----
    python src/significance.py v4            # full run, 10,000 replications
    python src/significance.py v4 --reps 1000    # quick look

Implements ANALYSIS_PLAN.md sections 8 and 9 on the daily loss series written by
losses.py (one value per model per day, averaged across stocks, D-005):

  classification   balanced misclassification (losses_classification.csv, D-027)
  economic         minus the long/short net return at 5 bps (losses_economic.csv)
  long_flat        minus the long/flat net return at 5 bps
                   (losses_economic_long_flat.csv): for the buy-and-hold claim

Tests
-----
Diebold-Mariano: for each model against each benchmark, d_t = loss(model) -
loss(benchmark). The statistic is mean(d) / its Newey-West standard error
(Bartlett kernel, lag floor(4 (T/100)^(2/9))), compared with a standard normal,
two-sided. Negative = the model has the lower loss.

Model Confidence Set (Hansen, Lunde & Nason, 2011; arch.bootstrap.MCS): the set
of models that contains the best one with 90% confidence, from a stationary
block bootstrap. A model outside it is significantly worse than the best.

SPA (Hansen, 2005; arch.bootstrap.SPA): is any model better than the
benchmark at all? Small p-values say yes. The "consistent" p-value is reported.

Benchmarks: for classification, the baseline with the highest pooled MCC (in
practice reversal); for the economic losses, always_long.

Verdicts (section 9)
--------------------
  beats_classification  MCC above the strongest baseline's AND a significantly
                        lower classification loss (DM, 5%)
  beats_economic        long/short Sharpe above always_long's AND a significantly
                        lower economic loss
  beats_buy_and_hold    long/flat Sharpe above always_long's AND a significantly
                        lower long/flat loss
  genuine_value         beats_classification, and the strongest baseline is
                        outside the classification MCS while the model is in it

Models whose loss series is identical to another's (majority and always_long
here) enter the MCS and SPA once, under the first name.

Writes results/<version>/significance_dm.csv, significance_mcs.csv,
significance_spa.csv and significance_summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOSSES = {"classification": "losses_classification.csv",
          "economic": "losses_economic.csv",
          "long_flat": "losses_economic_long_flat.csv"}


def newey_west_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def dm_test(d, lags: int | None = None) -> dict:
    """Diebold-Mariano test of mean(d) = 0, Newey-West standard error."""
    x = pd.Series(d, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 3:
        raise ValueError("fewer than 3 days to test")
    lags = newey_west_lag(n) if lags is None else lags
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    se = np.sqrt(var / n) if var > 0 else 0.0
    stat = x.mean() / se if se > 0 else (0.0 if x.mean() == 0 else np.sign(x.mean()) * np.inf)
    return {"days": n, "mean_diff": float(x.mean()), "dm_stat": float(stat),
            "p_value": float(2 * norm.sf(abs(stat))), "nw_lags": lags}


def complete(losses: pd.DataFrame) -> pd.DataFrame:
    """Days on which every model has a loss (drops all-dead-zone days)."""
    return losses.dropna(how="any")


def distinct(losses: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop columns identical to an earlier one: {dropped: kept}."""
    keep, same = [], {}
    for c in losses.columns:
        twin = next((k for k in keep if losses[c].equals(losses[k])), None)
        if twin is None:
            keep.append(c)
        else:
            same[c] = twin
    return losses[keep], same


def mcs(losses: pd.DataFrame, size: float, reps: int, block_size: int, seed: int) -> pd.DataFrame:
    from arch.bootstrap import MCS
    data, same = distinct(complete(losses))
    m = MCS(data, size=size, reps=reps, block_size=block_size, method="R",
            bootstrap="stationary", seed=seed)
    m.compute()
    p = m.pvalues["Pvalue"]
    out = pd.DataFrame({"mcs_pvalue": p, "in_mcs": p.index.isin(m.included)})
    for dropped, kept in same.items():
        out.loc[dropped] = out.loc[kept]
    return out.reindex(losses.columns)


def spa(losses: pd.DataFrame, benchmark: str, reps: int, block_size: int, seed: int) -> dict:
    from arch.bootstrap import SPA
    data, same = distinct(complete(losses))
    if benchmark not in data.columns:
        benchmark = same[benchmark]
    others = data.drop(columns=benchmark)
    s = SPA(data[benchmark], others, block_size=block_size, reps=reps, bootstrap="stationary",
            studentize=True, seed=seed)
    s.compute()
    pv = s.pvalues
    return {"benchmark": benchmark, "models": others.shape[1], "p_lower": float(pv["lower"]),
            "p_consistent": float(pv["consistent"]), "p_upper": float(pv["upper"])}


def strongest_baseline(summary: pd.DataFrame, baselines) -> str:
    present = [b for b in baselines if b in summary.index]
    return summary.loc[present, "mcc"].idxmax()


def analyse(losses: dict, summary: pd.DataFrame, cfg: dict | None = None, reps: int | None = None) -> dict:
    """All tests and verdicts. losses: {name: date x model table}; summary:
    results summary indexed by model (mcc, ls_sharpe, lf_sharpe)."""
    cfg = cfg or load_config()
    sig, seed = cfg["significance"], cfg["seed"]
    reps = sig["reps"] if reps is None else reps
    alpha, size, block = sig["alpha"], sig["mcs_size"], sig["block_size"]
    strongest = strongest_baseline(summary, sig["baselines"])
    econ = sig["economic_benchmark"]
    bench = {"classification": strongest, "economic": econ, "long_flat": econ}

    dm_rows = []
    for name, table in losses.items():
        for model in table.columns:
            for b in sorted({bench[name], econ, strongest} if name == "classification" else {bench[name]}):
                if model == b:
                    continue
                pair = table[[model, b]].dropna()
                dm_rows.append({"loss": name, "model": model, "benchmark": b,
                                **dm_test(pair[model] - pair[b])})
    dm = pd.DataFrame(dm_rows)

    mcs_tables = []
    for name, table in losses.items():
        if name == "long_flat":
            continue
        mcs_tables.append(mcs(table, size, reps, block, seed).assign(loss=name))
    mcs_all = pd.concat(mcs_tables).rename_axis("model").reset_index()

    spa_rows = [{"loss": name, **spa(table, bench[name], reps, block, seed)}
                for name, table in losses.items()]
    spa_all = pd.DataFrame(spa_rows)

    def dm_of(loss, model, b):
        r = dm[(dm["loss"] == loss) & (dm["model"] == model) & (dm["benchmark"] == b)]
        return r.iloc[0] if len(r) else None

    def significant_win(r):
        return r is not None and r["p_value"] < alpha and r["mean_diff"] < 0

    in_mcs = mcs_all[mcs_all["loss"] == "classification"].set_index("model")["in_mcs"]
    rows = []
    for model in summary.index:
        if model in sig["baselines"]:
            continue
        c, e, lf = (dm_of("classification", model, strongest), dm_of("economic", model, econ),
                    dm_of("long_flat", model, econ))
        beats_c = bool(summary.loc[model, "mcc"] > summary.loc[strongest, "mcc"] and significant_win(c))
        rows.append({
            "model": model, "mcc": summary.loc[model, "mcc"],
            "class_dm_vs_" + strongest: None if c is None else c["dm_stat"],
            "class_p": None if c is None else c["p_value"],
            "beats_classification": beats_c,
            "ls_sharpe": summary.loc[model, "ls_sharpe"],
            "econ_dm_vs_" + econ: None if e is None else e["dm_stat"],
            "econ_p": None if e is None else e["p_value"],
            "beats_economic": bool(summary.loc[model, "ls_sharpe"] > summary.loc[econ, "ls_sharpe"]
                                   and significant_win(e)),
            "lf_sharpe": summary.loc[model, "lf_sharpe"],
            "lf_p": None if lf is None else lf["p_value"],
            "beats_buy_and_hold": bool(summary.loc[model, "lf_sharpe"] > summary.loc[econ, "lf_sharpe"]
                                       and significant_win(lf)),
            "in_class_mcs": bool(in_mcs.get(model, False)),
            "genuine_value": bool(beats_c and in_mcs.get(model, False) and not in_mcs.get(strongest, True)),
        })
    verdict = pd.DataFrame(rows).sort_values("mcc", ascending=False)
    return {"dm": dm, "mcs": mcs_all, "spa": spa_all, "summary": verdict,
            "strongest": strongest, "economic_benchmark": econ, "reps": reps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="results version, e.g. v4")
    parser.add_argument("--reps", type=int, help="bootstrap replications (default: config.yaml)")
    args = parser.parse_args()
    folder = ROOT / "results" / args.version
    missing = [f for f in LOSSES.values() if not (folder / f).exists()]
    if missing:
        print(f"Missing {', '.join(missing)} in results/{args.version}/: run "
              f"`python src/losses.py {args.version}` first.")
        return 1
    losses = {name: pd.read_csv(folder / f, index_col="date", parse_dates=True) for name, f in LOSSES.items()}
    summary = pd.read_csv(folder / "summary.csv").set_index("model")
    print(f"Running tests on results/{args.version} ({len(losses['economic'])} days, "
          f"{summary.shape[0]} models, {args.reps or load_config()['significance']['reps']:,} "
          "bootstrap replications)...", flush=True)
    out = analyse(losses, summary, reps=args.reps)
    for key in ["dm", "mcs", "spa", "summary"]:
        out[key].to_csv(folder / f"significance_{key}.csv", index=False)

    s, strongest, econ = out["summary"], out["strongest"], out["economic_benchmark"]
    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)  # noqa: E731
    print(f"\nStrongest baseline for classification: {strongest} (MCC "
          f"{summary.loc[strongest, 'mcc']:.4f}); economic benchmark: {econ}.")
    print("DM statistics are negative when the model has the lower loss.\n")
    cols = ["model", "mcc", f"class_dm_vs_{strongest}", "class_p", "beats_classification",
            "ls_sharpe", f"econ_dm_vs_{econ}", "econ_p", "beats_economic", "lf_p", "beats_buy_and_hold",
            "in_class_mcs", "genuine_value"]
    print(s[cols].to_string(index=False, formatters={c: fmt for c in cols}))
    m = out["mcs"]
    for loss in ["classification", "economic"]:
        inside = m[(m["loss"] == loss) & m["in_mcs"]]["model"].tolist()
        print(f"\n90% Model Confidence Set, {loss} loss ({len(inside)} of {m['loss'].eq(loss).sum()} models):"
              f"\n  {', '.join(inside)}")
    print("\nSPA test: can any model beat the benchmark? (consistent p-value; small = yes)")
    for _, r in out["spa"].iterrows():
        print(f"  {r['loss']:<15} vs {r['benchmark']:<12} p = {r['p_consistent']:.3f}")
    winners = s.loc[s["genuine_value"], "model"].tolist()
    print("\nGenuine predictive value (section 9): "
          + (", ".join(winners) if winners else "no model meets the criteria (D-006)."))
    print(f"\nWrote results/{args.version}/significance_dm.csv, significance_mcs.csv, "
          "significance_spa.csv, significance_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
