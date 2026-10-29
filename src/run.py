"""One command: run models through every walk-forward fold and write a results table.

Usage
-----
    python src/run.py --version v1                    # every model in EXPERIMENTS
    python src/run.py --version v0 --models majority always_long previous_day random

Steps:
  1. Check data/raw against data/manifest.json (stop if the data changed).
  2. Build the pooled table of features and labels for the 22 selected stocks.
  3. For each fold (test years 2005-2022): fit each model on the training days
     that have a label, using only that model's feature columns, then predict
     every test day.
  4. Score the predictions (accuracy, balanced accuracy, MCC) and backtest them
     (long/short and long/flat at the headline cost).
  5. Write everything to results/<version>/, stamped with the git commit, and
     run the sanity checks for the baselines.

Results from uncommitted code are marked "+dirty": commit first if a result
might go in the paper, so the stamped commit really produced it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_table  # noqa: E402
from evaluate import backtest, summarize  # noqa: E402
from features import PATTERNS, load_config, pattern_column, window_columns  # noqa: E402
from models.baselines import AlwaysLong, Majority, PreviousDay, RandomGuess  # noqa: E402
from models.ml import LightGBMModel  # noqa: E402
from models.pattern import KNNModel  # noqa: E402
from splits import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def baseline_models() -> dict:
    """name -> function(fold) returning a fresh model. The random baseline gets a
    different (but fixed) seed in each fold, so its guesses aren't repeated year to year."""
    seed = load_config()["seed"]
    return {
        "majority": lambda fold: Majority(),
        "always_long": lambda fold: AlwaysLong(),
        "previous_day": lambda fold: PreviousDay(),
        "random": lambda fold: RandomGuess(seed=seed + fold.test_year),
    }


def experiments() -> dict:
    """Every model the runner knows: name -> (feature sets to build, columns, model maker).

    Names follow model_encoding: lightgbm_a uses encoding A's 61 pattern columns,
    lightgbm_b uses continuous encoding B over a window of days, and knn_b3 /
    knn_b5 compare 3- and 5-day windows of continuous encoding B.
    """
    seed = load_config()["seed"]
    a_cols = [pattern_column(p) for p in PATTERNS]
    b_cols = window_columns()
    spec = {name: {"feature_sets": ["normalized_ohlc"], "features": ["n_close"], "make": make}
            for name, make in baseline_models().items()}
    spec["lightgbm_a"] = {"feature_sets": ["encoding_a"], "features": a_cols,
                          "make": lambda fold: LightGBMModel(seed=seed + fold.test_year)}
    spec["lightgbm_b"] = {"feature_sets": ["encoding_b_window"], "features": b_cols,
                          "make": lambda fold: LightGBMModel(seed=seed + fold.test_year)}
    for days in load_config()["models"]["knn"]["windows"]:
        spec[f"knn_b{days}"] = {"feature_sets": ["encoding_b_window"], "features": window_columns(days),
                                "make": lambda fold: KNNModel()}
    return spec


def run_experiments(table: pd.DataFrame, folds, specs: dict, progress=None) -> pd.DataFrame:
    """The predictions table described in evaluate.py, for every model and fold.

    Each model trains only on its own feature columns, on training days with a
    label, and predicts every test day where its features exist. All models must
    end up predicting the same stock-days, so they are compared on equal terms.
    """
    out = []
    for fold in folds:
        train_all, test_all = fold.select(table, "train"), fold.select(table, "test")
        for name, sp in specs.items():
            cols = list(sp["features"])
            train = train_all[train_all["up"].notna() & train_all[cols].notna().all(axis=1)]
            test = test_all[test_all[cols].notna().all(axis=1)]
            if test.empty:
                continue
            model = sp["make"](fold).fit(train[cols], train["up"].astype(int))
            p = np.asarray(model.predict_proba(test[cols]), dtype=float)
            out.append(test[["ticker", "up", "fwd_return"]].assign(
                model=name, test_year=fold.test_year, p_up=p, pred=(p >= 0.5).astype(int)))
        if progress:
            progress(fold)
    preds = pd.concat(out)
    counts = preds.groupby("model").size()
    if counts.nunique() > 1:
        raise RuntimeError(f"models predicted different numbers of days: {counts.to_dict()}")
    return preds


def run_models(table: pd.DataFrame, folds, models: dict, feature_cols) -> pd.DataFrame:
    """Shortcut: run several models that all use the same feature columns."""
    return run_experiments(table, folds, {name: {"features": list(feature_cols), "make": make}
                                          for name, make in models.items()})


def git_stamp() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def summary_table(preds: pd.DataFrame, cost_bps: float, stamp: str, run_at: str) -> pd.DataFrame:
    """One row per model: pooled classification metrics plus the headline backtest."""
    metrics = summarize(preds).set_index("model")
    bt = backtest(preds, costs_bps=[cost_bps])
    cols = {}
    for strategy, short in [("long_short", "ls"), ("long_flat", "lf")]:
        b = bt[bt["strategy"] == strategy].set_index("model")
        for c in ["ann_return", "sharpe", "max_drawdown", "turnover_per_year"]:
            cols[f"{short}_{c}"] = b[c]
    table = metrics[["days", "accuracy", "balanced_accuracy", "mcc", "actual_up_share",
                     "predicted_up_share"]].join(pd.DataFrame(cols))
    return table.reset_index().assign(cost_bps=cost_bps, git_commit=stamp, run_at=run_at)


def baseline_checks(summary: pd.DataFrame) -> list[tuple[str, bool]]:
    s = summary.set_index("model")
    checks = []
    if "always_long" in s.index:
        a = s.loc["always_long"]
        checks.append(("always_long accuracy equals the share of up days",
                       abs(a["accuracy"] - a["actual_up_share"]) < 1e-12))
        checks.append(("always_long MCC is 0", a["mcc"] == 0))
    if "random" in s.index:
        checks.append(("random accuracy is within 1.5 points of 50%",
                       abs(s.loc["random", "accuracy"] - 0.5) < 0.015))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", default="v1", help="results go in results/<version>/")
    parser.add_argument("--models", nargs="+", help="which models to run (default: all)")
    args = parser.parse_args()
    specs = experiments()
    if args.models:
        unknown = set(args.models) - set(specs)
        if unknown:
            print(f"Unknown models: {sorted(unknown)}. Available: {sorted(specs)}")
            return 1
        specs = {name: specs[name] for name in args.models}

    import manifest
    if not manifest.check(quiet=True):
        print("Stopping: data/raw does not match data/manifest.json.")
        return 1

    cfg = load_config()
    stamp = git_stamp()
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if stamp.endswith("+dirty"):
        print("Note: uncommitted changes. Results are stamped +dirty; commit before using them in the paper.\n")

    feature_sets = sorted({fs for sp in specs.values() for fs in sp["feature_sets"]})
    table = build_table(feature_sets=feature_sets)
    folds = walk_forward(table.index)
    print(f"Running {len(specs)} models on {len(folds)} folds: ", end="", flush=True)
    preds = run_experiments(table, folds, specs,
                            progress=lambda f: print(f.test_year, end=" ", flush=True))
    print("\n")

    out = ROOT / "results" / args.version
    out.mkdir(parents=True, exist_ok=True)
    cost = cfg["backtest"]["cost_bps_per_side"]
    summary = summary_table(preds, cost, stamp, run_at)
    summary.to_csv(out / "summary.csv", index=False)
    summarize(preds, by=["model", "test_year"]).assign(git_commit=stamp).to_csv(out / "metrics_by_year.csv", index=False)
    summarize(preds, by=["model", "ticker"]).assign(git_commit=stamp).to_csv(out / "metrics_by_stock.csv", index=False)
    backtest(preds, costs_bps=cfg["backtest"]["cost_grid_bps"]).assign(git_commit=stamp).to_csv(out / "backtest.csv", index=False)
    preds.to_parquet(out / "predictions.parquet")
    (out / "run_info.json").write_text(json.dumps({
        "git_commit": stamp, "run_at": run_at, "folds": [f.test_year for f in folds],
        "models": {name: sp["features"] for name, sp in specs.items()},
        "stocks": int(table["ticker"].nunique()), "test_days": int(len(preds) // preds["model"].nunique()),
        "first_date": str(table.index.min().date()), "last_date": str(table.index.max().date()),
        "config": cfg}, indent=2, default=str) + "\n")

    show = summary[["model", "accuracy", "balanced_accuracy", "mcc", "ls_ann_return", "ls_sharpe",
                    "lf_ann_return", "lf_sharpe"]]
    print(f"Results {args.version} (commit {stamp}), {len(folds)} folds, "
          f"{table['ticker'].nunique()} stocks, costs {cost} bps per side:\n")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    ok = True
    for name, passed in baseline_checks(summary):
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    print(f"\nWrote results/{args.version}/: summary.csv, metrics_by_year.csv, metrics_by_stock.csv, "
          "backtest.csv, run_info.json, predictions.parquet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
