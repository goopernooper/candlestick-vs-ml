"""Time one fold of every model and estimate how long the full tuned run takes.

Usage
-----
    python src/time_models.py                      # every model that gets tuned
    python src/time_models.py --models svm_a svm_b

For each model, fits its config.yaml settings on the smallest training window
(the 2005 fold) and the largest (the 2022 fold), and predicts that fold's test
year, timing both. The time per fit is then treated as a straight line in the
number of training days, which gives an estimate for every other fold and for
every tuning trial (fitted on a training window minus its last year).

    estimated run     = the sum over the 18 folds
    estimated tuning  = n_trials x the retune years x one fit on that window

Tuning is cached in results/tuning/, so it is paid once: later reruns cost only
the run. Treat the estimates as rough (within a factor of 2 or so): tuning tries
settings that are slower than the defaults (more trees, smaller leaves, larger k).
Nothing here touches the test years' results; it only measures time.

Writes results/timing.csv, stamped with the computer and the git commit.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_table  # noqa: E402
from features import load_config  # noqa: E402
from run import experiments, git_stamp  # noqa: E402
from splits import walk_forward  # noqa: E402
from tune import retune_year_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def usable(frame: pd.DataFrame, cols, labelled: bool) -> pd.DataFrame:
    keep = frame[cols].notna().all(axis=1)
    if labelled:
        keep &= frame["up"].notna()
    return frame[keep]


def time_fold(table: pd.DataFrame, fold, spec: dict) -> tuple[float, int]:
    """Seconds to fit the model on the fold's training days and predict its
    test year, and the number of training days."""
    cols = list(spec["features"])
    train = usable(fold.select(table, "train"), cols, labelled=True)
    test = usable(fold.select(table, "test"), cols, labelled=False)
    start = time.perf_counter()
    model = spec["make"](fold).fit(train[cols], train["up"].astype(int))
    model.predict_proba(test[cols])
    return time.perf_counter() - start, len(train)


def line_through(n1: int, t1: float, n2: int, t2: float):
    """Seconds as a function of training days, through the two measurements
    (never below zero)."""
    slope = (t2 - t1) / (n2 - n1) if n2 != n1 else 0.0
    return lambda n: max(0.0, t1 + slope * (n - n1))


def estimate(seconds_at, fold_rows: dict, fit_rows: dict, n_trials: int) -> tuple[float, float]:
    """(run seconds, tuning seconds) from a seconds-per-fit function, the
    training days of each fold, and the fit days of each retune year."""
    run = sum(seconds_at(n) for n in fold_rows.values())
    tuning = n_trials * sum(seconds_at(n) for n in fit_rows.values())
    return run, tuning


def hours(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", help="which models to time (default: every tuned model)")
    args = parser.parse_args()
    specs = experiments()
    names = args.models or [n for n, sp in specs.items() if "tune" in sp]
    unknown = set(names) - set(specs)
    if unknown:
        print(f"Unknown models: {sorted(unknown)}. Available: {sorted(specs)}")
        return 1

    import manifest
    if not manifest.check(quiet=True):
        print("Stopping: data/raw does not match data/manifest.json.")
        return 1

    cfg = load_config()
    feature_sets = sorted({fs for n in names for fs in specs[n]["feature_sets"]})
    print("Building the table...", flush=True)
    table = build_table(feature_sets=feature_sets)
    folds = walk_forward(table.index)
    first, last = folds[0], folds[-1]
    n_trials, years = cfg["tuning"]["n_trials"], cfg["tuning"]["retune_years"]
    retune = sorted({retune_year_for(f.test_year, years) for f in folds})
    by_year = {f.test_year: f for f in folds}
    print(f"Timing {len(names)} models on the {first.test_year} and {last.test_year} folds "
          f"({os.cpu_count()} CPU cores). This fits each model twice.\n", flush=True)

    rows = []
    for name in names:
        sp = specs[name]
        cols = list(sp["features"])
        try:
            t1, n1 = time_fold(table, first, sp)
            t2, n2 = time_fold(table, last, sp)
        except Exception as exc:                                  # report and carry on
            print(f"  {name:<15} FAILED: {exc}", flush=True)
            continue
        at = line_through(n1, t1, n2, t2)
        fold_rows = {f.test_year: len(usable(f.select(table, "train"), cols, True)) for f in folds}
        fit_rows = {y: len(usable(by_year[y].tuning().select(table, "train"), cols, True))
                    for y in retune}
        run, tuning = estimate(at, fold_rows, fit_rows, n_trials if "tune" in sp else 0)
        rows.append({"model": name, f"seconds_{first.test_year}": round(t1, 1), f"rows_{first.test_year}": n1,
                     f"seconds_{last.test_year}": round(t2, 1), f"rows_{last.test_year}": n2,
                     "est_run_minutes": round(run / 60, 1), "est_tuning_minutes": round(tuning / 60, 1),
                     "est_total_minutes": round((run + tuning) / 60, 1)})
        print(f"  {name:<15} {first.test_year}: {t1:6.1f}s   {last.test_year}: {t2:6.1f}s   "
              f"-> run ~{hours(run)}, tuning ~{hours(tuning)}", flush=True)

    if not rows:
        return 1
    out = pd.DataFrame(rows).sort_values("est_total_minutes", ascending=False)
    out = out.assign(cpu_cores=os.cpu_count(), machine=platform.platform(), git_commit=git_stamp())
    path = ROOT / "results" / "timing.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    run_total, tune_total = out["est_run_minutes"].sum() * 60, out["est_tuning_minutes"].sum() * 60
    print(f"\nEstimated full run (python src/run.py --version v3), all {len(out)} models:")
    print(f"  tuning, paid once (cached afterwards): ~{hours(tune_total)}")
    print(f"  18 folds with the tuned settings:      ~{hours(run_total)}")
    print(f"  total:                                 ~{hours(run_total + tune_total)}")
    slow = out.iloc[0]
    print(f"Slowest: {slow['model']} (~{hours(slow['est_total_minutes'] * 60)}). "
          "Rough estimates: tuning tries slower settings than the defaults.")
    print(f"\nWrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
