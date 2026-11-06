"""Hyperparameter tuning with the same budget for every model family.

The runner (run.py) calls tuned_settings() for every model whose spec names a
search space ("tune": family). For each retune year (config.yaml, tuning) the
fold of that year is split again with fold.tuning(): the model is fitted on
the training window minus its last year and scored by MCC on that last year.
Optuna's TPE sampler proposes n_trials settings, and the first trial is always
the family's settings from config.yaml, models. The best-scoring settings are
then used, refitted on each fold's whole training window, for that fold and
every later fold up to the next retune year. The test years are never touched.

Every family gets the same number of trials, so differences between models
reflect the model class rather than the effort spent tuning it.

Results are cached in results/tuning/<model>.json, one entry per retune year,
recording every trial. An entry is reused only while the search space, the
trial budget, the seed, the model's settings in config.yaml and the tuning
split are unchanged; run.py --retune ignores the cache.

Caveat for the paper: one tuning year is about 5,500 labelled stock-days, so
MCC on it has a standard error of about 0.013, comparable to the differences
being tuned. Tuning can't find much signal that isn't there; the point is that
no model family is handicapped by settings chosen by hand.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import classification_metrics  # noqa: E402
from features import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "tuning"


# ---------------------------------------------------------------- search spaces
def _label(choice) -> str:
    """Optuna's categorical choices must be simple values, so lists such as an
    MLP's layer sizes are offered as their JSON text and decoded afterwards."""
    return json.dumps(choice)


def suggest(trial, space: dict) -> dict:
    """One set of settings drawn from a search space (config.yaml, tuning.spaces)."""
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            label = trial.suggest_categorical(name, [_label(c) for c in spec["choices"]])
            params[name] = json.loads(label)
        else:
            raise ValueError(f"unknown parameter type {kind!r} for {name}")
    return params


def default_trial(space: dict, defaults: dict) -> dict:
    """The config.yaml settings, in the form Optuna's enqueue_trial expects.
    Settings outside the search space (such as the SVM's gamma "scale") are
    left out, and Optuna draws them for that trial instead."""
    out = {}
    for name, spec in space.items():
        if name not in defaults:
            continue
        value = defaults[name]
        if spec["type"] == "categorical":
            if value in spec["choices"]:
                out[name] = _label(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) \
                and spec["low"] <= value <= spec["high"]:
            out[name] = value
    return out


# --------------------------------------------------------------------- one study
def tune_fold(table: pd.DataFrame, fold, spec: dict, family: str, n_trials: int | None = None,
              seed: int | None = None) -> dict:
    """Tune one model in one fold. Returns the best settings, their tuning-year
    MCC, the MCC of the config.yaml settings (trial 0), and every trial."""
    import optuna
    cfg = load_config()
    n_trials = cfg["tuning"]["n_trials"] if n_trials is None else n_trials
    seed = cfg["seed"] if seed is None else seed
    space = cfg["tuning"]["spaces"][family]
    cols = list(spec["features"])

    inner = fold.tuning()
    fit = inner.select(table, "train")
    fit = fit[fit["up"].notna() & fit[cols].notna().all(axis=1)]
    tune = inner.select(table, "test")
    tune = tune[tune["up"].notna() & tune[cols].notna().all(axis=1)]
    X_fit, y_fit = fit[cols], fit["up"].astype(int)
    X_tune, y_tune = tune[cols], tune["up"].astype(int).to_numpy()

    def objective(trial):
        params = suggest(trial, space)
        try:
            model = spec["make"](inner, **params).fit(X_fit, y_fit)
        except ValueError as exc:          # e.g. a sample with one direction only
            trial.set_user_attr("error", str(exc))
            return -1.0
        pred = (np.asarray(model.predict_proba(X_tune)) >= 0.5).astype(int)
        return float(classification_metrics(y_tune, pred)["mcc"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed + fold.test_year))
    study.enqueue_trial(default_trial(space, cfg["models"][family]))
    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    decode = {name: (json.loads if s["type"] == "categorical" else (lambda v: v))
              for name, s in space.items()}
    trials = [{"number": t.number, "mcc": t.value,
               "params": {k: decode[k](v) for k, v in t.params.items()}} for t in study.trials]
    return {"params": trials[study.best_trial.number]["params"], "mcc": study.best_value,
            "default_mcc": trials[0]["mcc"], "fit_days": len(y_fit), "tune_days": len(y_tune),
            "tuning_set": f"{inner.test[0].date()} to {inner.test[-1].date()}",
            "seconds": round(time.time() - start, 1), "trials": trials}


# ------------------------------------------------------------- across the folds
def retune_year_for(test_year: int, retune_years) -> int:
    """The retune year whose settings the fold of test_year uses."""
    earlier = [y for y in sorted(retune_years) if y <= test_year]
    if not earlier:
        raise ValueError(f"no retune year at or before {test_year}")
    return earlier[-1]


def _setup(family: str) -> dict:
    """Everything a cached result depends on, apart from the data (which the
    manifest check fixes) and the code (rerun with --retune after changing it)."""
    cfg = load_config()
    return {"space": cfg["tuning"]["spaces"][family], "n_trials": cfg["tuning"]["n_trials"],
            "seed": cfg["seed"], "model_settings": cfg["models"][family],
            "tuning_years": cfg["splits"]["tuning_years"], "labels": cfg["labels"]}


def tuned_settings(table: pd.DataFrame, folds, name: str, spec: dict, retune: bool = False,
                   cache_dir: Path | None = CACHE_DIR, stamp: str = "unknown", log=print) -> dict:
    """{test_year: settings} for every fold, tuning (or reusing the cache) in
    each retune year that the folds need."""
    family = spec["tune"]
    years = load_config()["tuning"]["retune_years"]
    by_fold = {f.test_year: f for f in folds}
    needed = sorted({retune_year_for(y, years) for y in by_fold})
    missing = [y for y in needed if y not in by_fold]
    if missing:
        raise ValueError(f"{name}: retune years {missing} have no fold to tune in")

    setup = _setup(family)
    path = None if cache_dir is None else Path(cache_dir) / f"{name}.json"
    cache = {}
    if path is not None and path.exists() and not retune:
        saved = json.loads(path.read_text())
        if saved.get("setup") == json.loads(json.dumps(setup)):
            cache = saved["years"]

    chosen = {}
    for year in needed:
        if str(year) in cache:
            entry = cache[str(year)]
            log(f"  {name} {year}: cached (tuning-year MCC {entry['mcc']:+.4f})")
        else:
            entry = tune_fold(table, by_fold[year], spec, family)
            entry["git_commit"] = stamp
            cache[str(year)] = entry
            log(f"  {name} {year}: tuned in {entry['seconds']:.0f}s, tuning-year MCC "
                f"{entry['mcc']:+.4f} (config settings {entry['default_mcc']:+.4f})")
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"model": name, "family": family, "setup": setup,
                                            "years": cache}, indent=2, default=str) + "\n")
        chosen[year] = entry["params"]
    return {y: chosen[retune_year_for(y, years)] for y in by_fold}


def with_settings(spec: dict, settings: dict) -> dict:
    """A copy of spec whose make() builds each fold's model with its tuned settings."""
    base = spec["make"]
    return {**spec, "make": lambda fold: base(fold, **settings[fold.test_year])}


def settings_table(all_settings: dict) -> pd.DataFrame:
    """One row per model and fold: the settings it used (for results/<version>/)."""
    rows = [{"model": name, "test_year": year, **{k: json.dumps(v) if isinstance(v, list) else v
                                                  for k, v in params.items()}}
            for name, by_year in all_settings.items() for year, params in sorted(by_year.items())]
    return pd.DataFrame(rows)
