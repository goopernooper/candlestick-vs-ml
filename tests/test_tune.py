"""Tests for hyperparameter tuning (tune.py) and its use in the runner."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from features import load_config, symbol_window_columns  # noqa: E402
from run import experiments, run_experiments  # noqa: E402
from splits import walk_forward  # noqa: E402
from tune import (default_trial, retune_year_for, settings_table, suggest, tune_fold,  # noqa: E402
                  tuned_settings, with_settings)

CAL = pd.bdate_range("2001-10-01", "2025-12-31")
STOCKS = ["AAA", "BBB", "CCC"]


def loader(ticker):
    rng = np.random.default_rng(STOCKS.index(ticker) + 10)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(CAL))))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(CAL)))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                         "Low": np.minimum(o, c) * 0.996, "Close": c}, index=CAL)


@pytest.fixture(scope="module")
def table():
    return build_table(STOCKS, ["normalized_ohlc", "encoding_b_symbol_window"], loader)


@pytest.fixture(scope="module")
def folds(table):
    return walk_forward(table.index)


@pytest.fixture(scope="module")
def miner():
    return experiments()["pattern_miner"]


# ---------------------------------------------------------------- search spaces
class FixedTrial:
    """Stands in for an Optuna trial: returns the low end, or the first choice."""
    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_int(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[-1]


def test_every_tuned_family_has_a_space_and_defaults():
    cfg = load_config()
    families = {sp["tune"] for sp in experiments().values() if "tune" in sp}
    assert families <= set(cfg["tuning"]["spaces"])
    for family, space in cfg["tuning"]["spaces"].items():
        assert family in cfg["models"]
        assert set(space) <= set(cfg["models"][family]), family   # tuned names match config names
    assert cfg["tuning"]["n_trials"] == 20
    assert cfg["tuning"]["retune_years"] == [2005, 2010, 2015, 2020, 2025]


def test_suggest_decodes_categorical_lists():
    space = load_config()["tuning"]["spaces"]["mlp"]
    p = suggest(FixedTrial(), space)
    assert p["hidden_layer_sizes"] == [64, 32] and p["alpha"] == space["alpha"]["low"]


def test_default_trial_uses_config_settings_inside_the_space():
    cfg = load_config()
    assert default_trial(cfg["tuning"]["spaces"]["pattern_miner"], cfg["models"]["pattern_miner"]) == \
        {"shrinkage_m": cfg["models"]["pattern_miner"]["shrinkage_m"],
         "min_count": cfg["models"]["pattern_miner"]["min_count"]}
    svm = default_trial(cfg["tuning"]["spaces"]["svm"], cfg["models"]["svm"])
    assert svm == {"C": cfg["models"]["svm"]["C"]}               # gamma "scale" isn't in the range
    rf = default_trial(cfg["tuning"]["spaces"]["random_forest"], cfg["models"]["random_forest"])
    assert rf["max_features"] == json.dumps("sqrt")
    mlp = default_trial(cfg["tuning"]["spaces"]["mlp"], cfg["models"]["mlp"])
    assert json.loads(mlp["hidden_layer_sizes"]) == cfg["models"]["mlp"]["hidden_layer_sizes"]


def test_retune_year_mapping():
    years = [2005, 2010, 2015, 2020, 2025]
    assert [retune_year_for(y, years) for y in [2005, 2009, 2010, 2014, 2022, 2025]] == \
        [2005, 2005, 2010, 2010, 2020, 2025]
    with pytest.raises(ValueError):
        retune_year_for(2004, years)


# --------------------------------------------------------------------- one study
def test_tune_fold(table, folds, miner):
    out = tune_fold(table, folds[0], miner, "pattern_miner", n_trials=6)
    cfg = load_config()["models"]["pattern_miner"]
    assert len(out["trials"]) == 6
    assert out["trials"][0]["params"] == {"shrinkage_m": cfg["shrinkage_m"], "min_count": cfg["min_count"]}
    assert out["mcc"] >= out["default_mcc"]                      # never worse than config settings
    # the last year of the 2005 training window, minus the purged last day
    # (this fake calendar has every weekday, so it starts on Jan 1)
    assert out["tuning_set"] == "2004-01-01 to 2004-12-30"
    assert out["fit_days"] > 0 and out["tune_days"] > 0


def test_tuning_is_reproducible(table, folds, miner):
    a = tune_fold(table, folds[0], miner, "pattern_miner", n_trials=6)
    b = tune_fold(table, folds[0], miner, "pattern_miner", n_trials=6)
    assert a["trials"] == b["trials"]


def test_tuning_never_sees_the_test_year(table, folds, miner):
    """Scramble every row from the first test day on: the tuning must not change."""
    fold = folds[0]
    scrambled = table.copy()
    later = scrambled.index >= fold.test[0]
    rng = np.random.default_rng(0)
    for c in symbol_window_columns():
        scrambled.loc[later, c] = rng.integers(0, 20, later.sum())
    scrambled.loc[later, "up"] = pd.array(rng.integers(0, 2, later.sum()), dtype="Int8")
    a = tune_fold(table, fold, miner, "pattern_miner", n_trials=6)
    b = tune_fold(scrambled, fold, miner, "pattern_miner", n_trials=6)
    assert a["trials"] == b["trials"]


# ------------------------------------------------------------- across the folds
def test_tuned_settings_cover_every_fold_and_are_cached(table, folds, miner, tmp_path):
    logs = []
    s = tuned_settings(table, folds, "pattern_miner", miner, cache_dir=tmp_path, log=logs.append)
    assert sorted(s) == list(range(2005, 2023))
    assert s[2005] == s[2009] and s[2010] == s[2014] and s[2020] == s[2022]
    assert sum("tuned" in line for line in logs) == 4                     # 2005, 2010, 2015, 2020

    saved = json.loads((tmp_path / "pattern_miner.json").read_text())
    assert sorted(saved["years"]) == ["2005", "2010", "2015", "2020"]
    assert all(len(e["trials"]) == 20 for e in saved["years"].values())   # the full budget

    logs.clear()
    again = tuned_settings(table, folds, "pattern_miner", miner, cache_dir=tmp_path, log=logs.append)
    assert again == s and all("cached" in line for line in logs)

    saved["setup"]["n_trials"] = 5                                         # a different setup ...
    (tmp_path / "pattern_miner.json").write_text(json.dumps(saved))
    logs.clear()
    tuned_settings(table, folds, "pattern_miner", miner, cache_dir=tmp_path, log=logs.append)
    assert sum("tuned" in line for line in logs) == 4                     # ... is tuned again


def test_retune_ignores_the_cache(table, folds, miner, tmp_path):
    tuned_settings(table, folds[:1], "pattern_miner", miner, cache_dir=tmp_path, log=lambda _: None)
    logs = []
    tuned_settings(table, folds[:1], "pattern_miner", miner, cache_dir=tmp_path, retune=True,
                   log=logs.append)
    assert "tuned" in logs[0]


def test_tuned_settings_reach_the_models(table, folds, miner):
    settings = {f.test_year: {"shrinkage_m": 5.0 + f.test_year % 3, "min_count": 7} for f in folds[:3]}
    spec = with_settings(miner, settings)
    m = spec["make"](folds[1])
    assert (m.m, m.min_count) == (settings[2006]["shrinkage_m"], 7)
    preds = run_experiments(table, folds[:3], {"pattern_miner": spec})
    assert set(preds["test_year"]) == {2005, 2006, 2007}
    t = settings_table({"pattern_miner": settings})
    assert list(t.columns) == ["model", "test_year", "shrinkage_m", "min_count"] and len(t) == 3


def test_untuned_specs_still_use_config_settings(folds):
    spec = experiments()
    cfg = load_config()["models"]
    assert spec["pattern_miner"]["make"](folds[0]).m == cfg["pattern_miner"]["shrinkage_m"]
    assert spec["knn_b3"]["make"](folds[0]).k == cfg["knn"]["k"]
    assert spec["lightgbm_a"]["make"](folds[0]).params == cfg["lightgbm"]
    assert spec["lightgbm_a"]["make"](folds[0], num_leaves=7).params["num_leaves"] == 7


def test_cache_is_tied_to_the_feature_columns(table, folds, miner, tmp_path):
    """Same name, different columns: tuned again, never reused (D-024)."""
    from models.baselines import Majority
    one = {"features": ["n_close"], "tune": "pattern_miner",
           "make": lambda fold, shrinkage_m=None, min_count=None: Majority()}
    two = {**one, "features": ["n_open", "n_close"]}
    tuned_settings(table, folds[:1], "same_name", one, cache_dir=tmp_path, log=lambda _: None)
    logs = []
    tuned_settings(table, folds[:1], "same_name", one, cache_dir=tmp_path, log=logs.append)
    assert "cached" in logs[0]
    tuned_settings(table, folds[:1], "same_name", two, cache_dir=tmp_path, log=logs.append)
    assert "tuned" in logs[1]


def test_committed_caches_match_the_models():
    """Every cache in results/tuning/ was tuned on the columns its model uses now."""
    cache = Path(__file__).resolve().parents[1] / "results" / "tuning"
    files = sorted(cache.glob("*.json"))
    if not files:
        pytest.skip("no committed tuning results")
    spec = experiments()
    for path in files:
        saved = json.loads(path.read_text())
        assert path.stem in spec, f"{path.name}: no such model"
        assert saved["setup"].get("features") == spec[path.stem]["features"], path.name
