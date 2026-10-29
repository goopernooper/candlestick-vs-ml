"""Tests for the LightGBM model, the encoding B window, and running ML models end to end."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from features import (B_CONTINUOUS, encoding_b_continuous, encoding_b_window,  # noqa: E402
                      window_columns)
from models.ml import LightGBMModel  # noqa: E402
from run import experiments, run_experiments  # noqa: E402
from splits import walk_forward  # noqa: E402

FAST = dict(n_estimators=30, min_child_samples=20)


def planted(n=4000, seed=0):
    """Feature x1 decides the direction 75% of the time; x2 is pure noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = np.where(rng.uniform(size=n) < 0.75, X["x1"] > 0, X["x1"] <= 0).astype(int)
    return X, y


# ------------------------------------------------------------------ LightGBM

def test_lightgbm_interface():
    X, y = planted()
    m = LightGBMModel(**FAST)
    assert m.fit(X, y) is m
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and ((p > 0) & (p < 1)).all()
    assert set(m.predict(X)) <= {0, 1}


def test_lightgbm_learns_a_real_signal():
    X, y = planted(seed=1)
    Xt, yt = planted(seed=2)
    acc = (LightGBMModel(**FAST).fit(X, y).predict(Xt) == yt).mean()
    assert 0.70 < acc < 0.80          # close to the 75% that's achievable


def test_lightgbm_is_reproducible():
    X, y = planted()
    a = LightGBMModel(seed=7, **FAST).fit(X, y).predict_proba(X)
    b = LightGBMModel(seed=7, **FAST).fit(X, y).predict_proba(X)
    c = LightGBMModel(seed=8, **FAST).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_lightgbm_uses_config_settings():
    from features import load_config
    m = LightGBMModel()
    assert m.params["n_estimators"] == load_config()["models"]["lightgbm"]["n_estimators"]
    assert m.seed == load_config()["seed"]


def test_lightgbm_refuses_mismatched_columns():
    X, y = planted()
    m = LightGBMModel(**FAST).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X[["x2", "x1"]])


def test_lightgbm_refuses_one_class_training():
    X, _ = planted()
    with pytest.raises(ValueError):
        LightGBMModel(**FAST).fit(X, np.ones(len(X), int))


# --------------------------------------------------------- encoding B window

def ohlc(n=300, seed=0, start="2002-01-02"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                         "Low": np.minimum(o, c) * 0.996, "Close": c},
                        index=pd.bdate_range(start, periods=n))


def test_window_columns_and_values():
    df = ohlc()
    w, cont = encoding_b_window(df, days=3), encoding_b_continuous(df)
    assert list(w.columns) == window_columns(3)
    for c in B_CONTINUOUS:
        pd.testing.assert_series_equal(w[c], cont[c])
        pd.testing.assert_series_equal(w[f"{c}_lag1"], cont[c].shift(1), check_names=False)
        pd.testing.assert_series_equal(w[f"{c}_lag2"], cont[c].shift(2), check_names=False)


def test_window_lags_never_mix_stocks():
    """In the pooled table, a stock's first day must not inherit the previous stock's candle."""
    loader = lambda t: ohlc(n=2000, seed={"AAA": 1, "BBB": 2}[t], start="2001-10-01")  # noqa: E731
    table = build_table(["AAA", "BBB"], feature_sets=["encoding_b_window"], loader=loader)
    for t in ["AAA", "BBB"]:
        own = encoding_b_window(loader(t)).loc[table.index.min():]
        mine = table[table["ticker"] == t][window_columns()]
        pd.testing.assert_frame_equal(mine, own.loc[mine.index], check_names=False)


# -------------------------------------------------------- end to end, quickly

def test_experiments_registry():
    spec = experiments()
    assert {"majority", "always_long", "previous_day", "random", "lightgbm_a", "lightgbm_b"} <= set(spec)
    assert len(spec["lightgbm_a"]["features"]) == 61
    assert spec["lightgbm_b"]["features"] == window_columns()


def test_ml_models_run_through_folds_alongside_baselines():
    cal = pd.bdate_range("2001-10-01", "2025-12-31")
    def loader(t):
        rng = np.random.default_rng({"AAA": 0, "BBB": 1, "CCC": 2}[t])
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(cal))))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(cal)))
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                             "Low": np.minimum(o, c) * 0.996, "Close": c}, index=cal)
    table = build_table(["AAA", "BBB", "CCC"], ["normalized_ohlc", "encoding_a", "encoding_b_window"], loader)
    folds = walk_forward(table.index)[:3]
    spec = experiments()
    fast = lambda fold: LightGBMModel(seed=fold.test_year, **FAST)  # noqa: E731
    specs = {"always_long": spec["always_long"],
             "lightgbm_a": {**spec["lightgbm_a"], "make": fast},
             "lightgbm_b": {**spec["lightgbm_b"], "make": fast}}
    preds = run_experiments(table, folds, specs)
    sizes = preds.groupby("model").size()
    assert sizes.nunique() == 1                            # everyone predicts the same stock-days
    assert preds["p_up"].between(0, 1).all()
    assert set(preds["test_year"]) == {2005, 2006, 2007}
    # on random-walk prices, no model should look meaningfully skilled
    for name, g in preds.dropna(subset=["up"]).groupby("model"):
        assert abs((g["pred"] == g["up"].astype(int)).mean() - 0.5) < 0.06, name
