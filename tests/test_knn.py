"""Tests for k-NN: in-fold standardization, neighbour voting by hand, and the runner setup."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_table  # noqa: E402
from features import load_config, window_columns  # noqa: E402
from models.pattern import KNNModel  # noqa: E402
from run import experiments, run_experiments  # noqa: E402
from splits import walk_forward  # noqa: E402


def planted(n=3000, seed=0, noise_scale=1.0):
    """x1 (small scale) decides the direction 75% of the time; x2 is noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n) * 0.01, "x2": rng.normal(size=n) * noise_scale})
    y = np.where(rng.uniform(size=n) < 0.75, X["x1"] > 0, X["x1"] <= 0).astype(int)
    return X, y


def test_config_settings():
    cfg = load_config()["models"]["knn"]
    assert cfg["windows"] == [3, 5] and cfg["k"] == 100
    assert KNNModel().k == 100


def test_interface_and_probability_steps():
    X, y = planted()
    m = KNNModel(k=50).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p * 50, np.round(p * 50))       # a share of 50 neighbours


def test_probability_is_share_of_up_neighbours_by_hand():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]})
    y = [1, 1, 0, 0, 0, 1]
    m = KNNModel(k=3).fit(X, y)
    # after standardizing, the 3 nearest to 0.5 are the points 0, 1, 2 -> 2 of 3 up
    # the 3 nearest to 11.5 are 10, 11, 12 -> 1 of 3 up
    assert m.predict_proba(pd.DataFrame({"x": [0.5, 11.5]})).tolist() == pytest.approx([2 / 3, 1 / 3])


def test_standardization_uses_training_days_only():
    X, y = planted()
    m = KNNModel(k=25).fit(X, y)
    np.testing.assert_allclose(m.mean_, X.mean().to_numpy())
    np.testing.assert_allclose(m.scale_, X.std(ddof=0).to_numpy())
    shifted = X + 100                                   # a test year that looks nothing like training
    m.predict_proba(shifted)
    np.testing.assert_allclose(m.mean_, X.mean().to_numpy())   # unchanged by the test data


def test_standardization_stops_a_large_column_from_dominating():
    """x2 is noise on a scale 1,000 times bigger than the informative x1. Without
    standardization the neighbours would be chosen almost entirely by x2."""
    X, y = planted(seed=1, noise_scale=10.0)
    Xt, yt = planted(seed=2, noise_scale=10.0)
    acc = (KNNModel(k=50).fit(X, y).predict(Xt) == yt).mean()
    assert acc > 0.65


def test_reproducible_and_column_checked():
    X, y = planted()
    a = KNNModel(k=30).fit(X, y).predict_proba(X)
    b = KNNModel(k=30).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        KNNModel(k=30).fit(X, y).predict_proba(X[["x2", "x1"]])


def test_k_larger_than_training_set_is_clipped():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    p = KNNModel(k=100).fit(X, [1, 0, 1]).predict_proba(X)
    assert np.allclose(p, 2 / 3)                        # every point sees all three neighbours


def test_window_columns_and_registry():
    assert window_columns(5)[:12] == window_columns(3)  # the 5-day window extends the 3-day one
    spec = experiments()
    assert spec["knn_b3"]["features"] == window_columns(3)
    assert spec["knn_b5"]["features"] == window_columns(5) and len(window_columns(5)) == 20


def test_knn_runs_through_folds():
    cal = pd.bdate_range("2001-10-01", "2025-12-31")
    def loader(t):
        rng = np.random.default_rng({"AAA": 0, "BBB": 1}[t])
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(cal))))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(cal)))
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                             "Low": np.minimum(o, c) * 0.996, "Close": c}, index=cal)
    table = build_table(["AAA", "BBB"], ["normalized_ohlc", "encoding_b_window"], loader)
    assert table.columns.is_unique and set(window_columns(5)) <= set(table.columns)
    spec = experiments()
    preds = run_experiments(table, walk_forward(table.index)[:2],
                            {k: spec[k] for k in ["always_long", "knn_b3", "knn_b5"]})
    assert preds.groupby("model").size().nunique() == 1
    for name, g in preds[preds["model"].str.startswith("knn")].dropna(subset=["up"]).groupby("model"):
        assert abs((g["pred"] == g["up"].astype(int)).mean() - 0.5) < 0.06, name   # random walk: no skill
