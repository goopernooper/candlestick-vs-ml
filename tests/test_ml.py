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


# ------------------------------------------- random forest, SVM, MLP (week 6)
from models.ml import MLPModel, RandomForestModel, SVMModel  # noqa: E402

SKLEARN_FAST = {
    RandomForestModel: dict(n_estimators=50, min_samples_leaf=20),
    SVMModel: dict(max_train_rows=1500),
    MLPModel: dict(hidden_layer_sizes=[16], max_iter=100),
}
SKLEARN = list(SKLEARN_FAST)


def fast(cls, **kw):
    return cls(**{**SKLEARN_FAST[cls], **kw})


@pytest.mark.parametrize("cls", SKLEARN)
def test_sklearn_interface(cls):
    X, y = planted()
    m = fast(cls)
    assert m.fit(X, y) is m
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
    assert set(m.predict(X)) <= {0, 1}


@pytest.mark.parametrize("cls", SKLEARN)
def test_sklearn_learns_a_real_signal(cls):
    X, y = planted(seed=1)
    Xt, yt = planted(seed=2)
    acc = (fast(cls).fit(X, y).predict(Xt) == yt).mean()
    assert 0.70 < acc < 0.80, cls.name          # close to the 75% that's achievable


@pytest.mark.parametrize("cls", SKLEARN)
def test_sklearn_finds_nothing_in_noise(cls):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(4000, 4)), columns=list("abcd"))
    y, yt = rng.integers(0, 2, 4000), rng.integers(0, 2, 4000)
    Xt = pd.DataFrame(rng.normal(size=(4000, 4)), columns=list("abcd"))
    acc = (fast(cls).fit(X, y).predict(Xt) == yt).mean()
    assert abs(acc - 0.5) < 0.04, cls.name


@pytest.mark.parametrize("cls", SKLEARN)
def test_sklearn_is_reproducible(cls):
    X, y = planted(n=3000)
    a = fast(cls, seed=7).fit(X, y).predict_proba(X)
    b = fast(cls, seed=7).fit(X, y).predict_proba(X)
    c = fast(cls, seed=8).fit(X, y).predict_proba(X)    # the SVM's seed picks its sample
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("cls", SKLEARN)
def test_sklearn_predicts_each_day_on_its_own(cls):
    """A day's prediction can't depend on which other test days are passed in:
    scaling uses the training days only."""
    X, y = planted(seed=4)
    Xt, _ = planted(seed=5)
    m = fast(cls).fit(X, y)
    np.testing.assert_allclose(m.predict_proba(Xt)[:100], m.predict_proba(Xt.iloc[:100]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.predict_proba(Xt.iloc[100:200] * 10)[:5],
                               m.predict_proba((Xt * 10).iloc[100:105]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls", SKLEARN)
def test_sklearn_checks(cls):
    X, y = planted(n=2000)
    m = fast(cls).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X[["x2", "x1"]])
    with pytest.raises(ValueError):
        fast(cls).fit(X, np.ones(len(X), int))


@pytest.mark.parametrize("cls,key", [(RandomForestModel, "random_forest"), (SVMModel, "svm"), (MLPModel, "mlp")])
def test_sklearn_uses_config_settings(cls, key):
    from features import load_config
    cfg = load_config()
    m = cls()
    assert m.params == cfg["models"][key] and m.seed == cfg["seed"]


def test_svm_caps_training_rows():
    X, y = planted(n=5000)
    m = SVMModel(max_train_rows=1200).fit(X, y)
    assert m.n_train_ == 1200 and m.diagnostics(X)["train_rows"] == 1200
    assert SVMModel(max_train_rows=None).fit(X.iloc[:800], y[:800]).n_train_ == 800
    assert SVMModel(max_train_rows=10_000).fit(X.iloc[:800], y[:800]).n_train_ == 800
    from features import load_config
    assert load_config()["models"]["svm"]["max_train_rows"] == 20000


def test_svm_score_agrees_with_its_decision_function():
    X, y = planted(n=3000)
    m = fast(SVMModel).fit(X, y)
    d, p = m.decision_function(X), m.predict_proba(X)
    assert ((p >= 0.5) == (d >= 0)).all()                         # same predictions as the SVM
    assert (m.predict(X) == m.model_.predict(np.asarray(X))).mean() > 0.999
    order = np.argsort(d)
    assert (np.diff(p[order]) >= 0).all()                          # same ranking of confidence


def test_mlp_records_convergence():
    X, y = planted(n=2000)
    m = MLPModel(max_iter=2, early_stopping=False).fit(X, y)
    assert m.converged_ is False and m.diagnostics(X)["epochs"] == 2
    assert fast(MLPModel).fit(X, y).converged_ is True


def test_sklearn_models_run_through_folds():
    cal = pd.bdate_range("2001-10-01", "2025-12-31")
    def loader(t):
        rng = np.random.default_rng({"AAA": 0, "BBB": 1}[t])
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(cal))))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(cal)))
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004,
                             "Low": np.minimum(o, c) * 0.996, "Close": c}, index=cal)
    table = build_table(["AAA", "BBB"], ["normalized_ohlc", "encoding_b_window"], loader)
    folds = walk_forward(table.index)[:2]
    cols = window_columns()
    specs = {"always_long": experiments()["always_long"]}
    for cls in SKLEARN:
        specs[cls.name] = {"features": cols, "make": lambda fold, cls=cls: fast(cls, seed=fold.test_year)}
    diag = []
    preds = run_experiments(table, folds, specs, diagnostics=diag)
    assert preds.groupby("model").size().nunique() == 1
    assert preds["p_up"].between(0, 1).all()
    assert {d["model"] for d in diag} == {"svm", "mlp"}
    for name, g in preds.dropna(subset=["up"]).groupby("model"):
        assert abs((g["pred"] == g["up"].astype(int)).mean() - 0.5) < 0.07, name
