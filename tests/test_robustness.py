"""Tests for the robustness checks (ANALYSIS_PLAN.md section 12)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labels import make_labels  # noqa: E402
from models.baselines import Majority  # noqa: E402
from models.ml import LightGBMModel  # noqa: E402
from models.pattern import KNNModel  # noqa: E402
from robustness import setting  # noqa: E402
from robustness_open import next_open_returns, with_next_open  # noqa: E402
from run import experiments, per_stock_settings, run_experiments  # noqa: E402
from splits import walk_forward  # noqa: E402
from tune import cached_settings  # noqa: E402

CAL = pd.bdate_range("2002-01-02", "2025-12-31")


# ---------------------------------------------------------------- per-stock training
def test_count_settings_are_scaled_per_stock():
    s = per_stock_settings({"k": 717, "min_child_samples": 86, "min_samples_leaf": 30, "min_count": 14,
                            "shrinkage_m": 808.5, "learning_rate": 0.05, "C": 3.0, "max_features": "sqrt"}, 22)
    assert s == {"k": 33, "min_child_samples": 5, "min_samples_leaf": 1, "min_count": 1,
                 "shrinkage_m": pytest.approx(808.5 / 22), "learning_rate": 0.05, "C": 3.0,
                 "max_features": "sqrt"}
    assert isinstance(s["k"], int) and isinstance(s["min_count"], int)
    assert per_stock_settings({"k": 10}, 22)["k"] == 1                   # floor


def opposite_stocks():
    """AAA goes up when x > 0, BBB when x < 0: a pooled model can't learn
    either, a per-stock model learns both."""
    rng = np.random.default_rng(0)
    rows = []
    for t, sign in [("AAA", 1), ("BBB", -1)]:
        x = rng.normal(size=len(CAL))
        up = (sign * x + rng.normal(0, 0.3, len(CAL)) > 0).astype(int)
        rows.append(pd.DataFrame({"ticker": t, "x": x, "up": pd.array(up, dtype="Int8"),
                                  "fwd_return": np.where(up == 1, 0.01, -0.01)}, index=CAL))
    return pd.concat(rows).sort_index(kind="stable")


def test_per_stock_training_fits_each_stock_separately():
    table = opposite_stocks()
    folds = walk_forward(table.index)[:2]
    spec = {"lgb": {"features": ["x"], "make": lambda fold: LightGBMModel(n_jobs=1, min_child_samples=20)}}
    pooled = run_experiments(table, folds, spec)
    single = run_experiments(table, folds, spec, per_stock=True)
    acc = lambda p: (p["pred"] == p["up"].astype(int)).mean()  # noqa: E731
    assert acc(single) > 0.8 and acc(pooled) < 0.65
    assert len(single) == len(pooled)
    assert (single.index == pooled.index).all() and (single["ticker"].to_numpy() == pooled["ticker"].to_numpy()).all()


def test_per_stock_caps_k_at_the_training_days():
    table = opposite_stocks()
    spec = {"knn": {"features": ["x"], "make": lambda fold: KNNModel(k=100_000, n_jobs=1)}}
    preds = run_experiments(table, walk_forward(table.index)[:1], spec, per_stock=True)
    assert preds["p_up"].between(0, 1).all()


# ------------------------------------------------------------------ saved settings
def test_cached_settings(tmp_path):
    spec = experiments()["knn_b3"]
    folds = walk_forward(pd.bdate_range("2002-01-02", "2022-12-30"))
    saved = {"setup": {"features": spec["features"], "labels": {"dead_zone_k": 0.1}},
             "years": {str(y): {"params": {"k": y - 2000}} for y in [2005, 2010, 2015, 2020]}}
    (tmp_path / "knn_b3.json").write_text(json.dumps(saved))
    s = cached_settings("knn_b3", spec, folds, cache_dir=tmp_path)
    assert s[2005] == {"k": 5} and s[2009] == {"k": 5} and s[2022] == {"k": 20}
    saved["setup"]["features"] = ["other"]
    (tmp_path / "knn_b3.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        cached_settings("knn_b3", spec, folds, cache_dir=tmp_path)


# ------------------------------------------------------------------------ dead zone
def test_dead_zone_settings():
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=pd.bdate_range("2002-01-02", periods=600))
    df = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c})
    none, main, wide = (make_labels(df, dead_zone_k=k) for k in [0.0, 0.1, 0.2])
    assert none["dead_zone"].sum() == 0
    assert 0 < main["dead_zone"].sum() < wide["dead_zone"].sum()
    labelled = main["up"].notna()
    assert (none.loc[labelled, "up"] == main.loc[labelled, "up"]).all()   # same direction where both label


def test_run_needs_frozen_settings_for_robustness(monkeypatch, capsys):
    import run
    monkeypatch.setattr(sys, "argv", ["run.py", "--version", "x", "--per-stock"])
    assert run.main() == 1
    assert "--frozen-settings" in capsys.readouterr().out


# ---------------------------------------------------------------------- next open
def test_next_open_returns():
    idx = pd.bdate_range("2005-01-03", periods=4)
    prices = pd.DataFrame({"Open": [10.0, 11.0, 12.0, 13.0], "Close": [10.5, 11.5, 11.0, 13.0]}, index=idx)
    r = next_open_returns(prices)
    assert r.iloc[0] == pytest.approx(11.5 / 11 - 1) and r.iloc[1] == pytest.approx(11 / 12 - 1)
    assert np.isnan(r.iloc[-1])
    preds = pd.DataFrame({"ticker": ["AAA", "BBB", "AAA"], "model": "m", "pred": 1,
                          "fwd_return": [0.5, 0.5, 0.5]}, index=pd.Index([idx[0], idx[0], idx[1]], name="date"))
    moved = with_next_open(preds, lambda t: prices if t == "AAA" else prices * 2)
    assert moved["fwd_return"].tolist() == pytest.approx([11.5 / 11 - 1, 11.5 / 11 - 1, 11 / 12 - 1])
    assert (moved.index == preds.index).all() and moved["ticker"].tolist() == ["AAA", "BBB", "AAA"]


def test_setting_names():
    assert setting({"dead_zone_k": 0.1}, 0.1) == "main"
    assert setting({}, 0.1) == "main"
    assert setting({"dead_zone_k": 0.0}, 0.1) == "dead zone 0x"
    assert setting({"dead_zone_k": 0.1, "per_stock": True}, 0.1) == "per-stock"


def test_holdout_run_guards(monkeypatch, capsys):
    """--holdout needs the holdout unlocked, and an unlocked holdout needs --holdout."""
    import run
    from features import load_config
    monkeypatch.setattr(sys, "argv", ["run.py", "--version", "x", "--holdout"])
    assert run.main() == 1
    assert "unlock_holdout: true" in capsys.readouterr().out
    monkeypatch.setitem(load_config()["splits"], "unlock_holdout", True)
    monkeypatch.setattr(sys, "argv", ["run.py", "--version", "x"])
    assert run.main() == 1
    assert "Use --holdout" in capsys.readouterr().out
