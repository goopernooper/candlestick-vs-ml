"""Tests for the coverage curves (coverage.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coverage import coverage_table, cutoff, plot, validation_folds  # noqa: E402
from splits import walk_forward  # noqa: E402


def test_cutoff():
    conf = np.arange(100) / 200                     # 0 to 0.495
    assert cutoff(conf, 1.0) == -np.inf
    assert (conf >= cutoff(conf, 0.25)).mean() == pytest.approx(0.25, abs=0.01)
    assert (conf >= cutoff(conf, 0.10)).mean() == pytest.approx(0.10, abs=0.01)
    assert cutoff(np.full(50, 0.02), 0.10) == -np.inf     # can't rank: acts on every day


def test_validation_folds_train_and_test_before_the_test_year():
    folds = walk_forward(pd.bdate_range("2002-01-02", "2025-12-31"))
    for f, v in zip(folds, validation_folds(folds)):
        assert v.test_year == f.test_year                           # named after the outer fold
        assert set(v.test.year) == {f.test_year - 1}                # predicts the validation year
        assert v.train.max() < v.test.min() and v.test.max() < f.test.min()
        assert len(v.purged) == f.horizon


def preds(model, years, rng, informative, shift=0.0, n=3000):
    """Confidence tells you how likely the call is right, when informative."""
    rows = []
    for year in years:
        conf = rng.uniform(0, 0.5, n)
        p_right = 0.5 + (conf if informative else 0.0)
        up = rng.integers(0, 2, n)
        right = rng.random(n) < p_right
        pred = np.where(right, up, 1 - up)
        p_up = np.where(pred == 1, 0.5 + conf + shift, 0.5 - conf - shift).clip(0, 1)
        rows.append(pd.DataFrame({"model": model, "test_year": year, "p_up": p_up, "pred": pred,
                                  "up": pd.array(up, dtype="Int8")}))
    return pd.concat(rows, ignore_index=True)


def test_an_edge_in_confident_days_shows_as_coverage_falls():
    rng = np.random.default_rng(0)
    years = [2005, 2006, 2007]
    val = pd.concat([preds("sharp", years, rng, True), preds("blunt", years, rng, False)])
    test = pd.concat([preds("sharp", years, rng, True), preds("blunt", years, rng, False)])
    t = coverage_table(val, test).set_index(["model", "target_coverage"])
    sharp = t.loc["sharp", "mcc"]
    assert sharp.loc[0.10] > sharp.loc[0.25] > sharp.loc[0.5] > sharp.loc[1.0] > 0.2
    assert t.loc["blunt", "mcc"].abs().max() < 0.06
    for model in ["sharp", "blunt"]:
        cov = t.loc[model, "coverage"]
        assert cov.loc[1.0] == 1 and cov.loc[0.10] == pytest.approx(0.10, abs=0.02)


def test_cutoffs_come_from_the_validation_year_not_the_test_year():
    """If test-year confidences run higher than validation's, more days pass:
    coverage achieved exceeds the target, as it would in practice."""
    rng = np.random.default_rng(1)
    val = preds("m", [2005], rng, False)
    test = preds("m", [2005], rng, False, shift=0.1)
    t = coverage_table(val, test).set_index("target_coverage")
    assert t.loc[0.25, "coverage"] > 0.35


def test_constant_models_stay_at_full_coverage():
    n = 500
    frame = pd.DataFrame({"model": "always_long", "test_year": 2005, "p_up": np.full(n, 0.53),
                          "pred": 1, "up": pd.array(np.arange(n) % 2, dtype="Int8")})
    other = frame.assign(p_up=0.52)                                 # a slightly different constant
    t = coverage_table(other, frame)
    assert (t["coverage"] == 1).all() and (t["mcc"] == 0).all()


def test_missing_validation_year_is_an_error():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        coverage_table(preds("m", [2005], rng, True), preds("m", [2006], rng, True))


def test_plot_writes_a_figure(tmp_path):
    rng = np.random.default_rng(3)
    years = [2005, 2006]
    val = pd.concat([preds(m, years, rng, i == 0) for i, m in enumerate(["lightgbm_b", "reversal", "x"])])
    test = pd.concat([preds(m, years, rng, i == 0) for i, m in enumerate(["lightgbm_b", "reversal", "x"])])
    plot(coverage_table(val, test), tmp_path / "c.png", "test")
    assert (tmp_path / "c.png").stat().st_size > 10_000


# --------------------------------------------- against reversal on the same days
from coverage import bootstrap_vs_benchmark, same_days  # noqa: E402


def dated(seed=5, days=600, stocks=8, edge=0.15):
    """Test and validation tables with dates and tickers. 'reversal' is a coin
    flip at constant confidence. 'copycat' copies reversal's call and is
    confident on random days. 'skilled' is right with probability 0.5 + edge on
    its confident days (confidence above 0.3) and a coin flip otherwise."""
    rng = np.random.default_rng(seed)
    out = []
    for year in [2005, 2006]:
        idx = pd.Index(pd.bdate_range(f"{year}-01-03", periods=days), name="date")
        rows = []
        for t in range(stocks):
            up = rng.integers(0, 2, days)
            rev = rng.integers(0, 2, days)
            conf = rng.uniform(0, 0.5, days)
            right = rng.random(days) < np.where(conf > 0.3, 0.5 + edge, 0.5)
            skilled = np.where(right, up, 1 - up)
            for model, pred, c in [("reversal", rev, np.full(days, 0.5)), ("copycat", rev, conf),
                                   ("skilled", skilled, conf)]:
                rows.append(pd.DataFrame({"ticker": f"S{t}", "model": model, "test_year": year,
                                          "pred": pred, "p_up": np.where(pred == 1, 0.5 + c, 0.5 - c),
                                          "up": pd.array(up, dtype="Int8")}, index=idx))
        out.append(pd.concat(rows))
    return out[0].assign(test_year=2006), out[1]          # validation for the 2006 fold, then its test year


def test_same_days_lines_up_stock_days():
    val, test = dated()
    sel = test[(test["model"] == "skilled")].iloc[::7]
    b = same_days(sel, test, "reversal")
    assert len(b) == len(sel) and (b["ticker"].to_numpy() == sel["ticker"].to_numpy()).all()
    assert (b.index == sel.index).all() and (b["model"] == "reversal").all()


def test_benchmark_on_the_same_days():
    val, test = dated()
    t = coverage_table(val, test).set_index(["model", "target_coverage"])
    # copycat makes reversal's calls, so on its chosen days the two agree exactly
    assert t.loc[("copycat", 0.25), "mcc_minus_benchmark"] == pytest.approx(0, abs=1e-12)
    assert t.loc[("skilled", 0.25), "mcc_minus_benchmark"] > 0.1
    assert (t.loc["reversal", "mcc_minus_benchmark"] == 0).all()


def test_bootstrap_intervals_separate_skill_from_copying():
    val, test = dated()
    b = bootstrap_vs_benchmark(val, test, reps=300, seed=1).set_index(["model", "target_coverage"])
    assert "reversal" not in b.index.get_level_values("model")
    assert b.loc[("skilled", 0.25), "diff_low"] > 0
    assert b.loc[("copycat", 0.25), ["diff_low", "diff_high"]].abs().max() < 1e-12
    point = coverage_table(val, test).set_index(["model", "target_coverage"])
    assert b.loc[("skilled", 0.10), "diff"] == pytest.approx(point.loc[("skilled", 0.10), "mcc_minus_benchmark"])
    assert (b["mcc_low"] <= b["mcc"]).all() and (b["mcc"] <= b["mcc_high"]).all()
