"""Tests for the classification metrics: hand-checked values, edge cases, grouping."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import classification_metrics, confusion, summarize  # noqa: E402


def test_confusion_by_hand():
    #        TP TP FN TN TN FP TP FP
    truth = [1, 1, 1, 0, 0, 0, 1, 0]
    guess = [1, 1, 0, 0, 0, 1, 1, 1]
    assert confusion(truth, guess) == (3, 2, 2, 1)


def test_metrics_by_hand():
    m = classification_metrics([1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 1])
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 2 / 4) / 2)
    assert m["mcc"] == pytest.approx((3 * 2 - 2 * 1) / np.sqrt(5 * 4 * 4 * 3))
    assert m["actual_up_share"] == 0.5 and m["predicted_up_share"] == pytest.approx(5 / 8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scikit_learn(seed):
    rng = np.random.default_rng(seed)
    t, p = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    m = classification_metrics(t, p)
    assert m["accuracy"] == pytest.approx(accuracy_score(t, p))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(t, p))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(t, p))


def test_constant_guess_scores_zero_mcc():
    """Always predicting up: accuracy equals the up share, but MCC is 0 and balanced accuracy 0.5."""
    truth = [1, 1, 1, 0, 1, 0, 1]
    m = classification_metrics(truth, [1] * 7)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["balanced_accuracy"] == 0.5 and m["mcc"] == 0.0


def test_perfect_and_inverted():
    t = [1, 0, 1, 0, 1]
    assert classification_metrics(t, t)["mcc"] == pytest.approx(1.0)
    inv = classification_metrics(t, [1 - x for x in t])
    assert inv["mcc"] == pytest.approx(-1.0) and inv["accuracy"] == 0


def test_dead_zone_days_are_skipped_but_counted():
    m = classification_metrics(pd.array([1, pd.NA, 0, pd.NA, 1], dtype="Int8"), [1, 0, 0, 1, 0])
    assert m["days"] == 3 and m["dead_zone_days"] == 2
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_no_labelled_days():
    m = classification_metrics(pd.array([pd.NA, pd.NA], dtype="Int8"), [1, 0])
    assert m["days"] == 0 and np.isnan(m["mcc"])


def make_preds():
    dates = pd.bdate_range("2005-01-03", "2006-12-29")
    rng = np.random.default_rng(3)
    rows = []
    for t in ["AAA", "BBB"]:
        up = rng.integers(0, 2, len(dates))
        for model, pred in [("always_long", np.ones(len(dates), int)), ("perfect", up)]:
            rows.append(pd.DataFrame({"ticker": t, "test_year": dates.year, "model": model,
                                      "up": pd.array(up, dtype="Int8"), "pred": pred}, index=dates))
    return pd.concat(rows)


def test_summarize_per_fold_and_per_ticker():
    preds = make_preds()
    per_fold = summarize(preds, by=["model", "test_year"])
    per_ticker = summarize(preds, by=["model", "ticker"])
    overall = summarize(preds)
    assert len(per_fold) == 4 and len(per_ticker) == 4 and len(overall) == 2
    assert (per_fold.loc[per_fold.model == "perfect", "mcc"] == 1).all()
    assert (per_ticker.loc[per_ticker.model == "always_long", "mcc"] == 0).all()
    # pooled days add up across folds
    assert overall.set_index("model")["days"].sum() == per_fold["days"].sum()


def test_summarize_refuses_holdout_dates():
    preds = make_preds()
    preds.index = preds.index + pd.DateOffset(years=18)   # shift into 2023-2024
    with pytest.raises(PermissionError):
        summarize(preds)
