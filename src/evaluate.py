"""Scoring predictions: classification metrics now, the trading backtest next.

Every evaluation starts from a predictions table with one row per stock per
test day and these columns:

    date (index)  the day the prediction is made, at that day's close
    ticker        which stock
    test_year     which walk-forward fold the day belongs to
    model         which model made the prediction
    p_up          the model's probability that the next return is up
    pred          1 (up) or 0 (down)
    up            what actually happened: 1, 0, or <NA> in the dead zone
    fwd_return    the actual next-day return (used by the backtest)

Classification metrics skip dead-zone days (up is <NA>); the backtest will not.

The metrics:
  accuracy           share of days called correctly
  balanced_accuracy  average of the hit rate on up days and on down days, so
                     always guessing "up" scores 0.5, not the share of up days
  mcc                Matthews correlation coefficient, from -1 (always wrong)
                     through 0 (no better than chance or a constant guess) to +1
                     (always right). The headline classification metric, because
                     it can't be inflated by guessing the more common direction
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import check_not_holdout  # noqa: E402

METRIC_COLUMNS = ["days", "dead_zone_days", "accuracy", "balanced_accuracy", "mcc",
                  "actual_up_share", "predicted_up_share"]


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    """(true positives, false positives, true negatives, false negatives), with up = positive."""
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    return tp, fp, tn, fn


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, balanced accuracy and MCC for days with a label (dead-zone days removed first)."""
    y_true = pd.array(y_true, dtype="Int8")
    y_pred = np.asarray(y_pred)
    labelled = ~np.asarray(y_true.isna())
    tp, fp, tn, fn = confusion(y_true[labelled], y_pred[labelled])
    n = tp + fp + tn + fn
    if n == 0:
        return {"days": 0, "dead_zone_days": int((~labelled).sum()), "accuracy": np.nan,
                "balanced_accuracy": np.nan, "mcc": np.nan,
                "actual_up_share": np.nan, "predicted_up_share": np.nan}
    hit_up = tp / (tp + fn) if tp + fn else np.nan        # recall on up days
    hit_down = tn / (tn + fp) if tn + fp else np.nan      # recall on down days
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "days": n,
        "dead_zone_days": int((~labelled).sum()),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": np.nanmean([hit_up, hit_down]),
        # A constant prediction makes the denominator zero; by convention MCC is then 0.
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
        "actual_up_share": (tp + fn) / n,
        "predicted_up_share": (tp + fp) / n,
    }


def summarize(preds: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """Classification metrics for each group of the predictions table.

    by=("model",)               one row per model, all test days pooled
    by=("model", "test_year")   one row per model per fold
    by=("model", "ticker")      one row per model per stock
    """
    check_not_holdout(preds.index)
    by = list(by)
    rows = []
    for key, g in preds.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **classification_metrics(g["up"], g["pred"])})
    return pd.DataFrame(rows, columns=by + METRIC_COLUMNS)
