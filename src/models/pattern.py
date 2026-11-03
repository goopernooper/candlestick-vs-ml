"""Similarity- and pattern-based models.

KNNModel (week 4): for each day to predict, find the k most similar candle
windows among the training days and predict the share of them that were
followed by an up day.

Pattern miner (week 5): count_contexts() tallies, for every sequence of 1 to 3
consecutive daily symbols, how often it was followed by an up day. The model
built on these counts follows in the next tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import load_config, symbol_label  # noqa: E402
from models.baselines import Model  # noqa: E402


class KNNModel(Model):
    """k-nearest neighbours with Euclidean distance on standardized features.

    Standardizing puts every column on the same footing before distances are
    measured, so no single column (say, the gap) dominates just because it varies
    more. The means and standard deviations come from the training days only,
    so the test year never influences the scaling.

    predict_proba is the fraction of the k neighbours that were up days, so it
    moves in steps of 1/k. Exact ties at 0.5 count as a prediction of up.
    """
    name = "knn"

    def __init__(self, k: int | None = None, n_jobs: int | None = None):
        cfg = load_config()["models"]["knn"]
        self.k = cfg["k"] if k is None else k
        self.n_jobs = cfg["n_jobs"] if n_jobs is None else n_jobs

    def fit(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training data contains only one direction")
        Z = np.asarray(X, dtype=float)
        self.features_ = list(X.columns)
        self.mean_ = Z.mean(axis=0)
        scale = Z.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)   # a constant column stays at 0
        self.knn_ = KNeighborsClassifier(n_neighbors=min(self.k, len(Z)), metric="euclidean",
                                         algorithm="brute", n_jobs=self.n_jobs)
        self.knn_.fit((Z - self.mean_) / self.scale_, y)
        return self

    def predict_proba(self, X):
        if list(X.columns) != self.features_:
            raise ValueError("prediction columns differ from the training columns")
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        up = list(self.knn_.classes_).index(1)
        return self.knn_.predict_proba(Z)[:, up]


# ---------------------------------------------------------------------------
# Pattern miner: counting contexts
# ---------------------------------------------------------------------------
def context_columns(length: int) -> list[str]:
    """Columns forming a context of `length` days, oldest first.
    length 3 -> ["b_symbol_lag2", "b_symbol_lag1", "b_symbol"]."""
    return [f"b_symbol_lag{k}" for k in range(length - 1, 0, -1)] + ["b_symbol"]


def count_contexts(X: pd.DataFrame, y, max_len: int | None = None) -> dict:
    """Count every context of 1 to max_len consecutive symbols in the training data.

    X holds b_symbol and its lags for the training days (all stocks pooled), y the
    labels (1 = next day up, 0 = down) for the same days. Returns {length: table}
    where each table is indexed by the context (oldest symbol first) and has
        n    how many training days ended with this context
        ups  how many of those were followed by an up day
    A day counts toward every length for which its symbols exist, so a day whose
    two-days-ago symbol is missing still counts for contexts of length 1 and 2.
    """
    max_len = load_config()["models"]["pattern_miner"]["max_context"] if max_len is None else max_len
    y = np.asarray(y, dtype=int)
    if len(y) != len(X):
        raise ValueError("X and y have different lengths")
    counts = {}
    for length in range(1, max_len + 1):
        cols = context_columns(length)
        d = X[cols].astype("float").assign(up=y).dropna(subset=cols)
        d[cols] = d[cols].astype(int)
        counts[length] = d.groupby(cols)["up"].agg(n="size", ups="sum").astype(int)
    return counts


def context_table(counts: dict) -> pd.DataFrame:
    """All counted contexts in one readable table, with symbol labels."""
    rows = []
    for length, table in counts.items():
        for key, r in table.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"length": length,
                         "context": "  >  ".join(symbol_label(s) for s in key),
                         "symbols": " ".join(str(s) for s in key),
                         "n": int(r["n"]), "ups": int(r["ups"]),
                         "up_rate": r["ups"] / r["n"]})
    return pd.DataFrame(rows)
