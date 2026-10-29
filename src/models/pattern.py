"""Similarity- and pattern-based models.

KNNModel (this file, week 4): for each day to predict, find the k most similar
candle windows among the training days and predict the share of them that were
followed by an up day. The pattern miner joins this file in week 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import load_config  # noqa: E402
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
