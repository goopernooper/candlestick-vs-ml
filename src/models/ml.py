"""Machine learning models, wrapped in the project's shared interface.

Every model here has fit(X, y) and predict_proba(X) -> probability of an up day,
exactly like the baselines, so the runner and evaluation treat them the same.
Settings come from the `models` section of config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import load_config  # noqa: E402
from models.baselines import Model  # noqa: E402


class LightGBMModel(Model):
    """Gradient-boosted decision trees (LightGBM).

    Reproducible: the random seed comes from config.yaml, and LightGBM's
    deterministic mode is on with a fixed thread count.
    """
    name = "lightgbm"

    def __init__(self, seed: int | None = None, **overrides):
        cfg = load_config()
        self.params = {**cfg["models"]["lightgbm"], **overrides}
        self.seed = cfg["seed"] if seed is None else seed

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training data contains only one direction")
        self.model_ = LGBMClassifier(**self.params, random_state=self.seed, deterministic=True,
                                     force_row_wise=True, verbose=-1)
        self.model_.fit(X, y)
        self.features_ = list(X.columns)
        return self

    def predict_proba(self, X):
        if list(X.columns) != self.features_:
            raise ValueError("prediction columns differ from the training columns")
        return self.model_.predict_proba(X)[:, 1]
