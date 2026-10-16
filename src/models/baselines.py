"""The four naive baselines every model has to beat.

All models in this project share one interface:

    model.fit(X, y)          X: features for the training days, y: 1 (up) / 0 (down)
    model.predict_proba(X)   probability that each day's next return is up
    model.predict(X)         1 where that probability is at least 0.5, else 0

The baselines:

  majority       predicts whichever direction was more common in the training
                 data, with probability equal to its training share (e.g. 0.523)
  always_long    always predicts up (probability 1). Differs from majority only
                 if a training window has more down days than up days
  previous_day   predicts that tomorrow moves the same way as today. Needs the
                 column n_close (today's close vs. yesterday's, from
                 normalized_ohlc): up if n_close > 0, down if below 0, 0.5 if flat
  random         a coin flip: probability drawn uniformly from [0, 1] with a fixed
                 seed, so it is right about half the time and reproducible

None of them can see the future: previous_day uses only today's candle, and the
others use nothing from the test days at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import load_config  # noqa: E402


class Model:
    """Base class: subclasses implement fit and predict_proba."""
    name = "model"

    def fit(self, X: pd.DataFrame, y) -> "Model":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


class Majority(Model):
    name = "majority"

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("no training labels")
        self.p_up_ = float(y.mean())
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p_up_)


class AlwaysLong(Model):
    name = "always_long"

    def predict_proba(self, X):
        return np.ones(len(X))


class PreviousDay(Model):
    name = "previous_day"
    column = "n_close"

    def predict_proba(self, X):
        if self.column not in X:
            raise KeyError(f"previous_day needs the column {self.column!r} (from normalized_ohlc)")
        today = X[self.column].to_numpy(dtype=float)
        return np.select([today > 0, today < 0], [1.0, 0.0], 0.5)


class RandomGuess(Model):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.seed = load_config()["seed"] if seed is None else seed

    def fit(self, X, y):
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "rng_"):
            self.fit(None, [0])
        return self.rng_.uniform(0, 1, len(X))


BASELINES = {cls.name: cls for cls in [Majority, AlwaysLong, PreviousDay, RandomGuess]}
