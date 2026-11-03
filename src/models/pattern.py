"""Similarity- and pattern-based models.

KNNModel (week 4): for each day to predict, find the k most similar candle
windows among the training days and predict the share of them that were
followed by an up day.

Pattern miner (week 5): count_contexts() tallies, for every sequence of 1 to 3
consecutive daily symbols, how often it was followed by an up day, and
shrink() turns those counts into probabilities pulled toward the base rate so
rare contexts can't produce extreme estimates. PatternMiner predicts from the
longest context seen often enough in training, backing off to shorter contexts
and finally to the base rate: a variable-order Markov model (Begleiter,
El-Yaniv & Yona, 2004).
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
                         "up_rate": r["ups"] / r["n"],
                         **({"p_up": float(r["p_up"])} if "p_up" in table else {})})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pattern miner: shrinking toward the base rate
# ---------------------------------------------------------------------------
def shrink(counts: dict, base_rate: float, m: float | None = None) -> dict:
    """Add a p_up column to every context table:

        p_up = (ups + m * base_rate) / (n + m)

    base_rate is the training share of up days. The estimate always lies between
    the context's raw up rate and the base rate; the fewer times a context was
    seen relative to m, the closer it stays to the base rate. m = 0 gives the
    raw rate. Returns new tables; the counts passed in are not changed.
    """
    m = load_config()["models"]["pattern_miner"]["shrinkage_m"] if m is None else m
    if m < 0:
        raise ValueError("m must be zero or positive")
    if not 0 <= base_rate <= 1:
        raise ValueError("base_rate must be a probability")
    return {length: table.assign(p_up=(table["ups"] + m * base_rate) / (table["n"] + m))
            for length, table in counts.items()}


# ---------------------------------------------------------------------------
# Pattern miner: the model
# ---------------------------------------------------------------------------
class PatternMiner(Model):
    """Variable-order Markov model on daily candle symbols.

    fit: count every context of 1 to max_context symbols in the training days
    (all stocks pooled), shrink each context's up rate toward the training base
    rate, and keep the contexts seen at least min_count times.

    predict_proba: for each day, use the longest kept context that ends on that
    day (3 days, then 2, then 1); if none was kept, use the base rate. explain()
    reports which context length each prediction came from.

    X must hold b_symbol and its lags (features.symbol_window_columns()).
    """
    name = "pattern_miner"

    def __init__(self, max_context: int | None = None, m: float | None = None,
                 min_count: int | None = None):
        cfg = load_config()["models"]["pattern_miner"]
        self.max_context = cfg["max_context"] if max_context is None else max_context
        self.m = cfg["shrinkage_m"] if m is None else m
        self.min_count = cfg["min_count"] if min_count is None else min_count

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if len(y) == 0:
            raise ValueError("no training labels")
        self.base_rate_ = float(y.mean())
        shrunk = shrink(count_contexts(X, y, self.max_context), self.base_rate_, self.m)
        # keep only contexts seen often enough to trust
        self.tables_ = {length: t.loc[t["n"] >= self.min_count, "p_up"]
                        for length, t in shrunk.items()}
        return self

    def _lookup(self, X, length) -> np.ndarray:
        """Each row's estimate for its context of this length, or NaN if the
        context wasn't kept or the row is missing one of its symbols."""
        table = self.tables_[length]
        cols = context_columns(length)
        keys = X[cols].astype("float")
        missing = keys.isna().any(axis=1).to_numpy()
        keys = keys.fillna(-1).astype(int)          # -1 never matches a real symbol
        if table.empty:
            return np.full(len(X), np.nan)
        index = pd.MultiIndex.from_frame(keys) if length > 1 else pd.Index(keys.iloc[:, 0])
        values = table.reindex(index).to_numpy(dtype=float, copy=True)   # writable copy
        values[missing] = np.nan
        return values

    def explain(self, X) -> pd.DataFrame:
        """p_up and the context length used for each row (0 = base rate)."""
        p = np.full(len(X), np.nan)
        used = np.zeros(len(X), dtype=int)
        for length in range(self.max_context, 0, -1):
            v = self._lookup(X, length)
            take = np.isnan(p) & ~np.isnan(v)
            p[take], used[take] = v[take], length
        p[np.isnan(p)] = self.base_rate_
        return pd.DataFrame({"p_up": p, "context_length": used}, index=X.index)

    def predict_proba(self, X):
        return self.explain(X)["p_up"].to_numpy()

    def diagnostics(self, X) -> dict:
        """What the fitted model looks like and how it predicted X, for the
        results files: the training base rate, how many contexts of each length
        were kept, and the share of X's predictions that used each length
        (0 = base rate)."""
        used = self.explain(X)["context_length"]
        out = {"base_rate": self.base_rate_}
        for length in range(self.max_context, 0, -1):
            out[f"kept_len{length}"] = len(self.tables_[length])
        for length in range(self.max_context, -1, -1):
            out[f"used_len{length}"] = float((used == length).mean())
        return out
