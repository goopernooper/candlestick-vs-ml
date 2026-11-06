"""Machine learning models, wrapped in the project's shared interface.

Every model here has fit(X, y) and predict_proba(X) -> probability of an up day,
exactly like the baselines, so the runner and evaluation treat them the same.
Settings come from the `models` section of config.yaml.

  LightGBMModel       gradient-boosted trees (week 4)
  RandomForestModel   scikit-learn random forest (week 6)
  SVMModel            scikit-learn SVC, RBF kernel, standardized features, at most
                      max_train_rows training days per fit (week 6)
  MLPModel            scikit-learn MLPClassifier, standardized features (week 6)

The SVM has no natural probability. Its "p_up" is the logistic function of its
decision function: a monotone confidence score that is above 0.5 exactly when
the SVM predicts up, and further from 0.5 the further the day is from the
decision boundary. It is not a calibrated probability, which no analysis here
needs: classification uses the 0.5 cut, and the coverage analysis ranks days by
|p_up - 0.5|, which only needs the order. (SVC's built-in probability=True would
fit five extra models by internal cross-validation and can disagree with the
SVM's own predictions.)

Trees need no feature scaling. The SVM and the MLP standardize every column
with the training days' mean and standard deviation, inside the model, so the
test year never influences the scaling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

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


class SklearnModel(Model):
    """Shared plumbing for the scikit-learn models: settings from config.yaml
    (models.<config_key>) with optional overrides, a seed, the one-direction and
    column checks, and fitting on a plain float array."""
    config_key = ""

    def __init__(self, seed: int | None = None, **overrides):
        cfg = load_config()
        self.params = {**cfg["models"][self.config_key], **overrides}
        self.seed = cfg["seed"] if seed is None else seed

    def _estimator(self):
        raise NotImplementedError

    def _training_rows(self, X: np.ndarray, y: np.ndarray):
        return X, y

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("training data contains only one direction")
        self.features_ = list(X.columns)
        Z, y = self._training_rows(np.asarray(X, dtype=float), y)
        self.n_train_ = len(y)
        self.model_ = self._estimator().fit(Z, y)
        return self

    def _matrix(self, X) -> np.ndarray:
        if list(X.columns) != self.features_:
            raise ValueError("prediction columns differ from the training columns")
        return np.asarray(X, dtype=float)

    def predict_proba(self, X):
        return self.model_.predict_proba(self._matrix(X))[:, 1]


class RandomForestModel(SklearnModel):
    """Random forest: many decision trees, each grown on a bootstrap sample of the
    training days and trying a random subset of columns at each split; P(up) is
    the average of the trees' leaf up-shares.

    Trees are grown in parallel, which is exactly reproducible for a given seed.
    Predictions are averaged on one thread: in parallel, the trees are added up
    in a varying order, which changes results in the 16th decimal place."""
    name = "random_forest"
    config_key = "random_forest"

    def _estimator(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**self.params, random_state=self.seed)

    def predict_proba(self, X):
        self.model_.set_params(n_jobs=1)
        return super().predict_proba(X)


class SVMModel(SklearnModel):
    """Support vector machine with an RBF kernel on standardized features.

    If the training window has more than max_train_rows labelled days, the
    model is fitted on a random sample of that many, drawn with the model's seed
    from the whole window (not just recent years) and kept in date order.
    predict_proba is the logistic function of the decision function (see the
    module docstring).
    """
    name = "svm"
    config_key = "svm"

    def _estimator(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        p = self.params
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=p["C"], gamma=p["gamma"],
                                 cache_size=p["cache_size_mb"], random_state=self.seed))

    def _training_rows(self, X, y):
        cap = self.params.get("max_train_rows")
        if cap is None or len(y) <= cap:
            return X, y
        rows = np.sort(np.random.default_rng(self.seed).choice(len(y), size=cap, replace=False))
        if len(np.unique(y[rows])) < 2:
            raise ValueError("the training sample contains only one direction")
        return X[rows], y[rows]

    def decision_function(self, X) -> np.ndarray:
        """Signed distance from the decision boundary: positive = up."""
        return self.model_.decision_function(self._matrix(X))

    def predict_proba(self, X):
        from scipy.special import expit
        return expit(self.decision_function(X))

    def diagnostics(self, X) -> dict:
        return {"train_rows": self.n_train_,
                "support_vectors": int(self.model_[-1].n_support_.sum())}


class MLPModel(SklearnModel):
    """A small fully connected neural network (multilayer perceptron) on
    standardized features, trained with Adam and early stopping. If training
    stops at max_iter before early stopping kicks in, the model is still used,
    and converged_ / diagnostics() record it."""
    name = "mlp"
    config_key = "mlp"

    def _estimator(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        p = {**self.params, "hidden_layer_sizes": tuple(self.params["hidden_layer_sizes"])}
        return make_pipeline(StandardScaler(), MLPClassifier(**p, random_state=self.seed))

    def fit(self, X, y):
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            super().fit(X, y)
        self.converged_ = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return self

    def diagnostics(self, X) -> dict:
        return {"train_rows": self.n_train_, "epochs": int(self.model_[-1].n_iter_),
                "converged": self.converged_}
