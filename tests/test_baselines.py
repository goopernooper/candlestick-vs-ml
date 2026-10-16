"""Tests for the four baselines, including an end-to-end check on one walk-forward fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import normalized_ohlc  # noqa: E402
from labels import make_labels  # noqa: E402
from models.baselines import (BASELINES, AlwaysLong, Majority, PreviousDay,  # noqa: E402
                              RandomGuess)
from splits import walk_forward  # noqa: E402

X = pd.DataFrame({"n_close": [0.8, -0.3, 0.0, 1.2, -2.0]})


def test_all_four_registered():
    assert set(BASELINES) == {"majority", "always_long", "previous_day", "random"}


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_common_interface(name):
    model = BASELINES[name]()
    assert model.fit(X, [1, 0, 1, 1, 0]) is model
    p = model.predict_proba(X)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
    assert set(model.predict(X)) <= {0, 1}


def test_majority_uses_training_share():
    m = Majority().fit(X, [1, 1, 1, 0])
    assert np.allclose(m.predict_proba(X), 0.75) and (m.predict(X) == 1).all()
    m = Majority().fit(X, [0, 0, 1])
    assert (m.predict(X) == 0).all()          # a down-majority window predicts down


def test_always_long_ignores_training():
    assert (AlwaysLong().fit(X, [0, 0, 0]).predict(X) == 1).all()


def test_previous_day_follows_today():
    p = PreviousDay().predict_proba(X)
    assert p.tolist() == [1.0, 0.0, 0.5, 1.0, 0.0]


def test_previous_day_needs_its_column():
    with pytest.raises(KeyError, match="n_close"):
        PreviousDay().predict_proba(pd.DataFrame({"other": [1.0]}))


def test_random_is_reproducible_and_fair():
    big = pd.DataFrame(index=range(20000))
    a = RandomGuess(seed=1).fit(big, [1]).predict_proba(big)
    b = RandomGuess(seed=1).fit(big, [1]).predict_proba(big)
    c = RandomGuess(seed=2).fit(big, [1]).predict_proba(big)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs((a >= 0.5).mean() - 0.5) < 0.02


def test_random_uses_config_seed():
    from features import load_config
    assert RandomGuess().seed == load_config()["seed"]


def test_end_to_end_on_one_fold():
    """The 'done when' sanity checks for week 3, on synthetic stocks with upward drift:
    always-long's accuracy equals the share of up days, and random is near 50%."""
    cal = pd.bdate_range("2001-10-01", "2025-12-31")
    rng = np.random.default_rng(0)
    frames = []
    for t in ["AAA", "BBB", "CCC"]:
        c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(cal))))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, len(cal)))
        df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.005,
                           "Low": np.minimum(o, c) * 0.995, "Close": c}, index=cal)
        frames.append(normalized_ohlc(df).join(make_labels(df)).assign(ticker=t))
    table = pd.concat(frames)
    fold = walk_forward(cal)[3]
    train = fold.select(table, "train").dropna(subset=["up", "n_close"])
    test = fold.select(table, "test").dropna(subset=["up", "n_close"])
    y_train, y_test = train["up"].astype(int), test["up"].astype(int).to_numpy()

    acc = {name: (cls().fit(train, y_train).predict(test) == y_test).mean()
           for name, cls in BASELINES.items()}
    assert acc["always_long"] == pytest.approx(y_test.mean())
    assert abs(acc["random"] - 0.5) < 0.04
    assert 0.4 < acc["previous_day"] < 0.6
