"""Candlestick features. Every feature dated t uses only data available at t's close.

This file starts with ATR (average true range), the volatility yardstick that
every other feature is divided by. More encodings are added in week 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATR_PERIOD = 14


def true_range(df: pd.DataFrame) -> pd.Series:
    """Daily true range: the day's full price span, including any overnight gap.

    TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)

    The first row has no previous close, so it is NaN (as in TA-Lib).
    """
    prev_close = df["Close"].shift(1)
    spans = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1)
    tr = spans.max(axis=1)
    tr[prev_close.isna()] = np.nan
    return tr.rename("true_range")


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder's average true range, matching talib.ATR.

    The first value is the plain average of the first `period` true ranges.
    After that, each day blends in the new true range:
        ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period

    ATR_t includes day t's own candle, so it is only known at t's close.
    Use lagged_atr() to scale day t's candle.
    """
    tr = true_range(df).to_numpy()
    out = np.full(len(tr), np.nan)
    if len(tr) > period:
        out[period] = tr[1:period + 1].mean()
        for t in range(period + 1, len(tr)):
            out[t] = (out[t - 1] * (period - 1) + tr[t]) / period
    return pd.Series(out, index=df.index, name=f"atr{period}")


def lagged_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR shifted forward one day: the value on day t is ATR as of t-1's close.

    This is the volatility known before day t's candle forms, so dividing day
    t's body and wicks by it uses no information from day t itself.
    """
    return atr(df, period).shift(1).rename(f"atr{period}_lag1")
