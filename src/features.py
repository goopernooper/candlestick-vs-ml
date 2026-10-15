"""Candlestick features. Every feature dated t uses only data available at t's close.

Contents
  ATR (average true range): the volatility yardstick other features are divided by.
  Encoding A: TA-Lib's named candlestick patterns, one signed column per pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

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


# ---------------------------------------------------------------------------
# Encoding A: named candlestick patterns
# ---------------------------------------------------------------------------

PATTERNS = talib.get_function_groups()["Pattern Recognition"]  # 61 CDL* functions


def encoding_a(df: pd.DataFrame) -> pd.DataFrame:
    """One column per TA-Lib candlestick pattern: +1 bullish, -1 bearish, 0 none.

    TA-Lib returns +100/-100, and +200/-200 for a few patterns when a later
    candle confirms them (for example CDLHIKKAKE). Only the sign is kept.

    Some patterns have no direction (CDLDOJI, for example). TA-Lib reports
    them as +100, so for those +1 means "present", not "bullish".
    pattern_frequency.py lists which patterns are like this.

    The value on day t is the pattern that completes on day t, using that
    day's candle and earlier ones only, so it is known at t's close.
    """
    o, h, l, c = (df[col].to_numpy(dtype=float) for col in ["Open", "High", "Low", "Close"])
    cols = {name: np.sign(getattr(talib, name)(o, h, l, c)).astype(np.int8) for name in PATTERNS}
    return pd.DataFrame(cols, index=df.index)


# ---------------------------------------------------------------------------
# Encoding A: TA-Lib's named candlestick patterns
# ---------------------------------------------------------------------------
import talib  # noqa: E402

OHLC = ["Open", "High", "Low", "Close"]
# All 61 pattern functions, in alphabetical order (CDL2CROWS ... CDLXSIDEGAP3METHODS).
# Seven take a "penetration" setting; TA-Lib's defaults are used for all of them.
PATTERNS = sorted(talib.get_function_groups()["Pattern Recognition"])


def pattern_column(name: str) -> str:
    """CDLENGULFING -> a_engulfing"""
    return "a_" + name[3:].lower()


def encoding_a(df: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
    """One column per TA-Lib candlestick pattern.

    Each value is +1 (pattern found, bullish), -1 (found, bearish) or 0 (not found).
    TA-Lib itself returns +100/-100, and +200/-200 when a pattern is "confirmed"
    (only the Hikkake patterns do this); both collapse to +1/-1 here.
    Pass raw=True to keep TA-Lib's original values.

    Note: some patterns have no direction (doji, spinning top and the like), and
    TA-Lib reports them as +100. For those, +1 means "present", not "bullish".

    Each day's value uses only that day's candle and the candles before it, so
    it is known at that day's close.
    """
    o, h, l, c = (df[col].to_numpy(dtype=float) for col in OHLC)
    cols = {pattern_column(name): getattr(talib, name)(o, h, l, c) for name in PATTERNS}
    out = pd.DataFrame(cols, index=df.index)
    return out if raw else np.sign(out).astype(np.int8)
