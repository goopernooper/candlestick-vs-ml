"""Candlestick features. Every feature dated t uses only data available at t's close.

Contents
  ATR (average true range): the volatility yardstick other features are divided by.
  Encoding A: TA-Lib's named candlestick patterns, one signed column per pattern.
"""
from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Encoding B (continuous): candle shape measured in ATR units
# ---------------------------------------------------------------------------
B_CONTINUOUS = ["b_body", "b_upper", "b_lower", "b_gap"]


def encoding_b_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Four numbers describing each day's candle, divided by the previous day's ATR.

    b_body   Close - Open. Signed: positive for an up candle, negative for a down one.
    b_upper  High - max(Open, Close). The upper wick, always >= 0.
    b_lower  min(Open, Close) - Low. The lower wick, always >= 0.
    b_gap    Open - previous Close. Signed: positive when the day opens higher.

    Dividing by lagged ATR puts every stock and every era on the same scale: a
    body of 1.0 means "as big as a typical day's range lately", whether the stock
    trades at $5 or $500. Because ATR is from the previous close, day t's values
    use only information available at day t's close.

    The first 15 rows are NaN while ATR warms up (the 2001 buffer covers this).
    """
    scale = lagged_atr(df)
    scale = scale.where(scale > 0)  # a zero ATR would divide by zero; leave those days NaN
    o, h, l, c = (df[col] for col in OHLC)
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    out = pd.DataFrame({
        "b_body": (c - o) / scale,
        # Adjusted prices carry tiny rounding errors, so clip wicks at zero.
        "b_upper": (h - top).clip(lower=0) / scale,
        "b_lower": (bottom - l).clip(lower=0) / scale,
        "b_gap": (o - c.shift(1)) / scale,
    }, index=df.index)
    return out


# ---------------------------------------------------------------------------
# Encoding B (symbolic): each candle as one of 20 symbols
# ---------------------------------------------------------------------------
from functools import lru_cache  # noqa: E402

import yaml  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
BODY_LEVELS = ["--", "-", "0", "+", "++"]  # big down, down, doji-like, up, big up
N_SYMBOLS = 20


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def symbol_label(symbol: int) -> str:
    """Readable name for a symbol number, e.g. 14 -> '+ U l' (up body, long upper
    wick, short lower wick). Capital U/L means the wick is long."""
    body, rest = divmod(int(symbol), 4)
    upper, lower = divmod(rest, 2)
    return f"{BODY_LEVELS[body]} {'U' if upper else 'u'} {'L' if lower else 'l'}"


def encoding_b_symbolic(df: pd.DataFrame, body_edges=None, wick_edge=None) -> pd.DataFrame:
    """Bin continuous encoding B into one symbol per day (0-19).

    b_body_level   0-4: big down, down, doji-like, up, big up (see config.yaml)
    b_upper_long   1 if the upper wick is at least `wick_edge` ATR, else 0
    b_lower_long   same for the lower wick
    b_symbol       body_level * 4 + upper_long * 2 + lower_long

    The edges come from config.yaml unless passed in. They are fixed numbers,
    not fitted to the data, so the encoding can't leak information across time.
    The first 15 days are missing (<NA>) while ATR warms up.
    """
    if body_edges is None or wick_edge is None:
        cfg = load_config()["encoding_b"]
        body_edges = cfg["body_edges"] if body_edges is None else body_edges
        wick_edge = cfg["wick_edge"] if wick_edge is None else wick_edge
    body_edges = np.asarray(body_edges, dtype=float)
    wick_edge = float(wick_edge)

    cont = encoding_b_continuous(df)
    ready = cont.notna().all(axis=1).to_numpy()
    body = np.digitize(cont["b_body"].to_numpy(), body_edges)          # 0..4
    upper = (cont["b_upper"].to_numpy() >= wick_edge).astype(int)
    lower = (cont["b_lower"].to_numpy() >= wick_edge).astype(int)
    symbol = body * 4 + upper * 2 + lower

    def col(values):
        arr = pd.array(np.where(ready, values, 0), dtype="Int8")
        arr[~ready] = pd.NA  # missing before ATR is ready
        return arr

    out = pd.DataFrame({
        "b_body_level": col(body), "b_upper_long": col(upper),
        "b_lower_long": col(lower), "b_symbol": col(symbol),
    }, index=df.index)
    return out


# ---------------------------------------------------------------------------
# Normalized OHLC (for the fair-features ablation)
# ---------------------------------------------------------------------------
NORMALIZED_OHLC = ["n_open", "n_high", "n_low", "n_close"]


def normalized_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Each day's open, high, low and close minus the previous close, in ATR units.

    n_close is the day's return measured in ATRs; n_open is the overnight gap.
    Raw price levels are never used, so a $5 stock and a $500 stock are on the
    same scale, exactly as in encoding B.

    These four numbers carry the same information as continuous encoding B,
    just arranged differently:
        b_gap   = n_open
        b_body  = n_close - n_open
        b_upper = n_high - max(n_open, n_close)
        b_lower = min(n_open, n_close) - n_low
    That is the point of the ablation: models given these features see the
    candle without any binning or named patterns, so any gap between them and
    the pattern models comes from the binning and pattern rules, not from
    having different information.

    Uses the previous day's ATR, so day t's values are known at day t's close.
    The first 15 rows are NaN while ATR warms up.
    """
    scale = lagged_atr(df)
    scale = scale.where(scale > 0)
    prev_close = df["Close"].shift(1)
    return pd.DataFrame({
        "n_open": (df["Open"] - prev_close) / scale,
        "n_high": (df["High"] - prev_close) / scale,
        "n_low": (df["Low"] - prev_close) / scale,
        "n_close": (df["Close"] - prev_close) / scale,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Encoding B over a window of days (what the ML models see)
# ---------------------------------------------------------------------------
def encoding_b_window(df: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    """Continuous encoding B for today and the previous days.

    With days=3: b_body, b_upper, b_lower, b_gap for today, the same four with
    the suffix _lag1 for yesterday, and _lag2 for the day before. Lags are taken
    within this one stock's history (one stock per call), so they never mix
    stocks, and they only reach backward, so they can't see the future.
    """
    days = load_config()["encoding_b"]["window_days"] if days is None else days
    cont = encoding_b_continuous(df)
    parts = [cont] + [cont.shift(k).add_suffix(f"_lag{k}") for k in range(1, days)]
    return pd.concat(parts, axis=1)


def max_window_days() -> int:
    """Longest window any model needs, so one table holds every model's columns."""
    cfg = load_config()
    return max([cfg["encoding_b"]["window_days"], *cfg["models"]["knn"]["windows"]])


def window_columns(days: int | None = None) -> list[str]:
    days = load_config()["encoding_b"]["window_days"] if days is None else days
    return B_CONTINUOUS + [f"{c}_lag{k}" for k in range(1, days) for c in B_CONTINUOUS]


# ---------------------------------------------------------------------------
# Registry of every feature set. tests/test_leakage.py runs each one through the
# look-ahead checks, and fails if a feature function exists that isn't listed here.
# When you add an encoding, add it here too.
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "lagged_atr": lambda df: lagged_atr(df).to_frame(),
    "encoding_a": encoding_a,
    "encoding_b_continuous": encoding_b_continuous,
    # Built with the longest window any model uses; each model selects its own days.
    "encoding_b_window": lambda df: encoding_b_window(df, days=max_window_days()),
    "encoding_b_symbolic": encoding_b_symbolic,
    "normalized_ohlc": normalized_ohlc,
}
