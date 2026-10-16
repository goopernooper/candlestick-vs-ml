## 3. Candlestick Encodings and Prediction Target

Each trading day is summarized by its candle: the open, high, low and close prices. We construct three representations of the candle, all computed once and shared by every model, so that differences in model performance cannot arise from differences in how the data were prepared. Every quantity dated day *t* uses only information available at the close of day *t*.

### 3.1 Volatility scaling

Candle dimensions are only meaningful relative to how much a stock typically moves: a $2 body is large for a stable utility and small for a volatile technology stock in a crisis. We therefore measure every candle in units of the average true range (ATR; Wilder, 1978). The true range of day *t* is the largest of the high minus the low, the high minus the previous close, and the previous close minus the low, so that overnight gaps are included. ATR is Wilder's 14-day smoothed average of the true range, matching the TA-Lib implementation. To keep day *t*'s own candle out of its scale, we divide by the ATR as of the previous close, ATR₍ₜ₋₁₎. The resulting features are unchanged if all prices are multiplied by a constant, which makes them comparable across stocks and over time, unaffected by split and dividend adjustment, and allows the stocks to be pooled into a single training set.

### 3.2 Encoding A: named candlestick patterns

Encoding A applies all 61 candlestick pattern functions in TA-Lib (Python package version 0.8.1), which provide standard, widely used definitions of patterns such as the doji, hammer, engulfing and morning star (Nison, 1991). Each function returns one column per day with value +1 when a bullish form of the pattern is present, −1 for a bearish form, and 0 otherwise. Two details of TA-Lib's conventions matter for interpretation. First, patterns that have no direction, such as the doji and spinning top, are reported as +1, so for these patterns +1 indicates presence rather than a bullish signal. Second, the two Hikkake patterns return ±200 for a "confirmed" signal; we map these to ±1 so that all columns share one scale. TA-Lib's default parameters are used throughout.

Pattern frequencies vary widely. Across the 22 stocks over 2002–2025, at least one pattern occurs on 80.5% of stock-days, largely because of simple single-candle shapes: the spinning top, long line and belt hold each appear on 16–22% of days. At the other extreme, 12 of the 61 patterns occur fewer than 100 times in roughly 133,000 stock-days, and two (concealing baby swallow and three stars in the south) never occur. Because a pattern that rarely occurs gives a model little to learn from, any decision to exclude rare patterns is based on their frequency in the first training window (2002–2004) rather than the full sample.

### 3.3 Encoding B: candle shape

Encoding B describes the candle's shape directly. Writing O, H, L and C for the open, high, low and close, the continuous version has four features:

- body = (Cₜ − Oₜ) / ATR₍ₜ₋₁₎, positive for an up day and negative for a down day;
- upper wick = (Hₜ − max(Oₜ, Cₜ)) / ATR₍ₜ₋₁₎;
- lower wick = (min(Oₜ, Cₜ) − Lₜ) / ATR₍ₜ₋₁₎;
- gap = (Oₜ − Cₜ₋₁) / ATR₍ₜ₋₁₎, the overnight move.

Wicks are floored at zero to absorb small rounding errors in adjusted prices. The signed body preserves the candle's color, and the gap captures how consecutive candles sit relative to each other, which multi-day patterns such as the engulfing depend on.

The symbolic version discretizes the body and the two wicks into a small alphabet, in the spirit of symbolic time-series representations such as SAX (Lin et al., 2007). The body is assigned to one of five levels using fixed edges at ±0.1 and ±0.6 ATR (Table 2), and each wick is classed as long if it is at least 0.25 ATR. This gives 5 × 2 × 2 = 20 symbols, one per day; the gap is omitted to keep the alphabet small enough for sequence mining. The edges were fixed in advance rather than estimated from the data, since edges set from quantiles of the full sample would leak information from the test period into every observation. They were checked only against 2002–2004 data, where every level is well populated: the five body levels hold 15.0%, 27.4%, 14.2%, 27.8% and 15.7% of days, and upper and lower wicks are long on 36.0% and 39.7% of days.

**Table 2. Body levels in symbolic encoding B**

| Level | Body (ATR units) | Interpretation |
|---|---|---|
| −− | below −0.6 | large down day |
| − | −0.6 to −0.1 | down day |
| 0 | −0.1 to 0.1 | little or no body (doji-like) |
| + | 0.1 to 0.6 | up day |
| ++ | 0.6 and above | large up day |

### 3.4 Normalized prices for the ablation

To separate the effect of the encoding from the effect of the model, machine learning models are also trained on normalized prices: the open, high, low and close each minus the previous close, divided by ATR₍ₜ₋₁₎. Raw price levels are never used. These four values contain exactly the same information as continuous encoding B; for example, the body equals the normalized close minus the normalized open, and the gap equals the normalized open. Any difference between models trained on normalized prices and models trained on the symbolic or pattern encodings can therefore be attributed to the binning and the pattern definitions, not to a difference in available information.

### 3.5 Prediction target

The target for day *t* is the direction of the next day's close-to-close return, rₜ₊₁ = Cₜ₊₁ / Cₜ − 1. Returns very close to zero are essentially coin flips, so we treat a day as unlabelled when |rₜ₊₁| is smaller than 0.1 times the standard deviation of the stock's previous 20 daily returns, measured through day *t*. Scaling this threshold by recent volatility keeps its meaning constant across calm and turbulent periods. These dead-zone days (9.2% of stock-days, between 8.0% and 10.7% in every year) are excluded from model training and from classification metrics, but they remain in the trading backtest, where positions held on them still incur transaction costs. Among the 120,660 labelled days, 52.3% are up days, which is the accuracy achieved by always predicting the majority class.

### 3.6 Look-ahead testing

Every feature was tested for look-ahead bias. For a cut-off day *T*, all prices after *T* were rewritten in four ways (scaled by a constant, perturbed randomly, replayed in reverse order, and deleted entirely, as if the features were computed live on day *T*), and we confirmed that no feature dated *T* or earlier changed. The test was run on synthetic data and on real stocks around the 2008 and 2020 crises, and it was verified to detect four deliberately leaky features, including a centered moving average and bins based on full-sample quantiles. The labels were confirmed to depend on exactly one future day, which determines the gap the walk-forward splits must leave between training and test data.

### References for this section

- Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). Experiencing SAX: A novel symbolic representation of time series. *Data Mining and Knowledge Discovery*, 15(2), 107–144.
- Nison, S. (1991). *Japanese Candlestick Charting Techniques*. New York Institute of Finance.
- TA-Lib: Technical Analysis Library. https://ta-lib.org
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
