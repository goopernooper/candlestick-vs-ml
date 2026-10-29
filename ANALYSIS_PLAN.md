# Analysis Plan

**Status: DRAFT** (week 4, 2026-10-29). To be frozen and tagged `analysis-plan-frozen` in week 8, before the 2023–2025 holdout is unlocked.

This document fixes, before the final results are seen, exactly what will be run and how it will be judged. Once frozen, nothing in it changes. Any later deviation is listed in the Deviations section at the end with its date and reason, and is reported in the paper.

Items marked **[confirm]** depend on decisions not yet recorded as Decided in DECISIONS.md, or are proposals for the next check-in.

---

## 1. Questions

**Primary.** Do candlestick pattern-based models (a variable-order pattern miner and k-nearest neighbours) predict next-day stock direction better than standard machine learning models (random forest, LightGBM, SVM, MLP) and naive baselines, after transaction costs?

**Headline analysis: the model × encoding grid** (D-012 **[confirm]**). Every machine learning model is trained on each encoding, so the effect of the encoding can be separated from the effect of the model class.

**Secondary.**
1. *Fair-features ablation:* does binning candles into symbols or named patterns add anything beyond the same information given as normalized prices?
2. *Cost sensitivity:* at what transaction cost does any edge disappear?
3. *Regimes:* does the ranking of models change between low-, medium- and high-volatility periods?
4. *Reversal:* is any edge more than the known short-term reversal effect? Added after results v1 showed the previous-day baseline with a negative MCC in 15 of 18 years. **[confirm]** This question arose from development results; it is reported as post hoc.

---

## 2. Data (fixed)

- 22 stocks: the two largest current S&P 500 members by market capitalization in each GICS sector with price data on 2002-01-02 (README, D-011 **[confirm]**). SPY is the market proxy; VIX is used only in robustness checks.
- Daily open, high, low and close from Yahoo Finance via yfinance 1.7.0, adjusted for splits and dividends, downloaded 2026-09-28. The exact files are fixed by the hashes in `data/manifest.json`; every run stops if they don't match.
- Study period: 2002-01-02 to 2025-12-31.
  - Development test years: 2005–2022.
  - Final holdout: 2023–2025.

## 3. Target

- Direction of the next day's close-to-close return (horizon 1 day).
- **Dead zone:** a day is unlabelled when |return| < 0.1 × the standard deviation of the previous 20 daily returns (`config.yaml`, `labels`). Dead-zone days are excluded from training and from classification metrics, but kept in the backtest.

## 4. Encodings

All computed once, shared by every model, and tested for look-ahead (`tests/test_leakage.py`).

| Encoding | Columns | Used by |
|---|---|---|
| A: TA-Lib patterns | 61 signed pattern indicators for day *t* | ML models |
| B continuous, 3-day window | body, upper wick, lower wick, gap in ATR units, for days *t*, *t*−1, *t*−2 (12 columns) | ML models; k-NN |
| B continuous, 5-day window | as above, for five days (20 columns) | k-NN |
| B symbolic | 20 symbols per day (5 body levels × long/short upper wick × long/short lower wick) | pattern miner |
| Normalized OHLC | open, high, low, close minus the previous close, in ATR units (4 columns) | ML models (ablation) |
| Today's return only | `n_close` (1 column) | LightGBM ablation **[confirm]** |

Bin edges for symbolic B are fixed in `config.yaml` (body ±0.1 and ±0.6 ATR, wicks 0.25 ATR) and are not changed after this plan is frozen.

## 5. Models (final list)

**Baselines** (in every table):
- `majority`, `always_long`, `previous_day`, `random` (seeded per fold)
- `reversal`: predicts the opposite of today's direction. Added post hoc. **[confirm]**

**Pattern-based models:**
- `pattern_miner`: a variable-order Markov model on symbolic B. Contexts of 1–3 symbols; P(up | context) shrunk toward the training base rate; backs off to shorter contexts below a minimum count (Begleiter et al., 2004).
- `knn_b3`, `knn_b5`: k-nearest neighbours, Euclidean distance, standardized within each fold, on 3- and 5-day windows of continuous B.

**Machine learning models**, each trained on encoding A, B continuous (3-day window), and normalized OHLC:
- random forest (`rf`)
- LightGBM (`lightgbm`)
- SVM with an RBF kernel (`svm`). Its decision function is used as the confidence score. If it is too slow on pooled data, training rows are capped by a fixed-seed random sample, and the cap is recorded here before results are produced.
- small multilayer perceptron (`mlp`)

**Optional** (only if ahead of schedule; listed so that adding them is not a deviation): DTW distance for k-NN; an LSTM; the 5-day forward target.

Models are named `model_encoding` (for example `lightgbm_a`, `svm_ohlc`). All models are reported; none are dropped for performing badly.

## 6. Validation

- Walk-forward: an expanding training window from 2002-01-02, with one test year at a time from 2005 to 2022 (18 folds). Splits are by calendar date, so all stocks' rows for a day land together.
- **Purge:** the last *h* trading days before each test year are removed from training (*h* = 1).
- **Tuning:** inside each training window only. Fit on all but the last year, score on that last year (with an *h*-day purge), then refit the chosen settings on the whole window.
  - Every model family, the pattern models included, gets the same budget: 20 Optuna trials.
  - Selection criterion: MCC on the tuning year.
  - Re-tuning happens in the folds starting 2005, 2010, 2015 and 2020; folds in between reuse the most recent settings.
  - The 2025 retuning applies only to the holdout run.

## 7. Metrics

**Primary:** MCC, pooled over all labelled test days 2005–2022.

**Secondary classification:** balanced accuracy and accuracy, each pooled, per year, and per stock.

**Economic** (equal-weight portfolio, positions from day *t*'s close to day *t*+1's close):
- long/short (primary) and long/flat (compared against buy-and-hold)
- Sharpe ratio (no risk-free rate) at 5 bps per side
- annualized return, maximum drawdown, turnover
- **break-even cost:** the cost per side at which annual long/short return reaches zero

**Cost grid:** 0, 1, 2, 5, 10, 20 bps per side. Headline: 5 bps (D-013 **[confirm]**).

## 8. Statistical tests

Daily losses are averaged across the 22 stocks before testing, because the stocks are not independent.

1. **Loss series:** for each model, two daily series:
   - the share of the day's labelled stock-days it got wrong (classification loss)
   - minus the portfolio's net long/short return at 5 bps (economic loss)
2. **Diebold–Mariano tests** on the difference in daily loss between two models: a regression on a constant with Newey–West standard errors, lag = ⌊4(T/100)^(2/9)⌋. Two-sided, α = 0.05.
3. **Model Confidence Set** (`arch.bootstrap.MCS`): all models, both loss series, 90% confidence, 10,000 block-bootstrap replications.
4. **SPA test** (`arch.bootstrap.SPA`): does any model beat the benchmark? Benchmarks: always-long for the economic loss; the strongest baseline for the classification loss.
5. **Confidence intervals** for MCC, accuracy and Sharpe: stationary block bootstrap over dates (all stocks kept together on each date), mean block length 20 trading days, 10,000 replications.

## 9. What counts as beating a baseline

A model **beats** a baseline on classification if:
- its pooled MCC is higher, **and**
- a Diebold–Mariano test on the daily classification-loss difference is significant at the 5% level.

A model **beats** a baseline economically if:
- its long/short Sharpe ratio at 5 bps is higher, **and**
- a Diebold–Mariano test on the daily economic-loss difference is significant at the 5% level.

Beating buy-and-hold requires the same comparison of long/flat against `always_long` long/flat.

A model **beats the baselines** only if it beats the strongest one: the baseline with the highest pooled MCC (including `reversal`, if confirmed) for classification, and buy-and-hold for the economic claim.

A model shows **genuine predictive value** if it beats the baselines *and* the strongest baseline is excluded from the 90% Model Confidence Set while the model is in it.

Consistency across years (the number of the 18 folds in which the model beats the baseline) is reported for context but is not a requirement.

**If no model meets these criteria, that is the finding** (D-006), and the paper says so plainly.

## 10. Coverage analysis

- On each fold's tuning year, choose confidence cutoffs (|P(up) − 0.5|) that would trade 100%, 50%, 25% and 10% of days.
- Apply them unchanged to the test year.
- Report accuracy and MCC against the coverage actually achieved, for every model.

## 11. Regimes

- **Definition** (D-014 **[confirm]**): SPY's trailing 21-day realized volatility on day *t*, assigned to low / medium / high by terciles. The tercile cutoffs on day *t* use only volatility values through day *t* − 1, on an expanding window starting 2002-01-02.
- **Analysis:** every model is re-evaluated within each regime, with the same metrics and tests.
- The high-volatility results are also broken down by episode (2008–09, 2011, 2020, 2022), because a few episodes dominate that regime.
- **Robustness:** VIX terciles in place of SPY volatility.

## 12. Robustness checks

1. Per-stock training instead of pooled.
2. Positions entered at the next day's open instead of day *t*'s close.
3. Dead-zone threshold of 0 and of twice the default.
4. VIX-based regimes.

Robustness results are reported whether or not they agree with the main results.

## 13. Holdout protocol

1. Freeze this plan: commit, then tag `analysis-plan-frozen`.
2. Set `unlock_holdout: true` in `config.yaml` and log it in DECISIONS.md.
3. Run the frozen pipeline once on 2023–2025 and commit the results, whatever they show.
4. No model, setting or feature is changed after seeing the holdout. If a bug is found afterwards, it is fixed and both the original and corrected holdout results are reported.

## 14. Deviations after freezing

| Date | What changed | Why | Effect on results |
|---|---|---|---|
| | | | |
