# Decisions

A running log of design decisions for this project: what was decided, when, by whom, and why.

**How to use this file**

- Add an entry whenever a check-in (or anything else) settles a question.
- Never delete an entry. If a decision changes, mark the old one **Superseded**, add a new entry, and link the two.
- Commit this file right after each Monday check-in, e.g. `git commit -m "decisions: 2026-09-28 check-in"`.
- Statuses: **Proposed** (waiting for a check-in), **Decided**, **Superseded**.

Professor Danisman gave full latitude on design, so entries marked "Decided by: Joshua" were settled while writing the project plan. Entries marked **Proposed** are defaults to confirm with him.

## Index

| ID | Decision | Status | Date |
|---|---|---|---|
| D-001 | Prediction target | Decided | 2026-09-23 |
| D-002 | Candlestick encodings | Decided | 2026-09-23 |
| D-003 | Models and baselines | Decided | 2026-09-23 |
| D-004 | Validation scheme | Decided | 2026-09-23 |
| D-005 | Metrics and significance tests | Decided | 2026-09-23 |
| D-006 | Commitment to report a null result | Decided | 2026-09-23 |
| D-007 | Data source | Decided | 2026-09-23 |
| D-008 | Venue and deadline | Proposed | |
| D-009 | Pooled vs. per-ticker training | Proposed | |
| D-010 | Date range and holdout | Proposed | |
| D-011 | Ticker universe | Proposed | |
| D-012 | Headline result | Proposed | |
| D-013 | Trading rules and costs | Proposed | |
| D-014 | Regime definition | Proposed | |
| D-015 | Preprint and authorship | Proposed | |

---

## Settled while writing the plan

### D-001: Prediction target
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** Primary target is next-day close-to-close direction. Days whose absolute return is below 0.1 × the trailing 20-day standard deviation form a dead zone: they are dropped from training labels and classification metrics but kept in the backtest. Secondary target is n-day forward return (on the cut list).
- **Why:** Direction is the natural question for candlestick patterns. The dead zone keeps near-zero moves from being labeled arbitrarily, and scaling it by volatility keeps it comparable across stocks and years.
- **Alternatives considered:** Fixed-percentage dead zone (not comparable across volatility levels); regression on returns (harder to compare with pattern rules).

### D-002: Candlestick encodings
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:**
  - Encoding A: TA-Lib CDL pattern functions, one signed column per pattern.
  - Encoding B (continuous): signed body, upper wick, lower wick, and gap from prior close, each divided by ATR(14) lagged one day.
  - Encoding B (symbolic): body and wicks binned with fixed edges (body at ±0.1 and ±0.6 ATR, wicks at 0.25 ATR), 20 symbols.
  - Normalized OHLC for the ablation: each price minus the prior close, in ATR units.
  - All encodings use only data available at each day's close, are computed once, and are shared by every model.
- **Why:** TA-Lib gives standard, citable pattern definitions. Fixed bin edges avoid leaking future data into the encoding. Signed bodies keep candle color; the gap captures how consecutive candles relate.
- **Alternatives considered:** Hand-coded patterns (non-standard); quantile bin edges over the full sample (leaks future information).

### D-003: Models and baselines
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:**
  - Pattern side: a variable-order pattern miner (contexts of 1–3 symbols, shrinkage toward the base rate, backoff to shorter contexts) and k-NN on continuous encoding B.
  - ML side: random forest, LightGBM, RBF-kernel SVM, small MLP.
  - Baselines: majority class, always-long, yesterday's direction, random.
  - Every model family gets the same hyperparameter tuning budget.
- **Why:** Covers the subsequence and similarity methods and the tree, SVM, and neural models named in the project description. Equal tuning budgets keep the comparison about model class rather than tuning effort.
- **Alternatives considered:** CNN/LSTM (kept as optional, first on the cut list); DTW distance for k-NN (optional, second on the cut list).

### D-004: Validation scheme
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** Expanding-window walk-forward, one test year at a time, with a purge gap equal to the label horizon. The last year of each training window is the tuning set. Splits are by calendar date and never shuffled.
- **Why:** Mirrors how a model would actually be used and prevents look-ahead leakage.
- **Alternatives considered:** Rolling window (possible robustness check); random k-fold (invalid for time series).

### D-005: Metrics and significance tests
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:**
  - Classification metrics: accuracy, balanced accuracy, MCC.
  - Economic metrics: annualized return and Sharpe ratio, net of costs.
  - Significance: Diebold-Mariano tests with Newey-West standard errors on each day's loss averaged across tickers, plus the Model Confidence Set over all models.
  - Uncertainty: stationary block-bootstrap intervals that resample dates, keeping all tickers together.
- **Why:** Tickers move together, so treating them as independent samples would overstate significance. The Model Confidence Set handles the many comparisons.
- **Alternatives considered:** Paired tests treating tickers as independent (overstates confidence).

### D-006: Commitment to report a null result
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** If no method reliably beats the baselines after costs, the paper reports that as its finding. This is fixed before any results are seen.
- **Why:** It is the most likely honest outcome, it is publishable, and committing in advance removes the temptation to keep searching until something looks significant.

### D-007: Data source
- **Status:** Decided, 2026-09-23. Decided by: Joshua.
- **Decision:** Daily OHLC from yfinance with `auto_adjust=True` set explicitly. The library version is pinned, raw data is cached as Parquet, and each snapshot gets a manifest (download date, version, row counts, file hashes).
- **Why:** Free and widely used. Pinning and snapshotting make results reproducible even if Yahoo's data changes.
- **Alternatives considered:** Stooq (fallback if yfinance becomes unreliable).

---

## To confirm at the September 28 check-in

Fill in **Decided** during the meeting, update the index, and commit.

### D-008: Venue and deadline
- **Status:** Proposed.
- **Proposed:** No default; this is the most important question for the meeting.
- **Why it matters:** The deadline decides whether week 12 ends in a submission or a polished draft, and how much of the cut list gets used.
- **Decided:**
- **Notes:**

### D-009: Pooled vs. per-ticker training
- **Status:** Proposed.
- **Proposed:** One model per fold trained on all tickers together; per-ticker models as a robustness check.
- **Why:** Per-ticker models get about 250 samples a year, which starves both the ML models and the pattern miner. ATR normalization makes pooling legitimate, and pooling directly tests the claim that candlestick patterns are universal.
- **Decided:**
- **Notes:**

### D-010: Date range and holdout
- **Status:** Proposed.
- **Proposed:** Daily data from January 2002 through December 2025. Development test years 2005–2022. 2023–2025 stays locked as a final holdout until the analysis plan is frozen in week 8.
- **Why:** Starting in 2002 is after decimalization and puts 2008 in the test set. Ending in 2025 adds the April 2025 selloff as another high-volatility episode. The holdout gives one clean out-of-sample check.
- **Decided:**
- **Notes:**

### D-011: Ticker universe
- **Status:** Proposed.
- **Proposed:** About 25 large caps, two or three per sector, with continuous data since 2002, plus SPY. The selection rule is written in the README, and survivorship bias is acknowledged as a limitation.
- **Question for Prof. Danisman:** Does he prefer a universe from prior work, for consistency?
- **Decided:**
- **Notes:**

### D-012: Headline result
- **Status:** Proposed.
- **Proposed:** Lead with the model × encoding grid; regime conditioning is a secondary analysis.
- **Why:** The grid separates the effect of the encoding from the effect of the model class, which prior work usually mixes together. The high-volatility regime rests on only a handful of episodes, so it is weaker as a headline.
- **Decided:**
- **Notes:**

### D-013: Trading rules and costs
- **Status:** Proposed.
- **Proposed:**
  - Signal at day t's close, trade at that close; next-day open as a robustness check.
  - Long/short is primary; long/flat against buy-and-hold is also reported.
  - Headline tables charge 5 bps per side; the cost curve runs from 0 to 20 bps.
- **Decided:**
- **Notes:**

### D-014: Regime definition
- **Status:** Proposed.
- **Proposed:** Terciles of SPY's trailing 21-day realized volatility, with cutoffs computed from past data only. VIX as a robustness check.
- **Why:** Using past data only means a regime finding is something a trader could have acted on at the time.
- **Decided:**
- **Notes:**

### D-015: Preprint and authorship
- **Status:** Proposed.
- **Proposed:** Agree on the author list and order, and whether to post a preprint before or after submission. First-time arXiv authors may need an endorsement, so raise it early.
- **Decided:**
- **Notes:**

---

## Check-in log

One line per meeting, pointing to the decisions it touched.

| Date | Summary | Decisions |
|---|---|---|
| 2026-09-28 | | D-008 to D-015 |

---

## Entry template

```markdown
### D-0XX: Short title
- **Status:** Proposed / Decided (YYYY-MM-DD) / Superseded by D-0YY
- **Decided by:**
- **Decision:**
- **Why:**
- **Alternatives considered:**
- **Consequences:** (what changes in code, config, or the paper)
```
