## 9. Discussion

### 9.1 What the study found

Candlesticks carry a small amount of information about the next day's direction, and almost all of it is the short-term reversal effect. Across 18 years of walk-forward tests, the best models reached an MCC of about 0.03, and most candle-based models were significantly better than always predicting an up day. None was significantly better than the one-line rule that a stock will reverse today's move, either over all days or within any volatility regime, and no trading strategy significantly beat buy-and-hold after costs. On the three untouched years of the holdout, the development signal disappeared: every model's MCC was within 0.02 of zero. The pre-registered test of the most promising development finding, that models beat reversal on their most confident days, was not confirmed (*p* = 0.062). The eight reversal patterns that Caginalp and Laurent (1998) found strongly predictive did not predict on our data, with their own definitions and test.

These are negative results, and the study was designed so that a negative result would be informative. The method, tests and criteria were fixed before the holdout was used, every comparison includes the baselines, and the analyses added after development results were seen are identified as such. Within the limits discussed in Section 9.6, the conclusion is that candlestick charting offers no usable edge on these stocks once short-term reversal, transaction costs and realistic execution are taken into account.

### 9.2 Why candlestick patterns can appear to work

The results suggest three reasons why earlier studies have disagreed about candlestick patterns, and why patterns can look predictive when they are not.

**Reversal looks like pattern recognition.** Most named candlestick patterns are reversal patterns: they describe a fall followed by a sign of recovery, or the reverse. Short-term reversal is a well-documented effect in stock returns (Jegadeesh, 1990; Lehmann, 1990), so a test that compares patterns only with chance, or with buy-and-hold, will find them predictive whenever reversal is. In our data, a rule that uses nothing but the sign of today's move is statistically indistinguishable from the best candle-based models. Studies that do not include such a baseline cannot tell pattern recognition from reversal.

**Correlated signals inflate significance.** Signals cluster in time: a market-wide sell-off produces the same pattern in many stocks on the same day, and their outcomes move together. A test that treats every signal as an independent observation overstates the evidence. This was the test behind Caginalp and Laurent's very large Z statistics. In our replication, the per-signal statistic is larger in size than the date-averaged one for exactly this reason (−2.1 against −1.4 for the bullish patterns), although here neither finds a positive effect.

**Results depend on the period.** Predictability in our data was concentrated in a few turbulent episodes, above all 2008–09, and differed in kind between episodes: reversal worked in crash-and-rebound markets and failed in the steady decline of 2022 and the steady rise of 2023–2024. A study covering a few years can find a strong effect or none, depending on which kind of market it happens to cover. The original positive result came from a five-year bull market, and our holdout came from another.

### 9.3 Encodings and model classes

How candles are described mattered less than which model read them, and named patterns mattered least. Encoding A's 61 TA-Lib pattern indicators gave the lowest MCC for three of the four machine learning models. A three-day window of continuous candle shape (encoding B) gave the best or equal-best results for all four, and normalized prices over the same three days gave the same results for the support vector machine and the multilayer perceptron. Describing a candle by its body and wicks therefore adds no information. It helps tree-based models only because they cannot easily form differences between prices on their own. Reducing candles to a vocabulary of named patterns discards information that the models can otherwise use.

The pattern-based models, which match past sequences of candles directly, ranked below the machine learning models given the same candles. The variable-order pattern miner, which matches exact sequences of symbols, faced a basic data problem. Even pooled across 22 stocks, most three-day sequences occurred too rarely to estimate reliably (a median of two occurrences each in the first training window), and its predictions came mainly from one- and two-day contexts. k-nearest neighbours, which matches similar rather than identical shapes, did slightly better on confident days in development, but neither carried a signal into the holdout. Pooling was essential: trained on one stock at a time, most models lost accuracy, and both pattern-based models and LightGBM became significantly worse than reversal.

### 9.4 The open question: confident predictions

The most interesting development result was that every candle-based model's accuracy rose on the days it was most confident about, and that on those days six models beat reversal on the same stock-days. That finding was exploratory, so it was registered as the study's confirmatory hypothesis and tested on the holdout. There it was not confirmed. The average difference from reversal was positive (+0.027) but not significant. More tellingly, the models' own accuracy on their confident holdout days was close to chance: the positive difference arose because reversal did badly on those days, not because the models did well.

Two readings are possible. The development result may have been an artifact of choosing, after the fact, the models and the comparison that looked best, which the holdout then corrected. Or a real but small effect may have been too weak to detect in three years, during a market in which even reversal failed. The holdout was too short to distinguish them; its power was below 50% by design. What the study can say is that any edge on confident days is small, unstable from one market environment to the next, and not large enough to show up reliably out of sample. A forward test on data collected after 2025 would settle the question.

### 9.5 From prediction to trading

Statistical predictability and trading value diverged throughout. Reversal had one of the highest MCCs but lost money at 5 basis points per side, because it changed position almost every day. The support vector machines had the best trading results, largely because they traded less than the other models and leaned long in a rising market. Their returns beyond that market exposure were significant in development, but did not survive realistic execution. Entering positions at the next day's open instead of the close just observed cut their Sharpe ratios by half or more and removed every strategy's significant alpha. Much of these stocks' return accrues overnight, between one day's close and the next day's open, and a daily strategy that cannot trade at the close misses it. For daily candlestick strategies, transaction costs and execution timing matter more than prediction accuracy.

### 9.6 Limitations

**Survivorship bias.** The 22 stocks are the two largest current members of each sector of the S&P 500 that had data in 2002 (Section 2). Choosing stocks by today's size favours companies that did well over the period, which inflates buy-and-hold's return (14.9% a year in development, 29% in the holdout) and, through their net-long exposure, the returns of most strategies. Comparisons between models, and with reversal, are less affected, since all are evaluated on the same stocks, but the absolute returns should not be read as achievable.

**A small, large-cap universe.** The universe contains 22 of the largest and most heavily traded U.S. stocks. Candlestick patterns may behave differently in smaller, less liquid stocks, where prices adjust more slowly. The original study of Caginalp and Laurent covered about 500 stocks, so our replication tests their patterns, not their full sample.

**Free, adjusted data.** Prices come from Yahoo Finance, adjusted for splits and dividends. The data were checked for errors (Section 2), but adjusted prices alter the open, high, low and close of past days by a common factor, and on ex-dividend days they can slightly change comparisons between consecutive candles.

**Execution and costs.** The main backtest trades at the close just observed, which is optimistic. The cost model charges a flat 5 basis points per side and ignores market impact and the cost of borrowing shares to sell short. The next-day-open check (Section 8.7) shows how much the trading results depend on the first assumption; the other two would only lower returns further.

**Few independent events.** Results in volatile markets rest largely on a handful of episodes, above all 2008–09 (Section 8.6). With so few independent crises, regime results describe these episodes more than volatile markets in general.

**A short holdout.** Three years gave the confirmatory test less than a 50% chance of detecting an effect of the size seen in development. A non-rejection is therefore weak evidence either way, as stated in advance.

**Tuning on noisy data.** With about 5,000 labelled stock-days per tuning year, the measured MCC of a setting has a standard error of about 0.014, comparable to the differences between settings. Tuning improved some models and hurt others on the test years (Section 4.5), and the chosen settings varied widely between retune years.

**Practical constraints.** The support vector machine was trained on at most 20,000 stock-days per fold, about a fifth of the largest training windows, because its training time grows with the square of the data. Per-stock training reused the pooled settings, with counts scaled down. Deep sequence models such as LSTMs were not included.

**Analyses added after seeing results.** Several analyses were added after development results had been seen: the reversal baseline, the today's-return model, the three-day window of normalized prices, the market-exposure regression, the replication, the balanced classification loss, and the comparison with reversal on confident days (all logged with dates in the project's decision record). Each was added before the holdout was used. The holdout's confirmatory hypothesis was registered for exactly this reason.

### 9.7 Future work

The clearest next step is a longer and wider test. A universe that includes delisted stocks, from a database such as CRSP, would remove survivorship bias. Smaller stocks would show whether patterns behave differently where prices adjust more slowly. A forward test on data collected after 2025 would give the confident-day hypothesis the power the holdout lacked. Intraday data would allow realistic execution, and deep sequence models and dynamic time warping for k-nearest neighbours would extend the comparison. Any such study should include short-term reversal as a baseline: without it, a candlestick result cannot be interpreted.

### References for this section

- Caginalp, G., & Laurent, H. (1998). The predictive power of price patterns. *Applied Mathematical Finance*, 5(3–4), 181–205.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.
