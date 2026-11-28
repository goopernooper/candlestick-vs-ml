## 1. Introduction

A daily candlestick summarizes a stock's trading day in four prices: the open, high, low and close. Candlestick charting holds that particular shapes, and short sequences of them, signal what the price will do next. A long lower wick after a fall is read as buyers returning; a large down day that engulfs the previous up day is read as the end of a rally. These patterns are a standard part of technical analysis, built into charting software and trading libraries, and widely used by traders for short-term decisions.

Whether they work is less clear. Caginalp and Laurent (1998) reported that eight three-day reversal patterns predicted the direction of S&P 500 stocks with very high significance, but most later studies of large U.S. stocks found no value after costs (Marshall et al., 2006; Horton, 2009; Duvinage et al., 2013), while studies of Taiwanese stocks found some patterns profitable (Lu et al., 2012; Lu, 2014). In parallel, machine learning models trained on daily prices have reported accuracy and trading profits well above chance (Krauss et al., 2017; Fischer & Krauss, 2018), including models given candlestick features (Lin et al., 2021a, 2021b). When Fischer and Krauss examined what their networks had learned, however, the stocks they traded showed a short-term reversal profile, and a simple reversal rule explained part of their returns.

That observation points to three gaps in this literature. First, most candlestick patterns describe a move followed by a sign of its reversal, and short-term reversal is a well-documented effect in stock returns (Jegadeesh, 1990; Lehmann, 1990). Yet patterns and models are usually compared with chance or with buy-and-hold, not with a rule that simply predicts each stock will reverse today's move, so a study cannot tell pattern recognition from reversal. Second, pattern-based methods and machine learning methods have not been compared on the same data, labels and costs, so it is unknown whether either approach, or either way of describing candles, is better. Third, signals from different stocks often occur on the same days and move together, and many studies try many patterns or models, so tests that treat every signal as independent, or report only the best result, overstate the evidence (Sullivan et al., 1999; Harvey et al., 2016).

This study addresses these gaps with four questions:

1. Do daily candles predict next-day direction better than naive baselines, and in particular better than short-term reversal?
2. Which works better, pattern-based models or machine learning models, and does it matter how candles are encoded?
3. Does any predictive edge survive transaction costs and realistic execution?
4. Do the development-period findings hold on later data that played no part in building the models?

We predict the next-day direction of 22 large U.S. stocks, two from each sector of the S&P 500, from 2002 to 2025. Two pattern-based models, a variable-order pattern miner that matches exact sequences of candle symbols and *k*-nearest neighbours that matches similar candle shapes, are compared with four machine learning models: LightGBM, a random forest, a support vector machine and a multilayer perceptron. The machine learning models are given the same candles in three forms: the 61 named patterns of the TA-Lib library, a continuous description of each candle's body, wicks and gap over three days, and normalized prices over the same three days. Every model family is tuned with the same budget and evaluated in 18 years of walk-forward tests from 2005 to 2022 against five naive baselines, including the reversal rule. Comparisons are tested on one loss value per day, which combines all 22 stocks' predictions, using Diebold–Mariano tests, the Model Confidence Set and a test for superior predictive ability. The analysis plan was fixed and tagged before a final three-year holdout, 2023–2025, was used once. We also repeat Caginalp and Laurent's test with their own pattern definitions.

The main findings are as follows.

- **Candles carry a small signal, and it is almost all reversal.** The best models reach a Matthews correlation coefficient (MCC) of 0.033, and thirteen of fifteen candle-based models significantly beat always predicting an up day. None is significantly better than the reversal rule (MCC 0.023), over all days or within any volatility regime.
- **Named patterns are the weakest encoding, and pattern-based models trail machine learning.** A three-day window of continuous candle shape gives the best or equal-best results for every machine learning model. The 61 named patterns give the lowest for three of the four. The pattern-based models rank below every machine learning model given the same candles.
- **No strategy beats buy-and-hold after costs.** Before costs, almost every candle-based strategy does. At 5 basis points per side, only the two support vector machines still earn more than buy-and-hold, not significantly, and entering at the next day's open instead of the observed close removes every strategy's significant return beyond its market exposure.
- **The development signal does not survive the holdout.** On 2023–2025, every model's MCC is within 0.02 of zero. A pre-registered test of the most promising development result, that models beat reversal on their most confident days, was not confirmed (*p* = 0.062).
- **The original positive result does not replicate.** With Caginalp and Laurent's definitions and test, their eight patterns show no predictive power on our stocks in 2002–2022.

Section 2.6 sets out the study's contributions relative to earlier work. In brief, it is a controlled comparison of pattern-based and machine learning models on identical data with a reversal baseline, tests designed for correlated signals and many models, and a pre-registered holdout. Its conclusion is negative, and the study was designed so that a negative result would be informative: it shows where apparent candlestick predictability comes from and why earlier studies have disagreed. All code, the analysis plan and the results files are public.

The rest of the paper is organized as follows. Section 2 reviews related work. Section 3 describes the data, Section 4 the candle encodings and prediction target, and Section 5 the models. Sections 6 and 7 describe the validation design and the evaluation metrics, baselines and backtest, and Section 8 the statistical tests. Section 9 reports the results, and Section 10 discusses them, with their limitations.

### References for this section

- Caginalp, G., & Laurent, H. (1998). The predictive power of price patterns. *Applied Mathematical Finance*, 5(3–4), 181–205.
- Duvinage, M., Mazza, P., & Petitjean, M. (2013). The intra-day performance of market timing strategies and trading systems based on Japanese candlesticks. *Quantitative Finance*, 13(7), 1059–1070.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Horton, M. J. (2009). Stars, crows, and doji: The use of candlesticks in stock selection. *Quarterly Review of Economics and Finance*, 49(2), 283–294.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689–702.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.
- Lin, Y., Liu, S., Yang, H., & Wu, H. (2021a). Stock trend prediction using candlestick charting and ensemble machine learning techniques with a novelty feature engineering scheme. *IEEE Access*, 9, 101433–101446.
- Lin, Y., Liu, S., Yang, H., Wu, H., & Jiang, B. (2021b). Improving stock trading decisions based on pattern recognition using machine learning technology. *PLOS ONE*, 16(8), e0255558.
- Lu, T.-H. (2014). The profitability of candlestick charting in the Taiwan stock market. *Pacific-Basin Finance Journal*, 26, 65–78.
- Lu, T.-H., Shiu, Y.-M., & Liu, T.-C. (2012). Profitable candlestick trading strategies—The evidence from a new perspective. *Review of Financial Economics*, 21(2), 63–68.
- Marshall, B. R., Young, M. R., & Rose, L. C. (2006). Candlestick technical trading strategies: Can they create value for investors? *Journal of Banking & Finance*, 30(8), 2303–2323.
- Sullivan, R., Timmermann, A., & White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance*, 54(5), 1647–1691.
