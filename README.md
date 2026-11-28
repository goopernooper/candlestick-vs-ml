# candlestick-vs-ml
Comparing candlestick pattern mining with machine learning for next-day stock direction prediction.
This repository compares candlestick pattern–based models (a variable-order pattern miner and k-nearest neighbors) against standard machine learning algorithms for predicting next-day direction on about 25 US large-cap stocks, 2002–2025. All models share the same encodings and are evaluated with walk-forward validation, naive baselines, and transaction costs. Undergraduate research at Queensborough Community College, advised by Professor Yusuf Danisman.
# Stock universe 
For each of the 11 GICS sectors, select the two largest current S&P 500 companies by market capitalization that have daily price data on January 2, 2002. The next two eligible companies in each sector are recorded as alternates; if a selected stock fails the data validation checks, it is replaced by the first alternate in its sector. SPY is included as the market proxy. Because selection uses present-day index members, the universe is subject to survivorship bias.
# Run it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goopernooper/candlestick-vs-ml/blob/main/notebooks/colab_review.ipynb)

`notebooks/colab_review.ipynb` runs the project in Colab without installing anything locally. It clones this repository, installs the pinned versions from `requirements.txt` into a separate environment, runs the test suite, recomputes the statistical tests and figures from the committed results, and shows the main tables (about 5 minutes; no price data needed). A second, optional part re-runs the whole experiment, which needs a privately shared copy of the `data/raw` snapshot, because the Yahoo Finance data cannot be redistributed in this public repository.
