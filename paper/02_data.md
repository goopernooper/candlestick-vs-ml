## 2. Data

We use daily open, high, low and close prices from Yahoo Finance, retrieved with the yfinance library (version 1.7.0) on September 28, 2026. Prices are adjusted for splits and dividends. The study period runs from January 2, 2002 to December 31, 2025, which starts after U.S. exchanges completed the move to decimal prices in 2001; fractional tick sizes would distort the small candle bodies that patterns such as the doji depend on. We also retrieve October through December 2001 so that rolling volatility measures are fully initialized on the first day of the study period.

The stock universe is selected by a fixed rule rather than by hand. For each of the 11 GICS sectors, we take the two largest current S&P 500 members by market capitalization that have price data on January 2, 2002, giving 22 stocks (Table 1). SPY serves as the market proxy, and the CBOE Volatility Index (VIX) is used in robustness checks. Because selection uses present-day index members, the universe is subject to survivorship bias: 134 of the 500 current constituents (27%) were excluded for lacking data in January 2002, among them Alphabet, Meta, Tesla, Visa and Netflix. The sample therefore leans toward older, established firms. The series for Linde and Prologis include predecessor-company history before mergers in 2018 and 2011, and neither shows a discontinuity at its merger date.

Each series contains 6,103 trading days, about 6,040 of them in the study period, for roughly 133,000 stock-days in total. Every series passed checks for internally consistent prices (high at or above the open and close, low at or below them), missing or duplicate observations, and alignment with SPY's trading calendar. We reviewed all 125 daily moves larger than 20% in either the close-to-close return or the overnight gap; all moves in the 22 study stocks were matched to news events or confirmed against a second data source, and no missed splits or erroneous prices were found. One anomaly was kept as recorded: on October 10, 2008, BRK-B opened about 22% below its previous close in a disorderly first trade that also appears in Google Finance's data, while its close agrees with Berkshire's Class A shares. We additionally compared ten randomly chosen 60-day windows against Stooq, an independent data source.

Yahoo Finance's terms do not permit redistributing its data, so the raw files are not included in the repository. Instead, the repository contains a manifest recording the row count, date range and SHA-256 hash of every file, which allows anyone with the same snapshot to verify that it is identical.

**Table 1. Stock universe**

| Sector | Stocks |
|---|---|
| Communication Services | VZ, DIS |
| Consumer Discretionary | AMZN, HD |
| Consumer Staples | WMT, COST |
| Energy | XOM, CVX |
| Financials | BRK-B, JPM |
| Health Care | LLY, JNJ |
| Industrials | CAT, GE |
| Information Technology | NVDA, AAPL |
| Materials | LIN, NEM |
| Real Estate | WELL, PLD |
| Utilities | NEE, SO |
