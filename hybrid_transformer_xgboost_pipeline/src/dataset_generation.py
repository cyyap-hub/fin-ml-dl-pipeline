import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta_classic as ta


def build_dataset(ticker="^GSPC", start_date="2000-01-01", end_date="2026-08-01"):
    # Gathers ticker data, macro indicators, and calculates technical indicators features to build a dataset ready for our models.

    # Fetch Raw Open, High, Low, Close, Volume Data from ticker data
    ticker_df = yf.download(tickers=ticker, start=start_date, end=end_date, progress=False)

    # Flatten multi-index columns if using newer versions of yfinance
    if isinstance(ticker_df.columns, pd.MultiIndex):
        ticker_df.columns = ticker_df.columns.get_level_values(0)


    # RSI (14-day)
    ticker_df.ta.rsi(length=14, append=True)

    # MACD
    ticker_df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Bollinger Bands
    ticker_df.ta.bbands(length=20, std=2, append=True)

    # Fetch Macroeconomic Features
    macro_tickers = ['^VIX', '^TNX']
    macro_df = yf.download(macro_tickers, start=start_date, end=end_date, progress=False)['Close']

    if isinstance(macro_df.columns, pd.MultiIndex):
        macro_df.columns = macro_df.columns.get_level_values(1)


    # Merging all the Features
    final_df = ticker_df.join(macro_df, how='left')

    final_df = final_df.rename(columns={'^TNX': '10Y_Yield', '^VIX': 'VIX_Close',
        'RSI_14': 'RSI',
        'MACD_12_26_9': 'MACD',
        'BBL_20_2.0': 'BB_Lower',
        'BBM_20_2.0': 'BB_Middle',
        'BBU_20_2.0': 'BB_Upper'})

    # Keep only the specifically requested columns
    requested_columns = [
        'Close', 'Volume',
        'RSI', 'MACD', 'BB_Lower', 'BB_Middle', 'BB_Upper',
        'VIX_Close', '10Y_Yield'
    ]
    final_df = final_df[requested_columns].copy()

    # To create our Target Variable which is Tomorrow's Log Return and Lags of Prior Days' Log Return
    final_df['Log_Return_Lag1'] = np.log(final_df['Close'] / final_df['Close'].shift(1))
    final_df['Target_Log_Return'] = final_df['Log_Return_Lag1'].shift(-1)
    final_df['Log_Return_Lag2'] = final_df['Log_Return_Lag1'].shift(1)
    final_df['Log_Return_Lag3'] = final_df['Log_Return_Lag1'].shift(2)
    final_df['Log_Return_Lag4'] = final_df['Log_Return_Lag1'].shift(3)
    final_df['Log_Return_Lag5'] = final_df['Log_Return_Lag1'].shift(4)
    final_df['Log_Return_Lag30'] = final_df['Log_Return_Lag1'].shift(30)

    return final_df



if __name__ == "__main__":
    # Fetch S&P 500 data up to recent dates
    dataset = build_dataset(ticker="^GSPC",
        start_date="2000-01-01",
        end_date="2026-08-01")

    print("\n--- Dissertation Dataset (First 5 Rows) ---")
    pd.set_option('display.max_columns', None)  # Show all columns in terminal

    # To save our Engineered Dataset into a CSV file
    dataset.to_csv('dataset.csv')


