import numpy as np
import pandas as pd
import os

# Try to import yfinance, but don't fail if it's not available
try:
    import yfinance as yf
    import requests
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

import time

Stocks = [
    "AMP.AX", "ANZ.AX", "BHP.AX", "CBA.AX", "CSL.AX",
    "IAG.AX", "MQG.AX", "NAB.AX", "RIO.AX", "SCG.AX",
    "S32.AX", "TLS.AX", "WES.AX", "QAN.AX", "WOW.AX",
    "WBC.AX", "COL.AX", "GMG.AX", "SGR.AX", "BEN.AX",
    "HVN.AX", "BXB.AX", "ORG.AX", "NCM.AX", "ASX.AX"
]

# Sample stock data columns available in the fallback CSV
SAMPLE_STOCKS = ["IAG.AX", "BHP.AX", "BEN.AX", "COL.AX", "TLS.AX"]


def get_sample_stock_data(n_stocks, stocks=None, seed=0):
    """
    Load sample stock data from bundled CSV file.
    
    This is used as a fallback when yfinance is unavailable or fails.
    The sample data contains 2020 historical prices for 5 ASX stocks.
    
    :param n_stocks: Number of stocks to retrieve
    :param stocks: Custom list of stocks (will be mapped to available sample stocks)
    :param seed: Random seed for reproducibility
    :return: DataFrame of adjusted close prices
    """
    sample_file = os.path.join(os.path.dirname(__file__), "sample_stock_data.csv")
    
    if not os.path.exists(sample_file):
        raise FileNotFoundError(
            f"Sample stock data file not found: {sample_file}\n"
            "Please ensure sample_stock_data.csv exists in the portfolio_rebalancing directory."
        )
    
    df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
    
    # Select the requested number of stocks
    available_stocks = list(df.columns)
    n_available = min(n_stocks, len(available_stocks))
    
    np.random.seed(seed)
    selected_indices = np.random.choice(len(available_stocks), n_available, replace=False)
    selected_stocks = [available_stocks[i] for i in selected_indices]
    
    print(f"Using sample data for: {', '.join(selected_stocks)}.", flush=True)
    
    return df[selected_stocks]


def get_stock_data(n_stocks, start_date, end_date, stocks=None, seed=0, retries=3):
    """
    Fetches adjusted close prices for a subset of ASX stocks.
    
    First attempts to download from Yahoo Finance. If that fails (due to
    network issues, missing SQLite driver, etc.), falls back to bundled
    sample data.

    :param n_stocks: Number of stocks to retrieve
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :param stocks: Custom list of stocks (if None, defaults to `Stocks`)
    :param seed: Random seed for reproducibility
    :param retries: Number of retry attempts in case of failures
    :return: DataFrame of adjusted close prices
    """

    np.random.seed(seed)

    if stocks is None:
        np.random.shuffle(Stocks)
        selected_stocks = Stocks[:n_stocks]
    else:
        selected_stocks = stocks[:n_stocks]

    # Check if yfinance is available
    if not YFINANCE_AVAILABLE:
        print("yfinance not available, using sample data.", flush=True)
        return get_sample_stock_data(n_stocks, stocks, seed)

    print(f"Retrieving adjusted close prices for: {', '.join(selected_stocks)}.", flush=True)

    session = requests.Session()

    for attempt in range(retries):
        try:
            stock_data = yf.download(
                selected_stocks,
                start=start_date,
                end=end_date,
                progress=True,
                session=session,
                auto_adjust=True,
                threads=False
            )

            # Check if we got valid data
            if stock_data.empty or stock_data["Close"].isna().all().all():
                raise ValueError("Downloaded data is empty or all NaN")

            return stock_data["Close"]

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    # All retries failed, fall back to sample data
    print("Failed to retrieve live data, falling back to sample data.", flush=True)
    return get_sample_stock_data(n_stocks, stocks, seed)


if __name__ == "__main__":
    df = get_stock_data(n_stocks=5, start_date="2024-01-01", end_date="2024-03-01")
    print(df.head())
