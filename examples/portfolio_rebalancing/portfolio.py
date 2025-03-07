import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time

Stocks = [
    "AMP.AX", "ANZ.AX", "BHP.AX", "CBA.AX", "CSL.AX",
    "IAG.AX", "MQG.AX", "NAB.AX", "RIO.AX", "SCG.AX",
    "S32.AX", "TLS.AX", "WES.AX", "QAN.AX", "WOW.AX",
    "WBC.AX", "COL.AX", "GMG.AX", "SGR.AX", "BEN.AX",
    "HVN.AX", "BXB.AX", "ORG.AX", "NCM.AX", "ASX.AX"
]

def get_stock_data(n_stocks, start_date, end_date, stocks=None, seed=0, retries=3):
    """
    Fetches adjusted close prices for a subset of ASX stocks.

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

            return stock_data["Close"]

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    print("Failed to retrieve data after multiple attempts.")
    return pd.DataFrame()

if __name__ == "__main__":
    df = get_stock_data(n_stocks=5, start_date="2024-01-01", end_date="2024-03-01")
    print(df.head())
