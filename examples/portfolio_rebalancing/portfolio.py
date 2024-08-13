import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()

Stocks = [
    "AMP.AX", "ANZ.AX", "AMC.AX", "BHP.AX",
    "BXB.AX", "CBA.AX", "CSL.AX", "IAG.AX",
    "MQG.AX", "GMG.AX", "NAB.AX", "RIO.AX",
    "SCG.AX", "S32.AX", "TLS.AX", "WES.AX",
    "BKL.AX", "CMW.AX", "HUB.AX", "ALU.AX",
    "SUL.AX", "APE.AX", "IPH.AX", 
    "SGR.AX", "BEN.AX", "HVN.AX", "QAN.AX", 
    "BKW.AX"
    ]
    
def get_stock_data(
    n_stocks,
    start_date,
    end_date,
    stocks,
    seed = 0
):

    np.random.seed(seed)

    if stocks is None:
        np.random.shuffle(Stocks)
    
    
    stock_data = []

    print(f'Retreving adjusted close prices for: {", ".join(Stocks[:n_stocks])}.', flush = True)

    return yf.download(Stocks[:n_stocks], start=start_date, end=end_date)["Adj Close"]
