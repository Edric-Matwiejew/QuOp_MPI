import numpy as np
import pandas as pd
from portfolio import get_stock_data

def qwoa_portfolio(
    n_stocks=4,
    stocks=None,
    risk=0.5,
    penalty=1,
    choose=None,
    start_date="2020-01-01",
    end_date="2020-12-31",
):
    if choose is None:
        choose = n_stocks // 2

    n_qubits = 2 * n_stocks
    data = get_stock_data(n_stocks, start_date, end_date, stocks)

    stock_ret = data.pct_change()
    mean_returns = stock_ret.mean()  # Average returns
    cov_matrix = stock_ret.cov()       # Covariance calculations

    costfunc = []
    portfolio_return = []
    portfolio_std_dev = []

    for k in range(2 ** n_qubits):
        binary = np.zeros(n_qubits // 2)
        temp = list(bin(k))
        temp.pop(0)  # remove '0'
        temp.pop(0)  # remove 'b'
        temp[::] = temp[::-1]
        temp.extend(np.zeros(n_qubits - len(temp)))
        temp[::] = temp[::-1]

        # Convert numeric zeros to string "0"
        for l in range(len(temp)):
            if temp[l] == 0.0:
                temp[l] = "0"

        # Process two bits at a time
        for i in range(0, len(temp), 2):
            if temp[i] == "0" and temp[i + 1] == "0":
                binary[i // 2] = 0
            elif temp[i] == "1" and temp[i + 1] == "1":  # Degenerate state
                binary[i // 2] = -10000
            elif temp[i] == "1" and temp[i + 1] == "0":
                binary[i // 2] = -1
            elif temp[i] == "0" and temp[i + 1] == "1":
                binary[i // 2] = 1

        if sum(binary) == choose:
            portfolio_return.append(np.dot(mean_returns, binary))
            portfolio_std_dev.append(np.dot(binary.T, np.dot(cov_matrix, binary)))
            costfunc.append(
                250 * (risk * portfolio_std_dev[-1] - (1 - risk) * portfolio_return[-1])
            )

    costfunc_df = pd.DataFrame(data=costfunc)
    costfunc_df.to_csv("qwoa_qualities.csv", header=False)

def main():
    qwoa_portfolio(n_stocks=5, choose=2)

if __name__ == "__main__":
    main()

