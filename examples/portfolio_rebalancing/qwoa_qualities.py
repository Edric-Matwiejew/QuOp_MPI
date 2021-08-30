import numpy as np
import pandas as pd
import pandas_datareader.data as web


def qwoa_portfolio(
        n_stocks = 4,
        stocks=None,
        risk=0.5,
        penalty=1,
        choose=None,
        start_date='1/01/2017',
        end_date='12/31/2018'
        ):

    if stocks is None:

        stocks = ['AMP.AX', 'ANZ.AX', 'AMC.AX', 'BHP.AX', 'BXB.AX',
                  'CBA.AX', 'CSL.AX', 'IAG.AX', 'MQG.AX', 'GMG.AX',
                  'NAB.AX', 'RIO.AX', 'SCG.AX', 'S32.AX', 'TLS.AX',
                  'WES.AX', 'BKL.AX', 'CMW.AX', 'HUB.AX', 'ALU.AX',
                  'SUL.AX', 'TPM.AX', 'APE.AX', 'OSH.AX', 'IPH.AX',
                  'SGR.AX', 'BEN.AX', 'HVN.AX', 'QAN.AX', 'BKW.AX'][0:n_stocks]

    if choose is None:

        choose = n_stocks // 2

    n_qubits = 2 * n_stocks

    data = web.DataReader(
        stocks,
        data_source="yahoo",
        start=start_date,
        end=end_date)['Adj Close']

    stock_ret = data.pct_change()
    mean_returns = stock_ret.mean()  # Avg returns and covariance calculations
    cov_matrix = stock_ret.cov()

    costfunc = []
    portfolio_return = []
    portfolio_std_dev = []

    for k in range(2**n_qubits):

        binary = np.zeros(n_qubits // 2)
        temp = list(bin(k))
        temp.pop(0)  # removes 0b start to binary
        temp.pop(0)
        temp[::] = temp[::-1]
        temp.extend(np.zeros(n_qubits - len(temp)))
        temp[::] = temp[::-1]

        for l in range(len(temp)):
            if temp[l] == 0.0:
                temp[l] = '0'

        for i in range(0, len(temp), 2):  # converts binary string to int in an array
            if temp[i] == '0' and temp[i + 1] == '0':
                binary[i // 2] = 0
            elif temp[i] == '1' and temp[i + 1] == '1': ##DEGENERATE STATE
                binary[i // 2] = -10000
            elif temp[i] == '1' and temp[i + 1] == '0':
                binary[i // 2] = -1
            elif temp[i] == '0' and temp[i + 1] == '1':
                binary[i // 2] = 1

        if sum(binary) == choose:
            portfolio_return.append(np.dot(mean_returns, binary))
            portfolio_std_dev.append(np.dot(binary.T, np.dot(cov_matrix, binary)))
            costfunc.append(250*(risk * portfolio_std_dev[-1] - (1 - risk) * portfolio_return[-1]))

    costfunc_df = pd.DataFrame(data = costfunc)
    costfunc_df.to_csv('qwoa_qualities.csv')

def main():
    qwoa_portfolio(n_stocks = 5, choose=2)

if __name__ == "__main__":
    main()



