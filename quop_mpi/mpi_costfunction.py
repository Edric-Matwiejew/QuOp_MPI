import pandas as pd
import numpy as np
import pandas_datareader.data as web
from mpi4py import MPI

def portfolio(
        N,
        local_i,
        local_i_offset,
        stocks,
        MPI_communicator,
        risk = 0.5,
        penalty = 1.0,
        choose = 7,
        start_date = '1/01/2017',
        end_date = '12/31/2018'):

    np.random.seed(N)

    n_qubits = int(np.sqrt(N))

    if MPI_communicator.Get_rank() == 0:

        data = web.DataReader(
                stock,
                data_source = "yahoo",
                start = start_date,
                end = end_date)['Adj Close']

        stock_ret = data.pct_change()
        mean_returns = stock_ret.mean()         # Avg returns and covariance calculations
        cov_matrix = stock_ret.cov()

    stock_ret = MPI_communicator.bcast(stock_ret, root = 0)
    mean_returns = MPI_communicator.bcast(mean_returns, root = 0)
    cov_matrix = MPI_communicator.bcast(cov_matrix, root = 0)

    costfunc = np.zeros(local_i)
    portfolio_return = np.zeros(local_i)
    portfolio_std_dev = np.zeros(local_i)

    for k in range(local_i):

        binary = np.zeros(n_qubits)
        temp = list(bin(k + local_i_offset))
        temp.pop(0)                 # removes 0b start to binary
        temp.pop(0)
        temp[::] = temp[::-1]
        temp.extend(np.zeros(n_qubits - len(temp)))
        temp[::] = temp[::-1]

        for i in range(len(binary)):        # converts binary string to int in an array
            binary[i] = int(temp[i])


        portfolio_return[k] = np.dot(mean_returns, binary)
        portfolio_std_dev[k] = np.dot(binary.T, np.dot(cov_matrix, binary))
        costfunc[k] = risk * portfolio_return[k] - (1 - risk) * portfolio_std_dev[k] - penalty * np.power((choose - np.sum(binary)), 2)

    return costfunc

#stock =['AMP.AX', 'ANZ.AX','AMC.AX', 'BHP.AX', 'BXB.AX','CBA.AX','CSL.AX','IAG.AX','MQG.AX',
#        'GMG.AX','NAB.AX','RIO.AX','SCG.AX','S32.AX','TLS.AX','WES.AX']

stock =['AMP.AX', 'ANZ.AX','AMC.AX', 'BHP.AX']

comm = MPI.COMM_WORLD

print(portfolio(16, 16, 0, stock, comm))

print(portfolio(16, 4, 4, stock, comm))

print(portfolio(16, 4, 12, stock, comm))

