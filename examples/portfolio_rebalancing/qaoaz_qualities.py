import numpy as np
from mpi4py import MPI


def qaoaz_portfolio(
    system_size,
    local_i,
    local_i_offset,
    MPI_COMM,
    stocks=None,
    risk=0.5,
    penalty=1.5,
    start_date="1/01/2017",
    end_date="12/31/2018",
):

    n_qubits = int(np.log(system_size) / np.log(2.0))
    n_stocks = n_qubits // 2

    if MPI_COMM.rank == 0:

        import pandas_datareader.data as web

        if int(2 ** (n_qubits)) != int(system_size):
            return print("System size does not correspond to qubit hilbert dimension")

        if stocks is None:
            stocks = [
                "AMP.AX",
                "ANZ.AX",
                "AMC.AX",
                "BHP.AX",
                "BXB.AX",
                "CBA.AX",
                "CSL.AX",
                "IAG.AX",
                "MQG.AX",
                "GMG.AX",
                "NAB.AX",
                "RIO.AX",
                "SCG.AX",
                "S32.AX",
                "TLS.AX",
                "WES.AX",
                "BKL.AX",
                "CMW.AX",
                "HUB.AX",
                "ALU.AX",
                "SUL.AX",
                "TPM.AX",
                "APE.AX",
                "OSH.AX",
                "IPH.AX",
                "SGR.AX",
                "BEN.AX",
                "HVN.AX",
                "QAN.AX",
                "BKW.AX",
            ][0:n_stocks]

        data = web.DataReader(
            stocks, data_source="yahoo", start=start_date, end=end_date
        )["Adj Close"]

        stock_ret = data.pct_change()
        mean_returns = stock_ret.mean()  # Avg returns and covariance calculations
        cov_matrix = stock_ret.cov()

    else:

        stock_ret = None
        mean_returns = None  # Avg returns and covariance calculations
        cov_matrix = None

    stock_ret = MPI_COMM.bcast(stock_ret, root=0)
    mean_returns = MPI_COMM.bcast(mean_returns, root=0)
    cov_matrix = MPI_COMM.bcast(cov_matrix, root=0)

    costfunc = np.zeros(local_i)
    portfolio_return = np.zeros(local_i)
    portfolio_std_dev = np.zeros(local_i)
    check = np.zeros(local_i)

    for k in range(local_i):

        if k == 0:
            temp = np.repeat("0", n_qubits)
            binary = np.zeros(n_qubits // 2)
        else:
            binary = np.zeros(n_qubits // 2)
            temp = list(bin(k + local_i_offset))
            temp.pop(0)  # removes 0b start to binary
            temp.pop(0)
            temp[::] = temp[::-1]
            temp.extend(np.zeros(n_qubits - len(temp)))
            temp[::] = temp[::-1]
            for l in range(len(temp)):
                if temp[l] == 0.0:
                    temp[l] = "0"

        for i in range(0, len(temp), 2):  # converts binary string to int in an array
            if temp[i] == "0" and temp[i + 1] == "0":
                binary[i // 2] = 0
            elif temp[i] == "1" and temp[i + 1] == "1":
                binary[i // 2] = 0
            elif temp[i] == "1" and temp[i + 1] == "0":
                binary[i // 2] = -1
            elif temp[i] == "0" and temp[i + 1] == "1":
                binary[i // 2] = 1

        portfolio_return[k] = np.dot(mean_returns, binary)
        portfolio_std_dev[k] = np.dot(binary.T, np.dot(cov_matrix, binary))
        costfunc[k] = 250 * (
            risk * portfolio_std_dev[k] - (1 - risk) * portfolio_return[k]
        )

    return costfunc
