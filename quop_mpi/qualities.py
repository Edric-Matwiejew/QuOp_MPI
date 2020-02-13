import numpy as np

def ordered_integers(N, local_i, local_i_offset, seed = None):
    """
    The array [1, ..., N].

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer.
    """

    return np.asarray(range(local_i_offset, local_i_offset + local_i), dtype = np.float64)

def random_integers(N, local_i, local_i_offset, seed = 0):
    """
    Random integers evenly distributed between :math:`(1, N)`.

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer

    :param seed: Integer to pass to np.random.seed(local_i_offset + seed).
    :type seed: integer, optional, default = 0
    """
    np.random.seed(local_i_offset + seed)
    return np.random.randint(1, N + 1, size = local_i).astype(np.float64)

def random_floats(N, local_i, local_i_offset, seed = 0, low = 0.0, high = 1.0):
    """
    Random floats evenly distributed between :math:`[low, high]`.

    :param N: Size of the distrubted system.
    :type N: integer

    :param local_i: Number of local input QWAO state values, given by qwao.local_i.
    :type local_i: integer

    :param local_i_offset: Offset of the local QWAO state values relative to the zero index of the distributed array. Given by qwao.local_i_offset.
    :type local_i_offset: integer

    :param seed: Integer to pass to np.random.seed(local_i_offset + seed).
    :type seed: integer, optional, default = 0

    :param low: Lower bound.
    :type low: float, optional, default = 0.0

    :param high: Upper bound.
    :type low: float, optional, default = 1.0

    """
    np.random.seed(local_i_offset + seed)
    return np.random.uniform(low = low, high = high, size = local_i)

def portfolio(
        N,
        local_i,
        local_i_offset,
        MPI_communicator,
        stocks = None,
        risk = 0.5,
        penalty = 0.2,
        choose = None,
        start_date = '1/01/2017',
        end_date = '12/31/2018'):

    import pandas_datareader.data as web

    np.random.seed(N)

    n_qubits = int(np.log(N)/np.log(2.0))

    if stocks is None:
        stocks = ['AMP.AX', 'ANZ.AX','AMC.AX', 'BHP.AX', 'BXB.AX',
                 'CBA.AX','CSL.AX','IAG.AX','MQG.AX', 'GMG.AX',
                 'NAB.AX','RIO.AX','SCG.AX','S32.AX','TLS.AX',
                 'WES.AX','BKL.AX','CMW.AX','HUB.AX','ALU.AX',
                 'SUL.AX','TPM.AX','APE.AX','OSH.AX','IPH.AX',
                 'SGR.AX','BEN.AX','HVN.AX','QAN.AX','BKW.AX'][0:n_qubits]

    if choose is None:
        choose = n_qubits//2

    if MPI_communicator.Get_rank() == 0:

        data = web.DataReader(
                stocks,
                data_source = "yahoo",
                start = start_date,
                end = end_date)['Adj Close']

        stock_ret = data.pct_change()
        stock_ret.to_csv('change.csv')
        mean_returns = stock_ret.mean()         # Avg returns and covariance calculations
        cov_matrix = stock_ret.cov()

    else:
        stock_ret = None
        mean_returns = None        # Avg returns and covariance calculations
        cov_matrix = None

    stock_ret = MPI_communicator.bcast(stock_ret, root = 0)
    mean_returns = MPI_communicator.bcast(mean_returns, root = 0)
    cov_matrix = MPI_communicator.bcast(cov_matrix, root = 0)

    costfunc = np.zeros(local_i)
    portfolio_return = np.zeros(local_i)
    portfolio_std_dev = np.zeros(local_i)

    print(n_qubits)

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
