import numpy as np

def uniform(
        n_params,
        seed,
        low = 0,
        high = 2*np.pi):

    np.random.seed(seed)

    return np.random.uniform(low = low, high = high, size = n_params)


def normal(
        system_size,
        n_params,
        operator,
        MPI_COMM,
        seed,
        loc = 1):

    from mpi4py import MPI

    local_sum = np.sum(operator, dtype = np.float64)
    mean = local_sum / system_size
    np.random.seed(seed)

    return np.random.normal(loc = loc, size = n_params)
