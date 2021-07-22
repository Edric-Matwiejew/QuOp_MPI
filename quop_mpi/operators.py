from importlib import import_module
import numpy as np
from quop_mpi.__utils.__mpi import __scatter_sparse, __scatter_1D_array

def circulant_complete(
        system_size,
        local_i,
        local_i_offset):

    fqwoa_mpi = import_module('quop_mpi.__lib.fqwoa_mpi')

    graph_array = np.ones(system_size, dtype = np.float64)
    graph_array[0] = 0

    return fqwoa_mpi.graph_eigenvalues(graph_array, local_i, local_i_offset)


def circulant_graph(
        system_size,
        local_i,
        local_i_offset,
        i = None):
    '''
    Returns an array c1=c2=...=ci=1, others equals to '0'
    array size: N
    ci(N, 1) will return circle graph
    ci(N, int(N/2)+1) will return complete graph
    '''

    fqwoa_mpi = import_module('quop_mpi.__lib.fqwoa_mpi')

    if (i > system_size//2) or (i is None):

        graph_array = np.ones(system_size, dtype = np.float64)
        graph_array[0] = 0

    else:

        circle_limit = system_size//2 + 1

        if i < circle_limit:
            i = circle_limit

        graph_array = np.zeros(system_size, dtype = np.float64)

        for j in range(1,i+1):
            graph_array[j] = 1
            graph_array[system_size - j]=1

    eigenvalues = fqwoa_mpi.graph_eigenvalues(
            graph_array,
            local_i,
            local_i_offset)

    return eigenvalues

def sparse_hypercube(
        system_size,
        lb,
        ub):

    from quop_mpi.__lib.mixers_mpi import hypercube

    if system_size % 2 != 0:
        raise RuntimeError("Specified system_size not compatible with hypercube mixer. Requires system_size = 2**n for integer n.")

    n_qubits = np.log2(system_size)

    W_row_starts, W_col_indexes, W_values = hypercube(
        n_qubits,
        lb + 1,
        ub + 1)

    return [W_row_starts], [W_col_indexes], [W_values]

def sparse_function(
        operator_function,
        partition_table,
        variational_parameters,
        MPI_COMM,
        **kwargs):

    if MPI_COMM.Get_rank() == 0:

        W = operator_function(*variational_parameters, **kwargs)

        row_starts = []
        col_indexes = []
        values = []

        for w in W:
            row_starts.append(w.indptr)
            col_indexes.append(w.indices)
            values.append(w.values)

    else:

        row_starts = None
        col_indexes = None
        values = None

    return __scatter_sparse(row_starts, col_indexes, values, partition_table, MPI_COMM)

def diagonal_uniform(
        system_size,
        partition_table,
        seed,
        MPI_COMM,
        low = 0,
        high = 1):

    if MPI_COMM.Get_rank() == 0:

        #seed = 0
        np.random.seed(seed)

        diagonal = np.random.uniform(low = low, high = high, size = system_size)
        #print(diagonal, flush = True)

    else:

        diagonal = None

    return __scatter_1D_array(diagonal, partition_table, MPI_COMM, np.float64)

def diagonal_function(
        operator_function,
        partition_table,
        variational_parameters,
        MPI_COMM,
        **kwargs):

    if MPI_COMM.Get_rank() == 0:
        operator = operator_function(*variational_parameters, **kwargs)
    else:
        operator = None

    return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)

def diagonal_csv(
        system_size,
        partition_table,
        MPI_COMM,
        filename = None,
        **kwargs):

    import pandas as pd

    if MPI_COMM.Get_rank() == 0:
        data_df = pd.read_csv(filename, **kwargs)
        diagonals = data_df.to_numpy(dtype = np.complex128)
    else:
        diagonals = None

    return __scatter_1D_array(diagonals, partition_table, MPI_COMM, np.float64)


def diagonal_hdf5(
        local_i,
        local_i_offset,
        MPI_COMM,
        filename = None,
        dataset_name = None):

    import h5py as h5

    f = h5.File(filename, 'r')

    operator = np.array(f[dataset_name][local_i_offset:local_i_offset + local_i]).view(np.float64)

    f.close()

    return operator

