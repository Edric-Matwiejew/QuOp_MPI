from quop_mpi.__utils.__mpi import __scatter_sparse
import numpy as np

def hypercube(
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

def serial(
        partition_table,
        MPI_COMM,
        variational_parameters,
        function = None,
        args = [],
        kwargs = {}
        ):

    if MPI_COMM.Get_rank() == 0:

        if variational_parameters is not None:
            input_args = [*args, variational_parameters]
        else:
            input_args = args

        W = function(*input_args, **kwargs)

        row_starts = []
        col_indexes = []
        values = []

        for w in W:
            row_starts.append((w.tocsr()).indptr + 1)
            col_indexes.append((w.tocsr()).indices + 1)
            values.append((w.tocsr()).data)

    else:

        row_starts = None
        col_indexes = None
        values = None

    return __scatter_sparse(row_starts, col_indexes, values, partition_table, MPI_COMM)
