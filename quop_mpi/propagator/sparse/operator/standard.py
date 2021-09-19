from quop_mpi.__utils.__mpi import __scatter_sparse
import numpy as np

def hypercube(
        system_size,
        lb,
        ub):
    """
    Generate a parallel partition of a :math:`N \\times N` hypercube mixing operator where for :math:`n` qubits :math:`N = 2^n`.

    :param system_size: Size of the quantum system :math`N`.
    :type system_size: integer

    :param lb: Lower bound of the local partition.
    :type lb: integer

    :param ub: Upper bound of the local partition.
    :type ub: integer

    :return: Local parallel parallel of the hypercube.
    :rtype: QuOp_MPI sparse matrix arrays
    """

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
    """Generate a mixing operator, or sequence of mixing operators, from a
    serial function.

    :param partition_table: Parallel partitioning scheme.
    :type partition_table: array, integer

    :param MPI_COMM: MPI communicator
    :type MPI_COMM: MPI4py communicator object

    :param variational_parameters: Variational parameters :math:`\\theta` assocaited with the mixing operator.
    :type variational_parameters: array, float

    :param function: Serial function that generates the mixing operators.
    :type function: callable

    :param args: Positional arguments assocaited with `function`.
    :type args: optional, list, default = None

    :param kwargs: Keyword arguments assocaited with `function`.
    :type kwargs: optional, dictionary, default = None
    """

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
