from importlib import import_module
import numpy as np
from quop_mpi.__utils.__mpi import __scatter_1D_array

def uniform(
        system_size,
        partition_table,
        seed,
        MPI_COMM,
        low = 0,
        high = 1):

    """Generates :math:`\hat{O})` where the :math:`o_{ii}` are sampled from a
    random distribution uniformly distributed between (`low`, `high`].

    :param system_size: Size of the quantum system :math:`N`.
    :type system_size: integer

    :param partition_table: Parallel partitioning scheme.
    :type partition_table: array, integer

    :param seed: Set the state of the random number generator.
    :type seed: integer

    :param MPI_COMM: MPI communicator over which :math:`\\text{diag}(\hat{O})` is distributed.
    :type MPI_COMM: MPI4py communicator object

    :param low: Lower bound of the uniform distribution.
    :type low: optional, float, default = 0

    :param high: Upper bound of the uniform distribution.
    :type high: optional, float, default = 1
    """


    if MPI_COMM.Get_rank() == 0:

        np.random.seed(seed)

        diagonal = np.random.uniform(low = low, high = high, size = system_size)

    else:

        diagonal = None

    return __scatter_1D_array(diagonal, partition_table, MPI_COMM, np.float64)

