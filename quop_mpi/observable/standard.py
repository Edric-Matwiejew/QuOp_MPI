from importlib import import_module
import numpy as np
from ..__utils.__mpi import __scatter_1D_array


def serial(
    partition_table,
    MPI_COMM,
    function=None,
    args=None,
    kwargs=None,
):

    """
    Defines :math:`\hat{Q}` via a serial function. The serial `function` is called at `rank = 0` of MPI communicator `MPI_COMM` and its output is distributed over `MPI_COMM` as described by `partition_table`. Argument `partition_table` is a class attribute of the :class:`Unitary` class.

    :param partition_table: Describes the parallel partitioning scheme.
    :type partition_table: array, integer

    :param MPI_COMM: MPI communicator
    :type MPI_COMM: MPI4py communicator object

    :param function: Function that returns :math:`\\text{diag}(\hat{Q})`.
    :type function: callable

    :param args: Positional arguments associated with `function`.
    :type args: optional, list, default = None

    :param kwargs: Keyword arguments associated with `function`.
    :type kwargs: optional, dictionary, default = None
    """

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if MPI_COMM.Get_rank() == 0:

        if len(args) == 0 and len(kwargs) == 0:
            operator = function()
        elif len(kwargs) != 0 and len(args) == 0:
            operator = function(*kwargs)
        elif len(args) != 0 and len(kwargs) == 0:
            operator = function(*args)
        else:
            operator = function(*args, **kwargs)

        operator_array = isinstance(operator[0], np.ndarray)

        if operator_array:
            n_terms = len(operator)
        else:
            n_terms = 1

    else:
        operator = None
        n_terms = None

    n_terms = MPI_COMM.bcast(n_terms, 0)

    if n_terms > 1:

        terms = []

        for i in range(n_terms):
            if MPI_COMM.Get_rank() == 0:
                terms.append(
                    __scatter_1D_array(
                        operator[i], partition_table, MPI_COMM, np.float64
                    )
                )
            else:
                terms.append(
                    __scatter_1D_array(None, partition_table, MPI_COMM, np.float64)
                )

        return terms

    else:

        return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)


def csv(system_size, partition_table, MPI_COMM, filename=None, **kwargs):

    """Import :math:`\\text{diag}(\hat{Q})` from a CSV file.

    :param system_size: Size of the quantum system :math:`N`.
    :type system_size: integer

    :param partition_table: Array describing the parallel partitioning scheme.
    :type partition_table: array, integer

    :param MPI_COMM: MPI communicator over which :math:`\\text{diag}(\hat{Q})` is partitioned.
    :type MPI_COMM: MPI4py communicator object

    :param partition_table: Array describing the parallel partitioning scheme.
    :type partition_table: array, integer

    :param filename: The location of the CSV file.
    :type filename: string

    :param \*\*kwargs: An arbitrary number of keyword arguments passed to the Pandas `read_csv` function.
    """

    import pandas as pd

    if MPI_COMM.Get_rank() == 0:
        data_df = pd.read_csv(filename, **kwargs)
        diagonals = data_df.to_numpy(dtype=np.complex128)
    else:
        diagonals = None

    return __scatter_1D_array(diagonals, partition_table, MPI_COMM, np.float64)


def hdf5(partition_table, MPI_COMM, filename=None, dataset_name=None):

    """Import :math:`\\text{diag}(\hat{Q})` from a HDF5 file.

    :param local_i: Number of elements in the local partition of :math:`\\text{diag}(\hat{Q})`.
    :type local_i: integer

    :param local_i_offset: Number of elements preceding the local partition.
    :param local_i_offset: integer

    :param MPI_COMM: MPI communicator over which :math:`\\text{diag}(\hat{Q})` is partitioned.
    :type MPI_COMM: MPI4py communicator object

    :param filename: Path to the HDF5 file.
    :type filename: string

    :param dataset_name: Path to :math:`\\text{diag}(\hat{Q})` in the HDF5 file.
    :type dataset_name: string
    """

    import h5py as h5

    if MPI_COMM.rank == 0:
        f = h5.File(filename, "r")

        operator = np.array(
            f[dataset_name],
            dtype = np.float64
        )

        f.close()

    else:
        operator = None

    return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)


def array(system_size, partition_table, MPI_COMM, array=None):

    """
    Define :math:`\\text{diag}(\hat{Q})` using an array defined at MPI rank = 0.

    :param system_size: Size of the quantum system :math:`N`.
    :type system_size: integer

    :param partition_table: Array describing the parallel partitioning scheme.
    :type partition_table: array, integer

    :param MPI_COMM: MPI communicator over which :math:`\\text{diag}(\hat{Q})` is partitioned.
    :type MPI_COMM: MPI4py communicator object

    :param array: Array defining :math:`\\text{diag}(\hat{Q})`.
    :type array: array, float

    """

    return __scatter_1D_array(array, partition_table, MPI_COMM, np.float64)
