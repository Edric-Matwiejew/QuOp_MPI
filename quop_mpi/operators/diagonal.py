from importlib import import_module
import numpy as np
from quop_mpi.__utils.__mpi import __scatter_1D_array

def random(
        system_size,
        partition_table,
        seed,
        MPI_COMM,
        low = 0,
        high = 1):

    if MPI_COMM.Get_rank() == 0:

        np.random.seed(seed)

        diagonal = np.random.uniform(low = low, high = high, size = system_size)

    else:

        diagonal = None

    return __scatter_1D_array(diagonal, partition_table, MPI_COMM, np.float64)

def serial(
        partition_table,
        MPI_COMM,
        variational_parameters,
        function = None,
        args = [],
        kwargs = {},
        ):

    if MPI_COMM.Get_rank() == 0:

        if variational_parameters is not None:
            input_args = [*args, variational_parameters]
        else:
            input_args = args

        if len(input_args) == 0 and len(kwargs) == 0:
            operator = function()
        elif len(kwargs) != 0 and len(args) == 0:
            operator = function(*kwargs)
        elif len(args) != 0 and len(kwargs) == 0:
            operator = function(*input_args)
        else:
            operator = function(*input_args, **kwargs)

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
                terms.append(__scatter_1D_array(operator[i], partition_table, MPI_COMM, np.float64))
            else:
                terms.append(__scatter_1D_array(None, partition_table, MPI_COMM, np.float64))

        return terms

    else:

        return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)

def csv(
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


def hdf5(
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

def array(
        system_size,
        partition_table,
        MPI_COMM,
        array = None
        ):

    return __scatter_1D_array(diagonal, partition_table, MPI_COMM, np.float64)

#def serial_array(
#        local_i,
#        local_i_offset,
#        array = None
#        ):
#
#    return array[local_i_offset:local_i_offset + local_i]
