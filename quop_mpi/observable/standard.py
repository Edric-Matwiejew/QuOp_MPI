from __future__ import annotations
import numpy as np
from ..__utils.__mpi import __scatter_1D_array

####################################
# imports and classes for type hints
####################################

from mpi4py import MPI
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm

####################################


def serial(
    partition_table: list[int],
    MPI_COMM: Intracomm,
    function: Callable,
    *args,
    **kwargs
) -> np.ndarray[np.float64]:
    """Generate :term:`observables` using a serial python function.

    An :term:`Observables Function`. The :literal:`function` argument must be passed to
     :meth:`quop_mpi.Ansatz.set_observables` in a :term:`FunctionDict`. Additional
    positional and keyword arguments in the :literal:`FunctionDict` are passed to
     :literal:`function`.

    Parameters
    ----------
    partition_table : list[int]
        1-D array describing the global partitioning scheme, 
         :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    function : Callable
        Python function returning a 1-D real array of :term:`system size` 
         observable values

    Returns
    -------
    ndarray[float64]
        :literal:`local_i` observable values with global index offset
        :literal:`local_i_offset` (see :meth:`quop_mpi.Ansatz`)
    """

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if MPI_COMM.Get_rank() == 0:
        operator = function(*args, **kwargs)
        operator_array = isinstance(operator[0], np.ndarray)
        n_terms = len(operator) if operator_array else 1
    else:
        operator = None
        n_terms = None

    n_terms = MPI_COMM.bcast(n_terms, 0)

    if n_terms <= 1:
        return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)
    terms = []

    for i in range(n_terms):
        if MPI_COMM.Get_rank() == 0:
            terms.append(
                __scatter_1D_array(operator[i], partition_table, MPI_COMM, np.float64)
            )
        else:
            terms.append(
                __scatter_1D_array(None, partition_table, MPI_COMM, np.float64)
            )

    return terms


#TODO Update docstring
def csv(
    partition_table: list[int], MPI_COMM: Intracomm, *args, **kwargs) -> np.ndarray[np.float64]:
    """Load :term:`observables` from a :literal:`*.csv` using `pandas
    <https://pandas.pydata.org/>`_.

    An :term:`Observables Function`. The :literal:`filename` argument must be passed to
    :meth:`quop_mpi.Ansatz.set_observables` in a :term:`FunctionDict`. Additional
    keyword arguments in the :literal:`FunctionDict` are passed to the `pandas.read_csv
    <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_ method.

    Parameters
    ----------
    partition_table : list[int]
        1-D array describing the global partitioning scheme,
        :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    filename : str
        path to a :literal:`*csv` file

    Returns
    -------
    ndarray[float64]
        :literal:`local_i` observable values with global index offset
        :literal:`local_i_offset` (see :meth:`quop_mpi.Ansatz`)
    """

    import pandas as pd

    if MPI_COMM.Get_rank() == 0:
        data_df = pd.read_csv(*args, **kwargs)
        diagonals = data_df.to_numpy(dtype=np.complex128)
    else:
        diagonals = None

    MPI_COMM.barrier()

    return __scatter_1D_array(diagonals, partition_table, MPI_COMM, np.float64)


def hdf5(
    partition_table: list[int],
    MPI_COMM: Intracomm,
    filename: str,
    dataset_name: str,
    **kwargs
) -> np.ndarray[np.float64]:
    """Load :term:`observables` from a :literal:`*.h5` file using `HDF5 for Python <https://docs.h5py.org/en/latest/index.html>`_.

    An :term:`Observables Function`. The :literal:`filename` and :literal:`dataset_name`
    arguments must be passed to :meth:`quop_mpi.Ansatz.set_observables` in a
    :term:`FunctionDict`. Additional positional and keyword arguments in the
    :literal:`FunctionDict` are passed to the `h5py.File <https://docs.h5py.org/en/latest/high/file.html>`_ method.

    Parameters
    ----------
    partition_table : list[int]
        1-D array describing the global partitioning scheme,
        :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    filename : str
        path to a :literal:`*.h5` file
    dataset_name : str
        path to the dataset in :literal:`filename` containing an ndarray[float64] of
        :term:`system size` observables.

    Returns
    -------
    ndarray[float64]
        :literal:`local_i` observable values with global index offset
        :literal:`local_i_offset` (see :meth:`quop_mpi.Ansatz`)
    """

    import h5py as h5

    if MPI_COMM.rank == 0:
        f = h5.File(filename, "r", **kwargs)

        operator = np.array(f[dataset_name], dtype=np.float64)

        f.close()

    else:
        operator = None

    return __scatter_1D_array(operator, partition_table, MPI_COMM, np.float64)


def array(
    partition_table: list[int],
    MPI_COMM: Intracomm,
    array: Union[list[float], np.ndarray[float]],
) -> np.ndarray[np.float64]:
    """Define :term:`observables` with a NumPy ndarray.

    An :term:`Observables Function`. The :literal:`array` argument must be passed to
    :meth:`quop_mpi.Ansatz.set_observables` in a :term:`FunctionDict` . 

    Parameters
    ----------
    partition_table : list[int]
        1-D array describing the global partitioning scheme, :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    array : Union[list[float], ndarray[float]]
        a 1-D real array containing :term:`system size` observable values

    Returns
    -------
    ndarray[float64]
        :literal:`local_i` observable values with global index offset :literal:`local_i_offset` (see :meth:`quop_mpi.Ansatz`)
    """

    return __scatter_1D_array(array, partition_table, MPI_COMM, np.float64)
