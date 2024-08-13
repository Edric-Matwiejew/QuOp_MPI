from __future__ import annotations
import numpy as np
from quop_mpi.__utils.__mpi import __scatter_1D_array

####################################
# imports and classes for type hints
####################################

from mpi4py import MPI
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm
iterable = Iterable

####################################

def uniform(
        system_size: int,
        partition_table: int,
        seed: int,
        MPI_COMM: Intracomm,
        low: float = 0,
        high: float = 1) -> np.ndarray[np.float64]:
    """Generate the diagonal of a :term:`phase-shift unitary` :term:`operator`
    from a uniform distribution.

    An :term:`Operator Function` for the
    :class:`quop_mpi.propagator.diagonal.unitary` class. The :literal:`low` and :literal:`high`
    arguments may be defined in a corresponding :term:`FunctionDict` on
    initialisation of the :literal:`unitary` instance.

    Parameters
    ----------
    system_size : int
        :term:`system size` of the simulated :term:`QVA`, :class:`quop_mpi.Unitary` attribute
    partition_table : int
        describes the parallel partitioning scheme, :class:`quop_mpi.Unitary` attribute
    seed : int
        seeds the random number generator, :class:`quop_mpi.Unitary` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Unitary` attribute
    low : float, optional
        lower bound of the uniform distribution (inclusive), by default 0
    high : float, optional
        upper bound of the uniform distribution (exclusive), by default 1

    Returns
    -------
    ndarray[float64]
        a 1-d real array or list of 1-D real arrays containing a :literal:`local_i`
        elements of the :term:`operator` diagonal with global index offset
        :literal:`local_i_offset`
    """
    if MPI_COMM.Get_rank() == 0:

        np.random.seed(seed)

        diagonal = np.random.uniform(low = low, high = high, size = system_size)

    else:

        diagonal = None

    return __scatter_1D_array(diagonal, partition_table, MPI_COMM, np.float64)

