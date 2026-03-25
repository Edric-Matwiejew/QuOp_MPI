"""Observable operator functions using random diagonal matrices."""

from __future__ import annotations

import numpy as np

####################################
# imports and classes for type hints
####################################
from mpi4py import MPI

from .._utils._mpi import __scatter_1d_array

Intracomm = MPI.Intracomm

####################################


def uniform(
    system_size: int,
    partition_table: list[int],
    seed: int,
    MPI_COMM: Intracomm,  # noqa: N803
    low: float = 0,
    high: float = 1,
) -> np.ndarray[np.float64]:
    """Generate random :term:`observables` from a uniform distribution.

    An :term:`Observables Function`. The ``low`` and ``high`` arguments can be
    passed to :meth:`quop_mpi.ansatz.set_observables` in a :term:`FunctionDict`.

    Parameters
    ----------
    system_size : int
        the :term:`size<system size>` of the simulated :term:`system <QVA>`,
        class:`quop_mpi.ansatz` attribute
    partition_table : list[int]
        1-D array describing the global partitioning scheme,
        :class:`quop_mpi.ansatz` attribute
    seed : int
        sets the seed of the random number generator, :class:`quop_mpi.ansatz`
        attribute
    MPI_COMM : Intracomm
        MPI intracommunicator, :class:`quop_mpi.ansatz` attribute
    low : float, optional
        lower bound of the generated observable values (inclusive), by default ``0``
    high : float, optional
        upper bound of the genereated observable values (exclusive), by default ``1``

    Returns
    -------
    np.ndarray[float64]
        ``local_i`` observable values with global index offset
        ``local_i_offset`` (see :meth:`quop_mpi.ansatz`)
    """

    if MPI_COMM.Get_rank() == 0:

        np.random.seed(seed)

        diagonal = np.random.uniform(low=low, high=high, size=system_size)

    else:

        diagonal = None

    return __scatter_1d_array(diagonal, partition_table, MPI_COMM, np.float64)
