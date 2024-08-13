"""Predefined :term:`Operator Functions<Operator Function>` for
:class:`quop_mpi.propagator.sparse.unitary`.
"""
from __future__ import annotations
from quop_mpi.__utils.__mpi import __scatter_sparse
import numpy as np

####################################
# imports and classes for type hints
####################################

from mpi4py import MPI
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm

####################################


def hypercube(
    system_size: int, lb: int, ub: int
) -> list[
    list[np.ndarray[np.int64]], list[np.ndarray[np.int64], list[np.ndarray[np.float64]]]
]:
    """Generate a hypercube (QAOA) sparse :term:`mixing unitary`
    :term:`operator`.

    An :term:`Operator Function` for
    :class:`quop_mpi.propagator.sparse.unitary`.

    .. warning:

        valid only for :term:`system size`` corresponding to an integer number of qubits.

    Parameters
    ----------
    system_size : int
        size of the simulated :term:`QVA`, :class:`quop_mpi.Unitary` attribute
    lb : int
        lower index of the :term:`system state` partition,
        :class:`quop_mpi.Unitary` attribute
    ub : int
        upper index of the system state partition, :class:`quop_mpi.Unitary`
        attribute

    Returns
    -------
    list[list[np.ndarray[np.int64]], list[np.ndarray[np.int64], list[np.ndarray[np.float64]]]]
        a CSR partition of the hypercube (QAOA) mixing operator

    Raises
    ------
    RuntimeError
        if :literal:`system_size % 2 != 0`
    """
    from quop_mpi.__lib.csr_generators import csr_generators

    if system_size % 2 != 0:
        raise RuntimeError(
            "Specified system_size not compatible with hypercube mixer. Requires system_size = 2**n for integer n."
        )

    n_qubits = np.log2(system_size)

    W_row_starts, W_col_indexes, W_values = csr_generators.hypercube(
        n_qubits, lb + 1, ub 
    )

    return [W_row_starts], [W_col_indexes], [W_values]


def serial(
    partition_table: list[int],
    MPI_COMM: Intracomm,
    variational_parameters: np.ndarray[np.float64],
    function: Callable,
    *args,
    **kwargs
) -> list[
    list[np.ndarray[np.int64]], list[np.ndarray[np.int64], list[np.ndarray[np.float64]]]
]:
    """Generate an :term:`operator` for a sparse :term:`mixing unitary` using a
    serial Python function.

    An :term:`Operator Function` for
    :class:`quop_mpi.propagator.sparse.unitary`. The :literal:`function` argument must
    be defined via a corresponding :term:`FunctionDict` on initialisation of the
    :literal:`unitary` instance. Additional positional and keyword arguments in the
    :literal:`FunctionDict` are passed the :literal:`function`. The signature for a
    :literal:`function` generating an operator with one or more 
    :term:`operator parameters <operator parameter>` is,

    .. code-block:: python

        def function(variational_parameters, *args, **kwargs) ->
        list[csr_matrix]

    where :literal:`variational_parameters` may be excluded if
    :literal:`unitary.operator_n_params = 0`.

    Parameters
    ----------
    partition_table : list[int]
        describes the parallel partitioning of the :term:`observables` and
        :term:`system state`, :class:`quop_mpi.Unitary` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Unitary` attribute
    variational_parameters : np.ndarray[np.float64]
        a 1-D real array of operator parameters, passed to :literal:`function` if
        :literal:`len(variational_parameters) > 0`, :class:`quop_mpi.Unitary` attribute
    function : Callable
        function returning a list of scipy CSR matrices

    Returns
    -------
    list[list[np.ndarray[np.int64]], list[np.ndarray[np.int64],
    list[np.ndarray[np.float64]]]]
        a CSR matrix partition

    Raises
    ------
    TypeError
        if :literal:`function` does not return a :literal:`list` of scipy CSR matrices
    """
    if MPI_COMM.Get_rank() == 0:

        if len(variational_parameters) > 0:
            input_args = [*args, variational_parameters]
        else:
            input_args = args

        W = function(*input_args, **kwargs)

        if not isinstance(W, list):
            raise TypeError(
                "User supplied function must return a list of scipy CSR matrices."
            )

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

def qmoa_mixer(local_i, local_i_offset, Ns, Gs): 

    from quop_mpi.__lib.csr_generators import csr_generators

    n_dim = len(Ns)

    elements_per_row = 0
    for i in range(n_dim):
        for j in range(2**Ns[i]):
            if Gs[i][j,0] != 0:
                elements_per_row += 1

    G_ptrs = np.empty(n_dim, dtype = np.int64)

    for i, G in enumerate(Gs):
        G_ptrs[i] = G.ctypes.data

    strides = np.empty(n_dim, dtype = np.int64)
    strides[0] = 1
    for i in range(1, n_dim):
        strides[i] = strides[i - 1] * 2**Ns[i - 1]

    n_grid_points = [2**N for N in Ns]

    W_row_starts, W_col_indexes, W_values = csr_generators.qmoa_mixer(  local_i,
                                                                        local_i_offset,
                                                                        n_grid_points,
                                                                        np.ones(n_dim, dtype = np.float64),
                                                                        G_ptrs,
                                                                        elements_per_row)

    return [W_row_starts], [W_col_indexes], [W_values]
