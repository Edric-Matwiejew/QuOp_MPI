from __future__ import annotations
from importlib import import_module
import numpy as np
from mpi4py import MPI
from ..__utils.__mpi import __scatter_1D_array

####################################
# imports and classes for type hints
####################################

from mpi4py import MPI
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm

####################################

def equal(system_size: int, local_i: int) -> np.ndarray[np.complex128]:
    """Generate an equal superposition over all :term:`system states<system state>`.

    The default :term:`Initial State Function` of the :class:`quop_mpi.Ansatz`
    class.

    Parameters
    ----------
    system_size : int
        size of the simulated :term:`QVA`, :class:`quop_mpi.Ansatz` attribute
    local_i : int
        size of the local :term:`system state` partitions, 
        :class:`quop_mpi.Ansatz` attribute

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` :term:`initial state` values with
        global index offset :literal:`local_i_offset` (see :class:`quop_mpi.Ansatz`)
    """

    if local_i is not None:
        initial_state = np.empty(local_i, np.complex128)
        initial_state[:] = 1 / np.sqrt(np.float64(system_size))
    else:
        initial_state = None

    return initial_state


def basis(
    local_i: int, local_i_offset: int, basis_states: list[int] = None
) -> np.ndarray[np.complex128]:
    """Generate an equal superposition over a subset of basis states.

    An :term:`Initial State Function`. The :literal:`basis_states` argument can be
    specified by passing a :term:`FunctionDict` to
    :meth:`quop_mpi.Ansatz.set_initial_state`.

    Parameters
    ----------
    local_i : int
        size of the local :term:`system state` partitions,
        :class:`quop_mpi.Ansatz` attribute
    local_i_offset : int
        global index offset of the local system state partitions,
        :class:`quop_mpi.Ansatz` attribute
    basis_states : list[int], optional
        global indexes specifying an equal superposition over a subset of
        states, by default [0]

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` :term:`initial state` values with
        global index offset :literal:`local_i_offset` (see :class:`quop_mpi.Ansatz` )
    """
    if basis_states is None:
        basis_states = [0]

    initial_state = np.zeros(local_i, np.complex128)

    n_basis_states = len(basis_states)

    for state in basis_states:
        if (state > local_i_offset) and (state <= local_i_offset + local_i):
            initial_state[state] = 1.0 / np.sqrt(n_basis_states, dtype=np.float64)

    return initial_state

def array(
    local_i: int,
    local_i_offset: int,
    MPI_COMM: Intracomm,
    state: np.ndarray[np.complex128],
    normalize: bool = True,
) -> np.ndarray[np.complex128]:
    """Define the :term:`initial state` using a Numpy array.

    An :term:`Initial State Function`. The :literal:`normalize` argument can be
    specified by passing a :term:`FunctionDict` to 
    :meth:`~quop_mpi.Ansatz.set_initial_state` .

    Parameters
    ----------
    local_i : int
        size of the local :term:`system state` partitions,
        :class:`quop_mpi.Ansatz` attribute
    local_i_offset : int
        global index offset of the local system state partitions,
        :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator of the :term:`QVA` simulation,
        :class:`quop_mpi.Ansatz` attribute
    state : ndarray[complex128]
        A 1-D array of :term:`system size` initial state values 
    normalize : bool, optional
        wether to normalize :literal:`state`, by default True

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` :term:`initial state` values with
        global index offset :literal:`local_i_offset` (see :class:`quop_mpi.Ansatz`)
    """
    initial_state = np.empty(local_i, dtype=np.complex128)

    initial_state[:local_i] = np.array(
        state[local_i_offset : local_i_offset + local_i], np.complex128
    )

    if not normalize:
        normalization = MPI_COMM.allreduce(
            np.dot(np.conjugate(state), state), op=MPI.SUM
        )
        initial_state = initial_state / np.sqrt(normalization)

    return initial_state

def serial(
    partition_table: list[int], MPI_COMM: Intracomm, function: Callable, *args, **kwargs
) -> np.ndarray[np.complex128]:
    """Generate the :term:`initial state` using a serial Python function.

    An :term:`Initial State Function`. The :literal:`function` argument must be
    specified by passing a :term:`FunctionDict` to
    :meth:`~quop_mpi.Ansatz.set_initial_state`. Additional positional and keyword
    arguments in the :literal:`FunctionDict` are passed to :literal:`function`.

    Parameters
    ----------
    partition_table : list[int]
        1-D array describing the global partitioning scheme,
        :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator of the :term:`QVA` simulation,
        :class:`quop_mpi.Ansatz` attribute
    function : Callable
        a Python function returning a 1-D complex array of :term:`system size`
        initial state values

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` :term:`initial state` values with
        global index offset :literal:`local_i_offset` (see :class:`quop_mpi.Ansatz`)
    """

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if MPI_COMM.Get_rank() == 0:

        state = np.array(function(*args, **kwargs), dtype=np.complex128)

    else:

        state = None

    return __scatter_1D_array(state, partition_table, MPI_COMM, np.complex128)


def position_grid(
    alloc_local: int,
    local_i: int,
    local_i_offset: int,
    MPI_COMM: Intracomm,
    Ns: list[int],
    deltas: list[float],
    mins: list[float],
    function: Callable,
) -> np.ndarray[np.complex128]:
    """Generate an :term:`initial state` discrete Cartesian coordinates.

    An :term:`Observables Function`. Arguments :literal:`Ns`, :literal:`deltas`, :literal:`mins` and
    :literal:`function` must be passed to :meth:`quop_mpi.Ansatz.set_observables` in a
    :term:`FunctionDict`.

    The :literal:`function` argument must take an :literal:`len(Ns)` -dimensional coordinate
    and return the complex amplitude of the initial state at that coordinate.

    If :literal:`function` is :literal:`None`, :literal:`position_grid` generates a squeezed
    Gaussian state with its mean situated at a randomly generated coordinate
    that has a distance of at least :literal:`length * 0.125` from the boundaries of
    the grid (where :literal:`length` is the length in each coordinate).

    Parameters
    ----------
    alloc_local : int
        size of the array containing the local partition of the system,
        :class:`quop_mpi.Ansatz` attribute
    local_i : int
        number of initial state values in local partition of the 
        :term:`system state`, :class:`quop_mpi.Ansatz` attribute
    local_i_offset : int
        global index offset of the local system state partition,
        :class:`quop_mpi.Ansatz` attribute
    MPI_COMM : Intracomm
        MPI communicator, :class:`quop_mpi.Ansatz` attribute
    Ns : list[int]
        number of grid points in each dimension of the Cartesian grid
    deltas : list[float]
        step size in each coordinate
    mins : list[float]
        minimum value in each coordinate
    function : Callable
        a Python function returning the value of the initial state at each grid
        point

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` :term:`initial state` values with
        global index offset :literal:`local_i_offset` (see :class:`quop_mpi.Ansatz`)
    """

    if MPI.COMM_WORLD.Get_rank() == 0:

        maxq = np.array(mins, dtype=np.float64) + np.array(
            Ns, dtype=np.float64
        ) * np.array(deltas)

        search_domain = list(zip(mins, maxq))

        mean = []
        for domain in search_domain:
            mean.append(
                np.random.uniform(
                    low=domain[0] - domain[0] * 0.125,
                    high=domain[1] - domain[1] * 0.125,
                )
            )
        mean = np.array(mean, dtype=np.float32)
    else:
        mean = None

    mean = MPI.COMM_WORLD.bcast(mean, root=0)

    def squeezed(x):

        n_dim = len(x)

        std = np.array(n_dim * [np.exp(1) / np.sqrt(2)], dtype=np.float64)
        velocity = np.array(n_dim * [0], dtype=np.float64)

        state = 0

        for x_i, m, s, v in zip(x, mean, std, velocity):
            state += (
                np.exp(1j * v * x_i)
                * np.exp((-((x_i - m) ** 2)) / (2 * s**2))
                / (np.sqrt(np.pi) * s)
            )
        return state

    if function is None:
        function = squeezed

    fCQAOA = import_module("quop_mpi.__lib.fCQAOA")

    strides = np.empty(len(Ns), dtype=int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1] * Ns[i]

    state = np.empty(shape=[alloc_local], dtype=np.complex128)

    fCQAOA.continuous.dist_vector(
        function, Ns, strides, deltas, mins, local_i_offset, state
    )

    norm = np.sum(np.abs(state[:local_i]) ** 2)
    norm = MPI_COMM.allreduce(norm, op=MPI.SUM)
    state = state / np.sqrt(norm)

    return state