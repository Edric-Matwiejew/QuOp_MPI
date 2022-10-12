from importlib import import_module
import numpy as np
from mpi4py import MPI
from ..__utils.__mpi import __scatter_1D_array

def equal(
        system_size,
        local_i):
    """Generate :math:`|\psi_0\\rangle_\\text{ANZ}` as an equal superposition
    over :math:`N` basis states.

    :param system_size: Size of the quantum system :math:`N`.
    :type system_size: integer

    :param local_i: Number of elements in the local partition of :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type local_i: integer
    """
    if not local_i is None:
        initial_state = np.empty(local_i, np.complex128)
        initial_state[:] = 1/np.sqrt(np.float64(system_size))
    else:
        initial_state = None

    return initial_state

def basis(
        local_i,
        local_i_offset,
        basis_states = [0]):
    """Generate :math:`|\psi_0\\rangle_\\text{ANZ}` localised over a subset of
    basis states.

    :param local_i: Number of elements in the local partition of :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type local_i: integer

    :param local_i_offset: Offset of the local parallel partition relative to its position in :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type local_i_offset: integer

    :param basis_states: List of basis states.
    :type basis_states: optional, list, default = [0]
    """
    initial_state = np.zeros(self.local_i, np.complex128)

    n_basis_states = len(self.basis_states)

    for state in basis_states:
        if (state > local_i_offset) and (state <= local_i_offset + local_i):
            initial_state[vertex] = 1.0/np.sqrt(n_basis_states, dtype = np.float64)

    return initial_state

def array(
        local_i,
        local_i_offset,
        MPI_COMM,
        state = None,
        normalize = True):
    """
    Define :math:`|\psi_0\\rangle_\\text{ANZ}` via an array at MPI rank = 0.

    :param local_i: Number of elements in the local partition of :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type local_i: integer

    :param local_i_offset: Offset of the local parallel partition relative to its position in :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type local_i_offset: integer

    :param MPI_COMM: MPI communicator over which :math:`|\psi_0\\rangle_\\text{ANZ}` is distributed.
    :type MPI_COMM: MPI4py communicator object
    
    :param state: Array defining :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type state: array, complex

    :param normalize: If True, normalize `array`.
    :type normalize: optional, boolean, default = True

    """

    initial_state = np.empty(local_i, dtype = np.complex128)

    initial_state[:local_i] = np.array(state[local_i_offset:local_i_offset + local_i], np.complex128)

    if not normalize:
        normalization = MPI_COMM.allreduce(np.dot(np.conjugate(state), state), op = MPI.SUM)
        initial_state = initial_state/np.sqrt(normalization)

    return initial_state

def serial(
        partition_table,
        MPI_COMM,
        function = None,
        args = None,
        kwargs = None):
    """Define :math:`|\psi_0\\rangle_\\text{ANZ}` using a serial function.

    :param partition_table: Array describing the parallel partitioning scheme.
    :type partition_table: array, integer

    :param MPI_COMM: MPI communicator over which :math:`|\psi_0\\rangle_\\text{ANZ}` is distributed.
    :type MPI_COMM: MPI4py communicator object


    :param function: Serial function that returns :math:`|\psi_0\\rangle_\\text{ANZ}`.
    :type function: callable

    :param kwargs: Keyword arguments associated with `function`.
    :type kwargs: optional, dictionary, default = None
    """

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if MPI_COMM.Get_rank() == 0:

        state = np.array(function(*args, **kwargs), dtype = np.complex128)

    else:

        state = None

    return __scatter_1D_array(state, partition_table, MPI_COMM, np.complex128)

def position_grid(
        alloc_local,
        local_i,
        local_i_offset,
        MPI_COMM,
        Ns = None,
        deltasq = None,
        minsq = None,
        func = None):

    if MPI.COMM_WORLD.Get_rank() == 0:

        maxq = np.array(minsq, dtype = np.float64) + np.array(Ns, dtype = np.float64) * np.array(deltasq)

        search_domain = [
                domain for domain in zip(minsq, maxq)
                ]

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

        std = np.array(n_dim*[np.exp(1)/np.sqrt(2)], dtype = np.float64)
        velocity = np.array(n_dim*[0], dtype = np.float64)
    
        state = 0
    
        for x_i, m, s, v in zip(x, mean, std, velocity): 
            state += np.exp(1j*v*x_i)*np.exp((-(x_i - m)**2)/(2*s**2))/(np.sqrt(np.pi)*s)
        return state

    if func == None:
        func = squeezed

    fCQAOA = import_module('quop_mpi.__lib.fCQAOA')

    strides = np.empty(len(Ns), dtype = int)
    strides[-1] = 1
    for i in range(len(Ns) - 2, -1, -1):
        strides[i] = strides[i + 1]*Ns[i]

    state = np.empty(shape = [alloc_local], dtype = np.complex128)

    fCQAOA.continuous.dist_vector(
            func,
            Ns,
            strides,
            deltasq,
            minsq,
            local_i_offset,
            state)

    norm = np.sum(np.abs(state[:local_i])**2)
    norm = MPI_COMM.allreduce(norm, op = MPI.SUM)
    state = state/np.sqrt(norm)

    return state
