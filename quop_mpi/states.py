import numpy as np
from mpi4py import MPI
from quop_mpi.__utils.__mpi import __scatter_1D_array

def equal(
        system_size,
        local_i):

    if not local_i is None:
        initial_state = np.empty(local_i, np.complex128)
        initial_state[:] = 1/np.sqrt(np.float64(system_size))
    else:
        initial_state = None

    return initial_state

def basis_states(
        alloc_local,
        local_i,
        local_i_offset,
        basis_states = [0]):

    initial_state = np.zeros(self.alloc_local, np.complex128)

    n_basis_states = len(self.basis_states)

    for state in basis_states:
        if (state > local_i_offset) and (state <= local_i_offset + local_i):
            initial_state[vertex] = 1.0/np.sqrt(n_basis_states, dtype = np.float64)

    return initial_state

def state(
        alloc_local,
        local_i,
        local_i_offset,
        MPI_COMM,
        state = None,
        normalized = True):

    initial_state = np.empty(alloc_local, dtype = np.complex128)

    initial_state[:local_i] = np.array(state[local_i_offset:local_i_offset + local_i], np.complex128)

    if not normalized:
        normalization = MPI_COMM.allreduce(np.dot(np.conjugate(state), state), op = MPI.SUM)
        initial_state = initial_state/np.sqrt(normalization)

    return initial_state

def state_function(
        partition_table,
        MPI_COMM,
        function = None,
        args = [],
        kwargs = {}):

    if MPI_COMM.Get_rank() == 0:

        state = np.array(function(*args, **kwargs), dtype = np.complex128)

    else:

        state = None

    return __scatter_1D_array(state, partition_table, MPI_COMM, np.complex128)
