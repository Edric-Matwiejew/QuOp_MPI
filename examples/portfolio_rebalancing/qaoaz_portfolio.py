from quop_mpi import ansatz
from quop_mpi.unitaries import diagonal
from quop_mpi.unitaries import sparse
from quop_mpi.operators.sparse import function
from quop_mpi.params import uniform
from quop_mpi.states import state_function
from quop_mpi.utils import *
from qaoaz_qualities import qaoaz_portfolio
from mpi4py import MPI
import numpy as np

def parity_ring(i, j, n_qubits):
    parity = X(i, n_qubits) @ X(j, n_qubits) \
            + Y(i, n_qubits) @ Y(j, n_qubits)
    return parity

def parity_mixer(qubits, n_qubits):

    odd = 0
    even = 0

    n_subset = len(qubits)

    for i in range(n_subset - 1):

        if (i % 2 != 0):
            odd += parity_ring(qubits[i],
                    qubits[(i + 1) % n_subset],
                    n_qubits)

        elif i % 2 == 0:
            even += parity_ring(qubits[i],
                    qubits[(i + 1) % n_subset],
                    n_qubits)

    mixer = [odd, even]

    if len(qubits) % 2 != 0:
        last = parity_ring(qubits[-1],
                qubits[1],
                n_qubits)

        mixer.append(last)

    return mixer

def mixer(n_qubits):

    short_qubits = [i for i in range(0, n_qubits, 2)]
    long_qubits = [i for i in range(1,n_qubits, 2)]

    short_mixer = parity_mixer(short_qubits, n_qubits)
    long_mixer = parity_mixer(long_qubits, n_qubits)

    return short_mixer + long_mixer

def parity_state(n_qubits, D):
    M = n_qubits//2
    term_1 = kronp(string('01'), D)
    term_2 = kronp(1/np.sqrt(2) * (string('11') + string('00')), M-D)
    state = kron([term_1, term_2])
    return state

COMM = MPI.COMM_WORLD

n_qubits = 8
system_size = 2**n_qubits

UQ = diagonal(
        qaoaz_portfolio,
        parameter_function = uniform)

UW = sparse(
        function,
        operator_kwargs = {
            'function':mixer,
            'args':[n_qubits]},
        parameter_function = uniform)

alg = ansatz(system_size)

alg.set_unitaries([UQ, UW])

alg.set_initial_state(
        state_function,
        {'function':parity_state,
            'args':[n_qubits, 2]})

alg.set_observables(0)

alg.set_log(
        'qaoaz_portfolio_log',
        'qaoaz',
        action = 'w')

alg.benchmark(
        range(1,6),
        1,
        filename = 'qaoaz_portfolio',
        param_persist = True,
        save_action = 'w')
