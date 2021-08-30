import matplotlib.pyplot as plt
from scipy.io import mmwrite
import networkx as nx
from quop_mpi.utils import *
import numpy as np

def parity_ring(i, j, n_qubits):
    parity = X(i, n_qubits) @ X(j, n_qubits) \
            + Y(i, n_qubits) @ Y(j, n_qubits)
    return parity

def parity_mixer(qubits, n_qubits):

    odd = 0
    even = 0

    n_subset = len(qubits)

    for i in range(n_subset):

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



mixers = parity_mixer([i for i in range(0, 4)], 4)

graph = nx.convert_matrix.from_scipy_sparse_matrix(np.real(mixers[0] + mixers[1]))

n_qubits = 4

labels = {}
for i in range(2**n_qubits):
    labels[i] = str(np.binary_repr(i, width = n_qubits))

nx.draw(graph, pos = nx.planar_layout(graph), labels = labels, with_labels = True)
plt.savefig('graph')
mmwrite('parity', np.real(mixers[0] + mixers[1]))


