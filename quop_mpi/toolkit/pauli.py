import numpy as np
from scipy import sparse as __sparse

def __pauli_term(matrix, index, n_qubits):

    kron_terms = [__sparse.coo_matrix(np.identity(2)) for _ in range(n_qubits)]
    kron_terms[index] = __sparse.coo_matrix(matrix)

    for i in range(1, n_qubits):
        kron_terms[0] = __sparse.kron(kron_terms[0], kron_terms[i], format = 'coo')

    return kron_terms[0].tocsr()

def I(n_qubits):
    """Generate :math:`I \in \mathbb{R}^{n \\times n}`.

    :param n_qubits: :math:`n`
    :type n_qubits: integer

    :return:  :math:`I \in \mathbb{R}^{n \\times n}`
    :rtype: Scipy sparse matrix
    """
    return __sparse.identity(2**n_qubits, format = 'csr')

def X(index, n_qubits):
    """Generate a sparse operator that applies a Pauli-X gate to qubit `index`
    in a state space of `n_qubits` qubits.

    :param index: Index of the target qubit.
    :type index: integer

    :param n_qubits: Number of qubits.
    :type n_qubits: integer
    """
    x = np.array([[0,1],[1,0]])
    return __pauli_term(x, index, n_qubits)

def Y(index, n_qubits):
    """Generate a sparse operator that applies a Pauli-Y gate to qubit `index`
    in a state space of `n_qubits` qubits.

    :param index: Index of the target qubit.
    :type index: integer

    :param n_qubits: Number of qubits.
    :type n_qubits: integer
    """

    y = np.array([[0, -1j], [1j, 0]])
    return __pauli_term(y, index, n_qubits)

def Z(index, n_qubits):
    """Generate a sparse operator that applies a Pauli-Z gate to qubit `index`
    in a state space of `n_qubits` qubits.

    :param index: Index of the target qubit.
    :type index: integer

    :param n_qubits: Number of qubits.
    :type n_qubits: integer
    """
    z = np.array([[1, 0], [0, -1]])
    return __pauli_term(z, index, n_qubits)


