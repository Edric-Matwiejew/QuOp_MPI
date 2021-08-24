import numpy as np
import copy
from scipy import sparse as __sparse

__zero = np.array([1, 0])
__one = np.array([0,1])

def kron(terms):
    if len(terms) == 0:
        return 1

    if len(terms) == 1:
        return terms[0]

    out = copy.copy(terms[0])
    if isinstance(terms[0], np.ndarray):
        for term in terms[1:]:
            out = np.kron(out, term)
        return out
    else:
        for term in terms[1:]:
            out = __sparse.kron(out, term, format = 'coo')
        return out.tocsr()

def kronp(term, n):
    return kron([term for _ in range(n)])

def string(state):

    terms = []

    for digit in state:
        if str(digit) == '0':
            terms.append(__zero)
        elif str(digit) == '1':
            terms.append(__one)

    return kron(terms)

def __pauli_term(matrix, index, n_qubits):

    kron_terms = [__sparse.coo_matrix(np.identity(2, dtype = np.complex128)) for _ in range(n_qubits)]
    kron_terms[index] = __sparse.coo_matrix(matrix, dtype = np.complex128)

    for i in range(1, n_qubits):
        kron_terms[0] = __sparse.kron(kron_terms[0], kron_terms[i], format = 'coo')

    return kron_terms[0].tocsr()

def I(n_qubits):
    return __sparse.identity(2**n_qubits, dtype = np.complex128, format = 'csr')

def X(index, n_qubits):
    X = np.array([[0,1],[1,0]])
    return __pauli_term(X, index, n_qubits)

def Y(index, n_qubits):
    Y = np.array([[0, -1j], [1j, 0]])
    return __pauli_term(Y, index, n_qubits)

def Z(index, n_qubits):
    Z = np.array([[1, 0], [0, -1]])
    return __pauli_term(Z, index, n_qubits)



