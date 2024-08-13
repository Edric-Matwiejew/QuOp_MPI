from __future__ import annotations
import numpy as np
from scipy import sparse as __sparse

y = __sparse.coo_matrix(np.array([[0, -1j], [1j, 0]]))
x = __sparse.coo_matrix(np.array([[0,1],[1,0]]))
z = __sparse.coo_matrix(np.array([[1, 0], [0, -1]]))

def __pauli_term(matrix, index, n_qubits):

    kron_terms = []
    if index > 0:
        kron_terms.append(__sparse.identity(2**index))
    kron_terms.append(matrix)
    if index != (n_qubits - 1):
        kron_terms.append(__sparse.identity(2**(n_qubits - index - 1)))
    
    for i in range(1, len(kron_terms)):
        kron_terms[0] = __sparse.kron(kron_terms[0], kron_terms[i])

    return kron_terms[0].tocsr()

def I(n_qubits: int) -> 'csr_matrix':
    """Generate a sparse identity matrix of size ``2 ** n_qubits``.

    Parameters
    ----------
    n_qubits: int
        generate the identity operator for ``n_qubits``

    Returns
    -------
    csr_matrix 
        the identity operator for ``n_qubits``
    """
    return __sparse.identity(2**n_qubits, format = 'csr')

def X(index: int, n_qubits: int) -> 'csr_matrix':
    """Generate the Pauli X operator acting on qubit ``index`` in a system of
    ``n_qubits``.

    Parameters
    ----------
    index : int
        index of the qubit to which the X operator is applied
    n_qubits : int
        total number of qubits in the system

    Returns
    -------
    csr_matrix
        the Pauli X operator acting on qubit ``index`` in a system of ``n_qubits``
    """
    return __pauli_term(x, index, n_qubits)

def Y(index: int, n_qubits: int) -> 'csr_matrix':
    """Generate the Pauli Y operator acting on qubit ``index`` in a system of
    ``n_qubits``.

    Parameters
    ----------
    index : int
        index of the qubit to which the Y operator is applied
    n_qubits : int
        total number of qubits in the system

    Returns
    -------
    csr_matrix
        the Pauli Y operator acting on qubit ``index`` in a system of ``n_qubits``
    """
    return __pauli_term(y, index, n_qubits)

def Z(index: int, n_qubits: int):
    """Generate the Pauli Z operator acting on qubit ``index`` in a system of
    ``n_qubits``.

    Parameters
    ----------
    index : int
        index of the qubit to which the Z operator is applied
    n_qubits : int
        total number of qubits in the system

    Returns
    -------
    csr_matrix
        the Pauli Z operator acting on qubit ``index`` in a system of ``n_qubits``
    """
    return __pauli_term(z, index, n_qubits)
