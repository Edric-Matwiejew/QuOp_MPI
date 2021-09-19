import numpy as np
from .kronecker import kron

zero = np.array([1, 0])
one = np.array([0,1])

def string(state):
    """Generate a quantum state-vector from a bit-string representation of the
    state. For example, '01' is parsed as :math:`|0\\rangle \otimes
    |1\\rangle`.

    :param state: A bit string state.
    :type state: string

    :return: The parsed quantu state.
    :rtype: array, complex
    """

    terms = []

    for digit in state:
        if str(digit) == '0':
            terms.append(zero)
        elif str(digit) == '1':
            terms.append(one)

    return np.array(kron(terms), dtype = np.complex128)


