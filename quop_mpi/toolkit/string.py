from __future__ import annotations
import numpy as np
from .kronecker import kron

zero = np.array([1, 0])
one = np.array([0,1])

def string(state: str) -> np.ndarray[np.complex128]:
    """Generate an :term:`initial state` from a bit-string representation.

    Parameters
    ----------
    state : str
        a bit string state.

    Returns
    -------
    ndarray[complex128]
        the parsed quantum state
    """

    terms = []

    for digit in state:
        if str(digit) == '0':
            terms.append(zero)
        elif str(digit) == '1':
            terms.append(one)

    return np.array(kron(terms), dtype = np.complex128)


