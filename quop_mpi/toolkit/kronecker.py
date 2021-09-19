from copy import copy
import numpy as np
from scipy import sparse as __sparse

def kron(terms):
    """Calculate :math:`A \otimes B \otimes C \otimes...`.

    :param terms: A list of square sparse matricies.
    :type terms: list, Scipy sparse matrix

    :return: The kronecker product of the input matricies ordered from left to right.
    :rtype: Scipy sparse matrix
    """

    if len(terms) == 0:
        return 1

    if len(terms) == 1:
        return terms[0]

    out = copy(terms[0])

    if isinstance(terms[0], np.ndarray):

        for term in terms[1:]:
            out = np.kron(out, term)

        return out

    else:

        for term in terms[1:]:
            out = __sparse.kron(out, term, format = 'coo')

        return out.tocsr()

def kron_power(term, n):
    """Calculate :math:`A^{\otimes n}`.

    :param term: :math:`A`.
    :type term: Scipy sparse matrix

    :param n: :math:`n`.
    :type n: integer

    :return: :math:`A^{\otimes n}`.
    :rtype: Scipy sparse matrix
    """
    return kron([term for _ in range(n)])

