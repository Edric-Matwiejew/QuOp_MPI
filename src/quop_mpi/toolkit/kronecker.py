"""Kronecker (tensor) product utilities for sparse matrices."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse as __sparse

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix, spmatrix


def kron(terms: list[spmatrix]) -> csr_matrix:
    """Compute the tensor (Kronecker) product of a sequence of sparse matrices.

    Parameters
    ----------
    terms : list[spmatrix]
        a list of scipy sparse matrices

    Returns
    -------
    csr_matrix
        the tensor product of ``terms``, computed from left to right
    """
    if not terms:
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
            out = __sparse.kron(out, term, format="coo")

        return out.tocsr()


def kron_power(term: spmatrix, n: int) -> csr_matrix:
    """Compute the tensor (Kronecker) product of ``n`` instances of a sparse matrix.

    Parameters
    ----------
    term : spmatrix
        a scipy sparse matrix
    n : int
        length of the tensor product sequence

    Returns
    -------
    csr_matrix
       tensor product of ``n`` occurences of ``term``
    """
    return kron([term for _ in range(n)])
