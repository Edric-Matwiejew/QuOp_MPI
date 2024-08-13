from __future__ import annotations
import numpy as np

def uniform(
        n_params: int,
        seed: int,
        low: float = 0,
        high: float = 2*np.pi) -> np.ndarray[np.float64]:
    """Generate initial :term:`variational parameters` from a uniform
    distribution.

    The default :term:`Parameter Function` of the :class:`quop_mpi.Unitary`
    class. User specified ``low`` and ``high`` values can be specified by
    passing a corresponding:term:`FunctionDict` to on initialisation of a
    ``unitary`` instance (see :meth:`quop_mpi.Unitary`).

    Parameters
    ----------
    n_params : int
        total number of :term:`unitary <unitary parameter>` and
        :term:`operator<operator parameter>` :term:`variational parameters`,
        :class:`quop_mpi.Unitary` attribute
    seed : int
        seeds random number generation, :class:`quop_mpi.Unitary` attribute
    low : float, optional
        lower bound of the generated variational parameters (inclusive), by
        default ``0``
    high : float, optional
        upper bound of the generated variational parameters (exclusive), by
        default ``2*pi``

    Returns
    -------
    ndarray[float64]
        a 1-D array of ``n_params`` variational parameters
    """

    np.random.seed(seed)

    return np.random.uniform(low = low, high = high, size = n_params)
