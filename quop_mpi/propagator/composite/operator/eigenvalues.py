"""Predefined :term:`Operator Functions` for :class:`quop_mpi.propagator.composite.unitary`."""
from __future__ import annotations
from importlib import import_module
import numpy as np

def ith(Ns: list[int], Cs: list[int] = None) -> np.ndarray[np.float64]:
    #TODO: NOW RETURNS GRAPH ARRAYS
    """Generate the eigenvalues of a :ref:`QMOA` :term:`mixing unitary` 
    :term:`operator`.

    An :term:`Operator Function` for
    :class:`quop_mpi.propagator.composite.unitary`. The :literal:`Cs` keyword argument
    may be defined via a corresponding :term:`FunctionDict` on initialisation of
    the receiving :literal:`unitary` instance.

    See Also
    --------
    :meth:`quop_mpi.propagate.circulant.operator.graph`
        Generate the eigenvalues of the i-th symmetric circulant graph with edge weights :literal:`1`.

    Parameters
    ----------
    Ns : list[int]
        the number of grid points in each dimension of the Cartesian grid,
        :class:`quop_mpi.propagator.composite.unitary` attribute
    Cs : list[int], optional
        specifies the i-th index of the circulant operators associated with
        each dimension of the Cartesian grid, complete graphs by default 

    Returns
    -------
    ndarray[complex128]
        a 2-D complex array containing :literal:`local_i` eigenvalues of the QMOA
        mixing unitary with global index offset :literal:`local_i_offset`
    """

    circulant_eigenvalues = import_module('quop_mpi.propagator.circulant.operator.eigenvalues')

    complete = [C >= N // 2 for C, N in zip(Cs, Ns)]

    if all(complete):
        return np.empty((len(Ns), 1), dtype = np.float64)

    eigenvalues = np.zeros((np.max(Ns), len(Ns)), dtype = np.float64)

    for i, (N, C)  in enumerate(zip(Ns, Cs)):
        eigenvalues[:N,i] = circulant_eigenvalues.graph(N, N, 0, C)
    return np.asfortranarray(eigenvalues)

 
