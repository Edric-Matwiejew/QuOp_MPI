from __future__ import annotations
from importlib import import_module
import numpy as np


def magnitude_squared(
    Ns: list[int], minsk: list[float], deltask: list[float]
) -> np.ndarray[np.complex128]:
    """Generate the :ref:`QMOA` :term:`mixing unitary` :term:`operator`.

    An :term:`Operator Function` for
    :class:`quop_mpi.propagator.momentum.unitary`.

    Parameters
    ----------
    Ns : list[int]
        the number of grid points in each dimension of the Cartesian grid in
        position and momentum space,
        :class:`quop_mpi.propagator.momentum.unitary` attribute
    minsk : list[float]
        the minimum of each Cartesian coordinate in momentum space,
        :class:`quop_mpi.propagator.momentum.unitary` attribute
    deltask : list[float]
        the step-size in each Cartesian coordinate in momentum space,
        :class:`quop_mpi.propagator.momentum.unitary` attribute
     
    Returns
    -------
    np.ndarray[np.complex128]
        a 1-D complex array of ``local_i`` elements of the QOWE diagonal
        momentum-space operator with global index offset ``local_i_offset``
    """
    grid = np.empty(max(Ns), dtype=np.float64)
    momentums = np.zeros((max(Ns), len(Ns)), dtype=np.complex128)

    for i, (N, mink, deltak) in enumerate(zip(Ns, minsk, deltask)):

        grid[0] = mink

        for j in range(1, N):
            grid[i] = (mink + j * deltak) ** 2

        momentums[:N, i] = grid[:N]

    return momentums
