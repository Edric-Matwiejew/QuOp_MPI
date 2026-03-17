"""Eigenvalue functions for circulant graph operators."""

from __future__ import annotations

import numpy as np


def complete(system_size: int) -> np.ndarray[np.float64]:
    """Return a placeholder array for a complete circulant graph.

    For a complete graph, eigenvalue computation is handled internally by the
    propagator, so this function returns an empty array as a signal.

    An :term:`Operator Function` associated with
    :class:`quop_mpi.propagator.circulant.unitary`.

    Parameters
    ----------
    system_size : int
        the size of the simulated :term:`QVA`

    Returns
    -------
    ndarray[float64]
        1-D array with a single element (placeholder for complete graph)
    """
    return np.empty(1, dtype=np.float64)


def graph(system_size: int, i: int = 1) -> np.ndarray[np.float64]:
    """Generate the eigenvalues of the i-th symmetric circulant graph with
    edge weightings :literal:`1`.

    An :term:`Operator Function` associated with
    :class:`quop_mpi.propagator.circulant.unitary`.

    Parameters
    ----------
    system_size : int
        the size of the simulated :term:`QVA`
    i : int, optional
        index of the graph (ordered by vertex degree), :literal:`1` corresponds to a
        cycle graph and :literal:`system_size // 2` or greater to a complete graph,
        by default :literal:`1`

    Returns
    -------
    ndarray[float64]
        1-D real array of :literal:`system_size` eigenvalues, or a single-element
        placeholder array if :literal:`i >= system_size // 2` (complete graph)
    """

    if i >= system_size // 2:
        graph_array = np.empty(1, dtype=np.float64)
    else:

        graph_array = np.zeros(system_size, dtype=np.float64)

        for j in range(1, i + 1):
            graph_array[j] = 1
            graph_array[system_size - j] = 1

    return graph_array
