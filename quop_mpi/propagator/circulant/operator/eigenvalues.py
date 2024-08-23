from __future__ import annotations
#from importlib import import_module
import numpy as np

#TODO update docstring new complete 
def complete(system_size: int) -> np.ndarray[np.float64]:
    """Generate a parallel partition of the eigenvalues of a complete
    circulant graph with edge weightings :literal:`1`.

    An :term:`Operator Function` associated with
    :class:`quop_mpi.propagator.circulant.unitary`.

    Parameters
    ----------
    system_size : int
        the size of the simulated :term:`QVA`
    local_i : int
        size of the local :term:`system state` partitions,
        :class:`quop_mpi.Ansatz` attribute
    local_i_offset : int
        global index offset of the local system state partitions,
        :class:`quop_mpi.Ansatz` attribute

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` eigenvalues with
        global index offset :literal:`local_i_offset`
    """
    return np.empty(1, dtype = np.float64)

    #return fqwoa_mpi.fqwoa_mpi.graph_eigenvalues(graph_array, local_i, local_i_offset)

#TODO update docstring, document case for complete graph
#TODO UPDATE DOCSTRING
def graph(
        system_size: int,
        i: int = 1) -> np.ndarray[np.float64]:
    """Generate the eigenvalues of the i-th symmetric circulant graph with
    edge weightings :literal:`1`.

    An :term:`Operator Function` associated with
    :class:`quop_mpi.propagator.circulant.unitary`.

    Parameters
    ----------
    system_size : int
        the size of the simulated :term:`QVA`
    local_i : int
        size of the local :term:`system state` partitions,
        :class:`quop_mpi.Ansatz` attribute
    local_i_offset : int
        global index offset of the local system state partitions,
        :class:`quop_mpi.Ansatz` attribute
    i : int, optional
        index of the graph (ordered by vertex degree), :literal:`1` corresponds to a
        cycle graph and :literal:`system_size // 2 + 1` to a complete graph, by default
        :literal:`1`

    Returns
    -------
    ndarray[complex128]
        1-D complex array of :literal:`local_i` eigenvalues with global index offset
        :literal:`local_i_offset`
    """

    if (i >= system_size//2):
        graph_array = np.empty(1, dtype = np.float64)
    else:

        graph_array = np.zeros(system_size, dtype = np.float64)

        for j in range(1,i+1):
            graph_array[j] = 1
            graph_array[system_size - j]=1

    return graph_array
