import numpy as np

def complete(N):
    """
    Returns an array corresponding to a complete graph of size N.

    :param N: Number of graph nodes.
    :type N: integer
    """
    graph_array = np.ones(N, dtype = np.float64)
    graph_array[0] = 0
    return graph_array
