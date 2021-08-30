from importlib import import_module
import numpy as np

def complete(
        system_size,
        local_o,
        local_o_offset):

    fqwoa_mpi = import_module('quop_mpi.__lib.fqwoa_mpi')

    graph_array = np.ones(system_size, dtype = np.float64)
    graph_array[0] = 0

    return fqwoa_mpi.graph_eigenvalues(graph_array, local_o, local_o_offset)


def graph(
        system_size,
        local_o,
        local_o_offset,
        i = None):
    '''
    Returns an array c1=c2=...=ci=1, others equals to '0'
    array size: N
    ci(N, 1) will return circle graph
    ci(N, int(N/2)+1) will return complete graph
    '''

    fqwoa_mpi = import_module('quop_mpi.__lib.fqwoa_mpi')

    if i is None:
        i = system_size

    if (i > system_size//2):

        graph_array = np.ones(system_size, dtype = np.float64)
        graph_array[0] = 0

    else:

        graph_array = np.zeros(system_size, dtype = np.float64)

        for j in range(1,i+1):
            graph_array[j] = 1
            graph_array[system_size - j]=1

    eigenvalues = fqwoa_mpi.graph_eigenvalues(
            graph_array,
            local_o,
            local_o_offset)

    return eigenvalues
