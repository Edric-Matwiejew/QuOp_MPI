import sys
sys.path.append('../../')
import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qaoa
import numpy as np
import networkx as nx

graph = nx.circular_ladder_graph(4)
qubits = graph.number_of_nodes()
system_size = 2 ** qubits

def parallel_maxcut_qualities(local_i, local_i_offset, graph=None, qubits=None):
    
    qualities = np.full(local_i, graph.number_of_edges(), dtype = np.float64)

    for i in range(local_i_offset, local_i_offset + local_i):
        bit_string = np.binary_repr(i, width=qubits)
        for edge in graph.edges:
            if bit_string[edge[0]] != bit_string[edge[1]]:
                qualities[i - local_i_offset] -= 1
    print('parallel',qualities.tolist())
    return qualities

alg = qaoa(system_size)

alg.set_qualities(parallel_maxcut_qualities, {"graph": graph, "qubits": qubits})

alg.verbose_objective = True
alg.set_depth(2)
alg.execute()
alg.print_optimiser_result()
alg.save("maxcut_parallel_qualities", "depth 2", "w")
