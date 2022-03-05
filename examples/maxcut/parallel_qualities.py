import sys
sys.path.append('../../')
import h5py
import mpi4py.MPI
from quop_mpi.algorithm import qaoa
import numpy as np
import networkx as nx

def parallel_maxcut_qualities(local_i, local_i_offset, graph=None):

    n_qubits = graph.number_of_nodes() 
    qualities = np.zeros(local_i, dtype = np.float64)

    start = local_i_offset
    finish = local_i_offset + local_i

    for i in range(start, finish):
        bit_string = np.binary_repr(i, width=n_qubits)
        for edge in Graph.edges:
            if bit_string[edge[0]] == bit_string[edge[1]]:
                qualities[i - local_i_offset] += 1 

    return qualities

n_qubits = int(sys.argv[1])
seed = int(sys.argv[2])

np.random.seed(seed)

Graph = nx.random_regular_graph(n=n_qubits, d=3,seed=0)

alg = qaoa(2**n_qubits)
alg.verbose_objective = True
alg.set_qualities(parallel_maxcut_qualities, {"graph": Graph})
alg.set_depth(2)
alg.execute()
#params = alg.gen_initial_params()
#alg.evolve_state(params)
#expectation = alg.get_expectation_value()
#print(expectation)
alg.print_optimiser_result()
#alg.save("maxcut_parallel_qualities", "depth 5", "w")
