import sys
sys.path.append('../../')
import mpi4py.MPI
import h5py
from quop_mpi.algorithm import qaoa
from quop_mpi import observable
from quop_mpi.toolkit import I, Z
import networkx as nx

graph = nx.circular_ladder_graph(4)

qubits = graph.number_of_nodes()
system_size = 2 ** qubits

def maxcut_qualities(graph, qubits):
    C = 0
    for edge in graph.edges():
        C += 0.5*(I(qubits) + (Z(edge[0], qubits) @ Z(edge[1], qubits)))
    print('maxcut', C.diagonal())
    return C.diagonal()

alg = qaoa(system_size)

alg.set_qualities(observable.serial, {"function": maxcut_qualities, "args": [graph, qubits]})

alg.verbose_objective = True
alg.set_depth(2)
alg.execute()
alg.print_optimiser_result()
alg.save("maxcut", "depth 2", "w")
