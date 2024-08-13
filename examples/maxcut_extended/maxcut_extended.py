import numpy as np
import networkx as nx
from quop_mpi import Ansatz
from quop_mpi.propagator import diagonal, sparse
from quop_mpi.observable import serial
from quop_mpi.toolkit import Z

Graph = nx.circular_ladder_graph(4)
vertices = len(Graph.nodes)
system_size = 2 ** vertices
G = nx.to_scipy_sparse_array(Graph)
n_edges = 2 * Graph.number_of_edges()

def maxcut_terms(G):
    vertices = G.shape[0]
    terms = []
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] != 0:
                term = Z(i, vertices) @ Z(j, vertices)
                terms.append(-0.5 * (1 - term.diagonal()))
    return terms

def maxcut_qualities(G):
    return np.sum(maxcut_terms(G), axis = 0)

UQ = diagonal.unitary(
    diagonal.operator.serial,
    operator_dict={"args": [maxcut_terms, G]},
    unitary_n_params=n_edges
)

UW = sparse.unitary(sparse.operator.hypercube)

alg = Ansatz(system_size)
alg.set_unitaries([UQ, UW])
alg.set_observables(serial, {"args": [maxcut_qualities, G]})
alg.set_depth(2)
alg.execute()
alg.print_result()
alg.save("maxcut_extended", "depth 2", "w")