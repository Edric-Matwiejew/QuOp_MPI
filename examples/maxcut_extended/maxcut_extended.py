from quop_mpi import Ansatz
from quop_mpi.propagator import diagonal, sparse
from quop_mpi.observable import serial
from quop_mpi.param.rand import uniform
from quop_mpi.toolkit import Z
import numpy as np
import networkx as nx

Graph = nx.circular_ladder_graph(4)
vertices = len(Graph.nodes)
system_size = 2 ** vertices

G = nx.to_scipy_sparse_matrix(Graph)

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


def maxcut_qualities(terms):
    return np.sum(terms, axis=0)


computed_terms = maxcut_terms(G)

UQ = diagonal.unitary(
    diagonal.operator.serial,
    operator_kwargs={"function": maxcut_terms, "args": [G]},
    unitary_n_params=n_edges,
    parameter_function=uniform,
)

UW = sparse.unitary(sparse.operator.hypercube, parameter_function=uniform)

alg = Ansatz(system_size)

alg.set_unitaries([UQ, UW])

alg.set_observables(serial, {"function": maxcut_qualities, "args": [computed_terms]})

alg.execute()
alg.print_optimiser_result()
alg.save("maxcut_extended", "depth 2", "w")
