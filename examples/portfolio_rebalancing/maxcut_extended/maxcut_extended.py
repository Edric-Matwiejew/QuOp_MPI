from quop_mpi import ansatz
from quop_mpi.unitaries import diagonal
from quop_mpi.unitaries import sparse
from quop_mpi.operators.diagonal import function
from quop_mpi.operators.sparse import hypercube
from quop_mpi.params import uniform
from quop_mpi.utils import *
import numpy as np
import networkx as nx

Graph = nx.circular_ladder_graph(4)

nodes = len(Graph.nodes)
system_size = 2**nodes

G = nx.to_scipy_sparse_matrix(Graph)

n_edges = 2*Graph.number_of_edges()

def maxcut_terms(G):
    nodes = G.shape[0]
    terms = []
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i,j] != 0:
                term = Z(i,nodes) @ Z(j,nodes)
                terms.append(0.5*(1 - term.diagonal()))
    return terms

def maxcut_qualities(terms):
    return -np.sum(terms, axis = 0)

computed_terms = maxcut_terms(G)

UQ = diagonal(
        function,
        operator_kwargs = {
            'function': maxcut_terms,
            'args':[G]},
        unitary_n_params = n_edges,
        parameter_function = uniform)

UW = sparse(
        hypercube,
        parameter_function = uniform)

alg = ansatz(system_size)

alg.set_unitaries([UQ, UW])

alg.set_observables(
        function,
        {'function':maxcut_qualities,
            'args':[computed_terms]})

alg.set_depth(1)
alg.execute()
alg.print_optimiser_result()
alg.save('maxcut_extended', 'depth 2', 'w')
