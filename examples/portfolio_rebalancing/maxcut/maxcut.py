from quop_mpi.algorithms import qaoa
from quop_mpi.operators.diagonal import function
from quop_mpi.utils import *
import networkx as nx

Graph = nx.circular_ladder_graph(4)

nodes = len(Graph.nodes)
system_size = 2**nodes

G = nx.to_scipy_sparse_matrix(Graph)

def maxcut(G):
    C = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i,j] != 0:
                C += 0.5*(I(nodes) \
                - (Z(i,nodes) @ Z(j,nodes)))
    return -C.diagonal()

alg = qaoa(system_size)

alg.set_qualities(
        function,
        {'function':maxcut,
            'args':[G]})

alg.set_depth(5)
alg.execute()
alg.print_optimiser_result()
alg.save('maxcut', 'depth 5', 'w')

