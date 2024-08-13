from quop_mpi.algorithm.combinatorial import qaoa, serial
from quop_mpi.toolkit import I, Z
import networkx as nx

Graph = nx.circular_ladder_graph(4)

vertices = len(Graph.nodes)
system_size = 2 ** vertices

G = nx.to_scipy_sparse_array(Graph)

def maxcut_qualities(G):
    C = 0
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] != 0:
                C += 0.5 * (I(vertices) - (Z(i, vertices) @ Z(j, vertices)))
    return -C.diagonal()


alg = qaoa(system_size)
alg.set_qualities(serial, {'args':[maxcut_qualities, G]})
alg.set_depth(2)

alg.execute()

alg.print_result()
alg.save("maxcut", "depth 2", "w")
