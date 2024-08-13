from mpi4py import MPI
from quop_mpi.algorithm.combinatorial import qaoa, serial
from quop_mpi.toolkit import I, Z
from quop_mpi.meta import swarm
import networkx as nx
import numpy as np

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


s = swarm(1, 4, 2, MPI.COMM_WORLD, qaoa, system_size)
s.set_qualities(serial, {"args": [maxcut_qualities, G]})
s.set_seed([[i] for i in range(s.subcomms.get_n_subcomms())])
np.random.seed(0)
tasks = []
for i in range(16):
    tasks.append(np.random.uniform(size = 2))

results = s.execute_swarm(tasks, basename = 'execute_swarm/maxcut')

result = s.get_optimal_result()
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f'fun: {result["fun"]}')
    print(f'variational parameters: {result["x"]}')
