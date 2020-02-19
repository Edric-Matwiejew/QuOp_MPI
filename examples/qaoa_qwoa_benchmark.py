import qwao_mpi as qw
import numpy as np
import networkx as nx
from mpi4py import MPI

comm = MPI.COMM_WORLD


def x0(p,seed):
    return np.random.uniform(low = 0, high = 1, size = 2*p)

for qubits in range(2,4):
    hypercube = nx.to_scipy_sparse_matrix(nx.hypercube_graph(qubits))
    qaoa = qw.MPI.qaoa(hypercube,comm)
    qaoa.set_initial_state(name = "equal")
    qaoa.log_success("benchmark_example","qaoa_equal",action="a")
    qaoa.set_qualities(qw.qualities.random_floats)
    qaoa.benchmark(
            range(1,6),
            5,
            param_func = x0,
            qual_func = qw.qualities.random_floats,
            filename = "qaoa_equal",
            label = "qaoa_" + str(qubits))

for qubits in range(2,4):
    qwao = qw.MPI.qwao(qubits,comm)
    qwao.set_initial_state(name="equal")
    qwao.log_success("qwao_complete_equal","equal",action="a")
    qwao.set_graph(qw.graph_array.complete(qwao.size))
    qwao.plan()
    qwao.benchmark(
            range(1,6),
            5,
            param_func = x0,
            qual_func = qw.qualities.random_floats,
            filename = "qwao_complete_equal",
            label = "qwao_" + str(qubits))
    qwao.destroy_plan()
