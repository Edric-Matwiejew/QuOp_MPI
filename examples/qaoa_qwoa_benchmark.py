import quop_mpi as qw
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
            range(1,3),
            2,
            param_func = x0,
            qual_func = qw.qualities.random_floats,
            filename = "qaoa_equal",
            label = "qaoa_" + str(qubits))

for qubits in range(2,4):
    qwoa = qw.MPI.qwoa(qubits,comm)
    qwoa.set_initial_state(name="equal")
    qwoa.log_success("qwoa_complete_equal","equal",action="a")
    qwoa.set_graph(qw.graph_array.complete(qwoa.size))
    qwoa.plan()
    qwoa.benchmark(
            range(1,3),
            2,
            param_func = x0,
            qual_func = qw.qualities.random_floats,
            filename = "qwoa_complete_equal",
            label = "qwoa_" + str(qubits))
    qwoa.destroy_plan()
