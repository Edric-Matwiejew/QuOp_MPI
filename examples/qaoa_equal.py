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
    qaoa.log_success("qaoa_equal","equal",action="a")
    qaoa.set_qualities(qw.qualities.random_floats)
    qaoa.benchmark(range(1,6), 5, param_func = x0, qual_func = qw.qualities.random_floats, filename = "qaoa_equal",label = "qaoa_" + str(qubits))
