from mpi4py import MPI
import networkx as nx
import numpy as np
import qwao_mpi as qw

comm = MPI.COMM_WORLD

p = 4
n_qubits = 3

np.random.seed(2)

def x0(p):
    return np.random.uniform(low = 0, high = 1, size = 2 * p)

qwao = qw.MPI.qwao(n_qubits, comm)
qwao.log_success("log", "qwao", action = "w")
qwao.graph(qw.graph_array.circle(qwao.size))
qwao.set_initial_state(name="split")
qwao.set_qualities(qw.qualities.random_floats)
qwao.plan()
qwao.execute(x0(p))
qwao.save("qwao", "example_config", action = "w")
qwao.benchmark(x0,[1,2,4,6],3,early_stopping=True)
qwao.destroy_plan()

if comm.Get_rank() == 0:
    print(qwao.result)

hyper_cube = nx.to_scipy_sparse_matrix(nx.hypercube_graph(n_qubits))

qaoa = qw.MPI.qaoa(hyper_cube,comm)
qaoa.log_success("log", "qaoa", action = "a")
qaoa.set_initial_state(name = "equal")
qaoa.set_qualities(qw.qualities.random_floats)
qaoa.execute(x0(p))
qaoa.save("qaoa", "example_config", action = "w")

qaoa.benchmark(x0,[1,2,4,6],3,early_stopping=True)

if comm.Get_rank() == 0:
    print(qaoa.result)

