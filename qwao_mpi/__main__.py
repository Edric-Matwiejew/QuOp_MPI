from mpi4py import MPI
import numpy as np
import qwao_mpi as qw

comm = MPI.COMM_WORLD

p = 2
n_qubits = 3

# random beta and gamma start anglels
np.random.seed(1)
x0 = np.random.rand(2*p)

qwao = qw.MPI.qwao(n_qubits, comm)
qwao.graph(qw.graph_array.complete(qwao.size))
qwao.set_initial_state(name="equal")
qwao.set_qualities(qw.qualities.ordered_integers)
qwao.plan()
qwao.execute(x0)
qwao.save("example", "example_config", action = "w")
qwao.destroy_plan()

if comm.Get_rank() == 0:
    print(qwao.result)
