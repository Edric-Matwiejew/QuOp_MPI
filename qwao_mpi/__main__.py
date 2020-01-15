from mpi4py import MPI
import numpy as np
from MPI import *
from qualities import *
from graph_array import *

comm = MPI.COMM_WORLD

p = 2
n_qubits = 3

# random beta and gamma start anglels
np.random.seed(1)
x0 = np.random.rand(2*p)

qwao = qwao(n_qubits, comm)
qwao.graph(complete(qwao.size))
qwao.qualities(integer)
qwao.plan()

result = qwao.execute(x0)

qwao.save("example", "example_config", action = "w")
qwao.destroy_plan()

if comm.Get_rank() == 0:
    print(result)

