import quop_mpi as qw
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

def x0(p,seed):
    return np.random.uniform(low = 0, high = 2*np.pi, size = 2*p)

for qubits in range(18,32):
    qwoa = qw.MPI.qwoa(qubits,comm)
    qwoa.set_initial_state(name="equal")
    qwoa.log_results("output/" + str(comm.size) + "_qwoa_benchmark","qwoa_equal",action="a")
    qwoa.plan()
    qwoa.benchmark(
            range(1,6),
            3,
            param_func = x0,
            qual_func = qw.qualities.random_floats,
            filename = "output/" + str(comm.size) + "_qwoa_benchmark",
            label = "qwoa_" + str(qubits))
    qwoa.destroy_plan()
