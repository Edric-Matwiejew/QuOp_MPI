import quop_mpi as qw
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

rng = np.random.RandomState(1)

def x0(p,seed):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

for qubits in range(18,32):
    qaoa = qw.MPI.qaoa(qubits,comm)
    qaoa.set_initial_state(name = "equal")
    qaoa.log_results("output/" + str(comm.size) + "_qaoa_benchmark","qaoa_equal",action="a")
    qaoa.set_qualities(qw.qualities.random_floats)
    qaoa.benchmark(
            range(1,6),
            3,
            param_func = x0,
            qual_func = qw.qualities.random_floats,
            filename = "output/" +  str(comm.size) + "_qaoa_benchmark",
            label = "qaoa_" + str(qubits))
