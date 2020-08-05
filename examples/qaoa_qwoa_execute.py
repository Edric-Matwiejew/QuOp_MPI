from mpi4py import MPI
import numpy as np
import quop_mpi as qu

comm = MPI.COMM_WORLD

p = 3
n_qubits = 7

np.random.seed(2)

def x0(p):
    return np.random.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_qubits, comm)
qwoa.log_results("log", "qwoa", action = "w")
qwoa.set_initial_state(name="equal")
qwoa.set_qualities(qu.qualities.random_floats)
qwoa.plan()
qwoa.execute(x0(p))
qwoa.save("qwoa", "example_config", action = "w")
qwoa.destroy_plan()
qwoa.print_result()

qaoa = qu.MPI.qaoa(n_qubits,comm)
qaoa.log_results("log", "qaoa", action = "a")
qaoa.set_initial_state(name = "equal")
qaoa.set_qualities(qu.qualities.random_floats)
qaoa.execute(x0(p))
qaoa.save("qaoa", "example_config", action = "w")
qaoa.print_result()
