from mpi4py import MPI
import numpy as np
import quop_mpi as qu

comm = MPI.COMM_WORLD

p = 3
n_qubits = 5

rng = np.random.RandomState(1)

def x0(p):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)

qwoa = qu.MPI.qwoa(n_qubits, comm, parallel = "jacobian")
qwoa.set_log("log", "qwoa", action = "w")
qwoa.set_initial_state(name="equal")
qwoa.set_observables(qu.qualities.random_floats)
qwoa.execute(x0(p))
qwoa.save("qwoa", "example_config", action = "w")
qwoa.print_optimiser_result()

qaoa = qu.MPI.qaoa(n_qubits,comm, parallel = "jacobian")
qaoa.set_log("log", "qaoa", action = "a")
qaoa.set_initial_state(name = "equal")
qaoa.set_observables(qu.qualities.random_floats)
qaoa.execute(x0(p))
qaoa.save("qaoa", "example_config", action = "w")
qaoa.print_optimiser_result()
