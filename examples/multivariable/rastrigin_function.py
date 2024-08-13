import numpy as np
from quop_mpi.algorithm.multivariable import qmoa, cartesian, setup_cartesian

def rastrigin(x):
    return (len(x[0]) * 10 + (x**2 - 10 * np.cos(2 * np.pi * x))).sum(axis=1)

dimension = 4
bounds = dimension * [[-5.12, 5.12]] # bounds of the discretised solution space
Ns = dimension * [4] # number of qubits per grid dimension 

deltas, mins = setup_cartesian(Ns, bounds)
alg = qmoa(Ns)
alg.set_qualities(cartesian, {"args":[deltas, mins, rastrigin]})
alg.set_log("multivariable", "rastrigin", "w")
alg.set_depth(5)
alg.execute()
alg.print_result()
