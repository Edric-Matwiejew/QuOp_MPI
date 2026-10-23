import numpy as np

from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian
from quop_mpi.parameter_maps import linear_ramp


def rastrigin(x):
    return (len(x[0]) * 10 + (x**2 - 10 * np.cos(2 * np.pi * x))).sum(axis=1)


dimension = 4
bounds = dimension * [[-5.12, 5.12]]  # bounds of the discretised solution space
Ns = dimension * [4]  # number of qubits per grid dimension

deltas, mins = setup_cartesian(Ns, bounds)
alg = QMOA(Ns)
alg.verbose_objective = True
alg.set_qualities(cartesian, {"args": [deltas, mins, rastrigin]})
alg.set_log("multivariable", "rastrigin", "w")
alg.set_depth(5)
alg.set_parameter_map(linear_ramp())
alg.set_optimiser(
    "scipy",
    {"method": "L-BFGS-B", "bounds": [(None, 0.0), (0.0, None), (0.0, 1.0)]},
)
alg.execute([-1.0, 0.1, 0.1])  # gamma, t, beta
alg.print_result()
