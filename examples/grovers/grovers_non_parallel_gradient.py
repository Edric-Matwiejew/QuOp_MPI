from quop_mpi.algorithm.combinatorial import qwoa
import pandas as pd
import numpy as np


def free_param_function(total_params, ansatz_depth):
    return list(range((ansatz_depth - 1) * total_params, ansatz_depth * total_params))


def marked_state(local_i, MPI_COMM):
    q = np.zeros(local_i, dtype=np.float64)
    if MPI_COMM.Get_rank() == 0:
        q[0] = -1
    return q


system_size = 2**6

alg = qwoa(system_size)
alg.set_qualities(marked_state)
alg.set_log("grovers_non_parallel_gradient", "qwoa", action="w")
alg.set_free_params(free_param_function)

alg.benchmark(
    range(1, 7),
    1,
    param_persist=True,
    filename="grovers_non_parallel_gradient",
    save_action="w",
)
