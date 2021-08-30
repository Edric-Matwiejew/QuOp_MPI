import copy
from quop_mpi.algorithms import qaoa
from quop_mpi.operators.diagonal import random

def function(system_size, COMM):
    try:
        alg = qaoa(system_size, COMM)
        alg.set_qualities(random)
        alg.set_depth(15)
        params = alg.get_initial_params()
        alg.evolve_state(params)
        return alg.local_i, copy.deepcopy(alg.final_state)
    finally:
        alg.post()
