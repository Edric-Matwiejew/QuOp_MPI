import copy
from quop_mpi.algorithms import qwoa
from quop_mpi.operators.diagonal import random
import numpy as np

def function(system_size, COMM):
    try:
        alg = qwoa(system_size, COMM)
        alg.set_qualities(random)
        alg.set_depth(15)
        params = alg.get_initial_params()
        alg.evolve_state(params)
        return alg.local_i, copy.deepcopy(alg.final_state)
    finally:
        alg.post()
