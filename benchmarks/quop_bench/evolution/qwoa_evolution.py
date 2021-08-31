import copy
from quop_mpi.algorithms import qwoa
from quop_mpi.operators.diagonal import random

def function(system_size, COMM):
    alg = qwoa(system_size, COMM)
    alg.set_qualities(random)
    alg.set_depth(15)
    params = alg.gen_initial_params()
    alg.evolve_state(params)
    return alg.local_i, copy.deepcopy(alg.final_state)
