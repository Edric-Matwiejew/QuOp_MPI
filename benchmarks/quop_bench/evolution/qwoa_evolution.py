import copy
from quop_mpi.algorithm import qwoa
from quop_mpi.observable.rand import uniform


def function(system_size, COMM):
    alg = qwoa(system_size, COMM)
    alg.set_qualities(uniform)
    alg.set_depth(15)
    params = alg.gen_initial_params()
    alg.evolve_state(params)
    return alg.local_i, copy.deepcopy(alg.final_state)
