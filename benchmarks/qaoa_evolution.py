from quop_mpi.algorithms import qaoa
from quop_mpi.operators import diagonal_uniform

def function(system_size, COMM):
    alg = qaoa(system_size, COMM)
    alg.set_qualities(diagonal_uniform)
    alg.set_depth(15)
    params = alg.get_initial_params()
    alg.pre()
    alg.evolve_state(params)
    alg.post()
    return alg.final_state
