from quop_mpi.algorithms import qwoa
from quop_mpi.operators import diagonal_uniform

def function(system_size, depth, log_path, COMM):
    try:
        alg = qwoa(system_size, COMM)
        alg.set_log(log_path, 'qwoa')
        alg.set_qualities(diagonal_uniform)
        alg.set_depth(depth)
        alg.execute()
    finally:
        alg.post()

