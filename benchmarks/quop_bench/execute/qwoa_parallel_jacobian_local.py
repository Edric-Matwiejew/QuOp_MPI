from quop_mpi.algorithms import qwoa
from quop_mpi.operators import diagonal_uniform

def function(system_size, depth, log_path, COMM):
    try:
        alg = qwoa(system_size, COMM, parallel = 'jacobian_local')
        alg.set_log(log_path, 'qwoa parallel jacobian (local)')
        alg.set_qualities(diagonal_uniform)
        alg.set_depth(depth)
        alg.execute()
    finally:
        alg.post()

