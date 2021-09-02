from quop_mpi.algorithms import qwoa
from quop_mpi.operators.diagonal import random

def function(system_size, depth, log_path, COMM):
    alg = qwoa(system_size, COMM)
    alg.set_parallel('jacobian')
    alg.set_log(log_path, 'qwoa parallel jacobian')
    alg.set_qualities(random)
    alg.set_depth(depth)
    alg.execute()
