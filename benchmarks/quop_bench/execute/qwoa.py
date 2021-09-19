from quop_mpi.algorithm import qwoa
from quop_mpi.observable.rand import uniform


def function(system_size, depth, log_path, COMM):
    alg = qwoa(system_size, COMM)
    alg.set_log(log_path, "qwoa")
    alg.set_qualities(uniform)
    alg.set_depth(depth)
    alg.execute()
