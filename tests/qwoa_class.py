from mpi4py import MPI
from quop_mpi.algorithms import qwoa
from quop_mpi.operators import diagonal_uniform

COMM = MPI.COMM_WORLD

qubits = 4
system_size = 2**4

alg = qwoa(system_size, COMM)
alg.set_qualities(diagonal_uniform)
alg.set_depth(2)
alg.execute()
alg.print_optimiser_result()
