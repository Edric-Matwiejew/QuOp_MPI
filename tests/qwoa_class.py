from quop_mpi.algorithms import qwoa
from quop_mpi.operators import diagonal_uniform

qubits = 4
system_size = 2**qubits

alg = qwoa(system_size)
alg.set_qualities(diagonal_uniform)
alg.set_depth(2)
alg.execute()
alg.print_optimiser_result()
