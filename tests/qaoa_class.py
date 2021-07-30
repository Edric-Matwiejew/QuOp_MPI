import numpy as np
from quop_mpi.algorithms import qaoa
from quop_mpi.operators import diagonal_uniform

qubits = 2
system_size = 2**qubits

alg = qaoa(system_size)
alg.set_qualities(diagonal_uniform)
alg.set_depth(2)
alg.execute()
alg.print_optimiser_result()
probs = alg.get_probabilities()
if alg.COMM.Get_rank() == 0:
    print(probs)
