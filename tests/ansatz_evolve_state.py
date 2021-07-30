import numpy as np
from quop_mpi import ansatz
from quop_mpi.unitaries import diagonal, sparse
from quop_mpi.states import equal
from quop_mpi.operators import diagonal_uniform, sparse_hypercube
from quop_mpi.params import uniform

qubits = 2
system_size = 2**qubits

U1 = diagonal(diagonal_uniform, parameter_function = uniform)
U2 = sparse(sparse_hypercube, parameter_function = uniform)

qaoa = ansatz(system_size)
qaoa.set_unitaries([U1, U2])
qaoa.set_observables(0)
qaoa.set_depth(2)
x = qaoa.get_initial_params()
qaoa.evolve_state(x)
if qaoa.final_state is not None:
    print(np.abs(qaoa.final_state)**2, qaoa.COMM_OPT.Get_rank(), flush = True)
qaoa.post()
