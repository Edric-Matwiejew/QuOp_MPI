from quop_mpi import ansatz
from quop_mpi.unitaries import diagonal, sparse
from quop_mpi.states import equal
from quop_mpi.operators import diagonal_uniform, sparse_hypercube
from quop_mpi.params import uniform

qubits = 4
system_size = 2**qubits

U1 = diagonal(diagonal_uniform, parameter_function = uniform)
U2 = sparse(sparse_hypercube, parameter_function = uniform)

qaoa = ansatz(system_size, parallel = "jacobian")
qaoa.set_unitaries([U1, U2], 0)
qaoa.set_depth(5)
qaoa.execute()
qaoa.print_optimiser_result()
