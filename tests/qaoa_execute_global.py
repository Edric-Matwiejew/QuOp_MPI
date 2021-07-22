from quop_mpi.algorithms import ansatz
from mpi4py import MPI
from quop_mpi.unitaries import diagonal, sparse
from quop_mpi.states import equal
from quop_mpi.operators import diagonal_uniform, sparse_hypercube
from quop_mpi.params import uniform

COMM = MPI.COMM_WORLD

qubits = 4
system_size = 2**qubits

U1 = diagonal(diagonal_uniform, parameter_function = uniform)
U2 = sparse(sparse_hypercube, parameter_function = uniform)

qaoa = ansatz(system_size, COMM, parallel = "jacobian")
qaoa.set_unitaries([U1, U2], 0)
qaoa.set_depth(5)
qaoa.execute()
qaoa.print_optimiser_result()
