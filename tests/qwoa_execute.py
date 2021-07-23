from quop_mpi import ansatz
from quop_mpi.unitaries import diagonal, circulant
from quop_mpi.states import equal
from quop_mpi.operators import diagonal_uniform, circulant_complete
from quop_mpi.params import uniform

qubits = 4
system_size = 2**qubits

U1 = diagonal(diagonal_uniform, parameter_function = uniform)
U2 = circulant(circulant_complete, parameter_function = uniform)

qwoa = ansatz(system_size)
qwoa.set_unitaries([U1, U2], 0)
qwoa.set_depth(1)
qwoa.execute()
qwoa.print_optimiser_result()
