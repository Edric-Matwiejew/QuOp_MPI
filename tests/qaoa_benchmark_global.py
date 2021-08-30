from quop_mpi import ansatz
from quop_mpi.unitaries import diagonal, sparse
from quop_mpi.states import equal
from quop_mpi.operators.diagonal import uniform as diag_uniform
from quop_mpi.operators.sparse import hypercube
from quop_mpi.params import uniform

qubits = 5
system_size = 2**qubits

U1 = diagonal(
        diag_uniform,
        parameter_function = uniform)

U2 = sparse(
        hypercube,
        parameter_function = uniform)

qaoa = ansatz(system_size)
qaoa.set_parallel("jacobian")
qaoa.set_unitaries([U1, U2])
qaoa.set_observables(0)
qaoa.set_log('test', 'test1')
qaoa.verbose_objective = True
qaoa.benchmark(
        range(1,4),
        5,
        param_persist = True,
        verbose = True,
        filename = 'test',
        label = 'test',
        save_action = 'a')
