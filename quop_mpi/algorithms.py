from mpi4py import MPI
from quop_mpi.__init__ import ansatz

class __phase_and_mixer(ansatz):

    def __init__(self, system_size, MPI_communicator = MPI.COMM_WORLD, parallel = "global"):

        super().__init__(system_size, MPI_communicator, parallel = "global")

        self.operator_function = None
        self.param_function = None

    def set_qualities(self, operator_function, **kwargs):
        self.operator_function = operator_function
        self.operator_kwargs = kwargs

    def set_params(self, param_function, **kwargs):
        self.param_function
        self.param_kwargs = kwargs

    def _pre(self):

        if self.operator_function is None:
            raise RuntimeError("Rank {}: Solution qualities not defined.".format(self.rank))

        if self.param_function is None:
            from quop_mpi.params import uniform
            self.set_params(uniform)

class qwoa(__phase_and_mixer):

    def pre(self):

        self._pre()

        from quop_mpi.unitaries import diagonal, circulant
        from quop_mpi.operators import circulant_complete

        UQ = diagonal(
                self.operator_function,
                operator_kwargs = self.operator_kwargs,
                parameter_function = self.param_function)

        UW = circulant(
                circulant_complete,
                parameter_function = self.param_function,
                parameter_kwargs = self.param_kwargs)

        self.set_unitaries([UQ, UW], 0)

        super().pre()

class qaoa(__phase_and_mixer):

    def pre(self):

        self._pre()

        from quop_mpi.unitaries import diagonal, sparse
        from quop_mpi.operators import sparse_hypercube

        UQ = diagonal(
                self.operator_function,
                operator_kwargs = self.operator_kwargs,
                parameter_function = self.param_function)

        UW = sparse(
                sparse_hypercube,
                parameter_function = self.param_function,
                parameter_kwargs = self.param_kwargs)

        self.set_unitaries([UQ, UW], 0)

        super().pre()

