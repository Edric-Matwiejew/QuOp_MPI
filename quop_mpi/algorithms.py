from mpi4py import MPI
from quop_mpi.__init__ import phase_and_mixer

class qwoa(phase_and_mixer):

    def pre(self):

        self._pre()

        from quop_mpi.unitaries import diagonal, circulant
        from quop_mpi.operators import circulant_complete

        UQ = diagonal(
                self.operator_function,
                operator_kwargs = self.operator_function_kwargs,
                parameter_function = self.param_function)

        UW = circulant(
                circulant_complete,
                parameter_function = self.param_function,
                parameter_kwargs = self.param_kwargs)

        self.set_unitaries([UQ, UW])

        super().pre()

class qaoa(phase_and_mixer):

    def pre(self):

        self._pre()

        from quop_mpi.unitaries import diagonal, sparse
        from quop_mpi.operators import sparse_hypercube

        UQ = diagonal(
                self.operator_function,
                operator_kwargs = self.operator_function_kwargs,
                parameter_function = self.param_function)

        UW = sparse(
                sparse_hypercube,
                parameter_function = self.param_function,
                parameter_kwargs = self.param_kwargs)

        self.set_unitaries([UQ, UW])

        super().pre()

