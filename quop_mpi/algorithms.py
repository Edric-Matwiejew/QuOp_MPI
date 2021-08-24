from mpi4py import MPI
from quop_mpi.__init__ import phase_and_mixer

class qwoa(phase_and_mixer):

    def pre(self):

        from quop_mpi.unitaries import diagonal, circulant
        from quop_mpi.operators.circulant import complete

        if self.operator_function is None:
            raise RuntimeError("Rank {}: Solution qualities not defined.".format(self.rank))

        if self.param_function is None:
            from quop_mpi.params import uniform
            self.set_params(uniform)

        UQ = diagonal(
                self.operator_function,
                operator_kwargs = self.operator_function_kwargs,
                parameter_function = self.param_function)

        UW = circulant(
                complete,
                parameter_function = self.param_function,
                parameter_kwargs = self.param_kwargs)

        self.set_unitaries([UQ, UW])

        super().pre()

class qaoa(phase_and_mixer):

    def pre(self):

        from quop_mpi.unitaries import diagonal, sparse
        from quop_mpi.operators.sparse import hypercube

        if self.operator_function is None:
            raise RuntimeError("Rank {}: Solution qualities not defined.".format(self.rank))

        if self.param_function is None:
            from quop_mpi.params import uniform
            self.set_params(uniform)

        UQ = diagonal(
                self.operator_function,
                operator_kwargs = self.operator_function_kwargs,
                parameter_function = self.param_function)

        UW = sparse(
                hypercube,
                parameter_function = self.param_function,
                parameter_kwargs = self.param_kwargs)

        self.set_unitaries([UQ, UW])

        super().pre()

