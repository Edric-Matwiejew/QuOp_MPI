from importlib import import_module
from ..propagator import diagonal, sparse
from .phase_and_mixer import phase_and_mixer


class qaoa(phase_and_mixer):
    """A __setup-defined :class:`Ansatz` that implements the Quantum Approximation
    Optimisation Algorithm.

    See :class:`phase_and_mixer`.
    """

    def setup(self):

        if not self.setup_called:

            if self.operator_function is None:
                raise RuntimeError(
                    "Rank {}: Solution qualities not defined.".format(self.rank)
                )

            if self.param_function is None:

                from ..param.rand import uniform

                self.set_params(uniform)

            UQ = diagonal.unitary(
                self.operator_function,
                operator_kwargs=self.operator_function_kwargs,
                parameter_function=self.param_function,
            )

            UW = sparse.unitary(
                sparse.operator.hypercube,
                parameter_function=self.param_function,
                parameter_kwargs=self.param_kwargs,
            )

            self.set_unitaries([UQ, UW])

        super().setup()
