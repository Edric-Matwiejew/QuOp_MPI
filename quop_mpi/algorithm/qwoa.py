from importlib import import_module
from ..propagator import diagonal, circulant
from .phase_and_mixer import phase_and_mixer


class qwoa(phase_and_mixer):
    """A pre-defined :class:`Ansatz` that implements the Quantum Walk-based
    Optimisation Algorithm.

    See :class:`phase_and_mixer`.
    """

    def pre(self):

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

        UW = circulant.unitary(
            circulant.operator.complete,
            parameter_function=self.param_function,
            parameter_kwargs=self.param_kwargs,
        )

        self.set_unitaries([UQ, UW])

        super().pre()
