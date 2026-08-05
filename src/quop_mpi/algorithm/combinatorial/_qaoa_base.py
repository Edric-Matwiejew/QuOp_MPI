"""Shared implementation for QAOA variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

from ..._scope import scope
from ...ansatz import Ansatz
from ...propagator import diagonal

if TYPE_CHECKING:
    from typing import Callable

    from ...unitary import UnitaryBase

    Intracomm = MPI.Intracomm


class _QAOABase(Ansatz):
    """Common QAOA setup shared by mixer-specific variants."""

    def __init__(self, system_size: int, MPI_communicator: Intracomm = MPI.COMM_WORLD) -> None:  # noqa: N803
        """Initialise a QAOA-like instance."""
        if system_size < 1 or (system_size & (system_size - 1)) != 0:
            raise ValueError(
                f"QAOA requires system_size = 2**n for integer n, got {system_size}."
            )
        super().__init__(system_size, MPI_communicator)

        self.operator_function = None
        self.param_function = None
        self.param_dict = None

    def set_qualities(self, function: Callable, observables_dict: dict | None = None) -> None:
        """Define the problem qualities used by the phase-separation unitary."""
        self.set_observables(function, observables_dict)

    def set_params(self, param_function: Callable, param_dict: dict | None = None) -> None:
        """Define the parameter function for the phase and mixer unitaries."""
        self.param_function = param_function
        self.param_dict = param_dict

    def _build_phase_unitary(self) -> UnitaryBase:
        """Create the standard QAOA phase-separation unitary."""
        return diagonal.Unitary(
            diagonal.operator.observables,
            parameter_function=self.param_function,
            param_dict=self.param_dict,
        )

    def _build_mixer_unitary(self) -> UnitaryBase:
        """Create the mixer unitary for a specific QAOA variant."""
        raise NotImplementedError("QAOA subclasses must implement _build_mixer_unitary().")

    @scope("world")
    def setup(self) -> None:
        """Configure the QAOA unitaries and prepare the ansatz for execution."""
        if not self.setup_called:
            if self.observable_function is None:
                raise RuntimeError(
                    "Rank {}: Solution qualities not defined.".format(
                        self.MPI_COMM_WORLD.Get_rank()
                    )
                )

            if self.param_function is None:
                from ...param.rand import uniform

                self.set_params(uniform)

            self.set_unitaries([self._build_phase_unitary(), self._build_mixer_unitary()])

        super().setup()
