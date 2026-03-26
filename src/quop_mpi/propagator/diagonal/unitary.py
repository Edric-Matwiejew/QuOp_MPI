"""Diagonal (phase-shift) unitary propagator."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from quop_mpi._lib.propagator import Propagator
from quop_mpi.unitary import UnitaryBase


class Unitary(UnitaryBase):
    """Compute the action of a :term:`mixing unitary` with a phase_shift
    :term:`operator` or a sequence of mixing-unitaries with phase_shift
    operators (see the :literal:`unitary_n_params` attribute below).

    **Inheritance Diagram:**

        .. graphviz::

            digraph "sphinx-ext-graphviz" {
                rankdir="LR";
                node [fontsize="10"];
                Unitary[label="quop_mpi.unitary", shape="rectangle"];
                unitary[label="quop_mpi.propagator.phase_shift.unitary", shape="rectangle"];

                Unitary -> unitary;
            }

    See :class:`quop_mpi.unitary`.

    Attributes
    ----------
    unitary_type
        :literal:`'phase_shift'`
    planner
        :literal:`false`
    unitary_n_params
        Set on initialisation to :literal:`1` or more. If :literal:`unitary_n_parameters > 1`,
        the :term:`Operator Function` must return a :literal:`list[csr_matrix]` of
        length :literal:`unitary_n_parameters` containing :literal:`csr_matrix` partitions of
        of :literal:`local_i` rows.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialise the diagonal unitary propagator."""

        super().__init__(*args, **kwargs)

        self.unitary_type = "phase_shift"

        self.context = None

    def assign_backend(self, backend: ModuleType) -> None:
        """Assign the compute backend for diagonal propagation."""
        self.propagator_module = backend.diagonal_propagator

        self.propagators = []
        for _ in range(self.unitary_n_params):
            self.propagators.append(Propagator(self.propagator_module.diagonal_propagator_wrapper))

    def gen_operator(self, *args: Any) -> None:  # noqa: ANN401
        """Generate the diagonal operator and plan the propagators."""
        for propagator in self.propagators:
            propagator.plan(self.context)

        self.planned = True  # Mark as planned so destroy() is called during cleanup
        super().gen_operator(*args)

        diagonals = self.operator

        for i, propagator in enumerate(self.propagators):
            if self.unitary_n_params > 1:
                operator_args = [diagonals[i]]
            else:
                operator_args = [diagonals]

            propagator.gen_operator(operator_args)

    def propagate(self, gammas: list[float]) -> None:
        """Apply diagonal phase-shift propagators with parameters ``gammas``."""
        for gamma, propagator in zip(gammas, self.propagators, strict=True):
            propagator.propagate(gamma)

    def destroy(self) -> None:
        """Free diagonal propagator resources."""
        for propagator in self.propagators:
            propagator.destroy()
