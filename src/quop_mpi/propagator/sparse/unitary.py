"""Sparse matrix unitary propagator."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from quop_mpi._lib.propagator import Propagator
from quop_mpi.unitary import UnitaryBase


class Unitary(UnitaryBase):
    """Compute the action of a :term:`mixing unitary` with a sparse
    :term:`operator` or a sequence of mixing-unitaries with sparse
    operators (see the :literal:`unitary_n_params` attribute below).

    **Inheritance Diagram:**

        .. graphviz::

            digraph "sphinx-ext-graphviz" {
                rankdir="LR";
                node [fontsize="10"];
                Unitary[label="quop_mpi.unitary", shape="rectangle"];
                unitary[label="quop_mpi.propagator.sparse.unitary", shape="rectangle"];

                Unitary -> unitary;
            }

    See :class:`quop_mpi.unitary`.

    Attributes
    ----------
    unitary_type
        :literal:`'sparse'`
    planner
        :literal:`false`
    unitary_n_params
        Set on initialisation to :literal:`1` or more. If :literal:`unitary_n_parameters > 1`,
        the :term:`Operator Function` must return a :literal:`list[csr_matrix]` of
        length :literal:`unitary_n_parameters` containing :literal:`csr_matrix` partitions of
        of :literal:`local_i` rows.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialise the sparse unitary propagator."""

        super().__init__(*args, **kwargs)

        self.unitary_type = "sparse"

        self.context = None
        self.backend = None

    def assign_backend(self, backend: ModuleType) -> None:
        """Assign the compute backend for sparse propagation."""
        self.propagator_module = backend.sparse_propagator

        self.propagators = []
        for _ in range(self.unitary_n_params):
            self.propagators.append(Propagator(self.propagator_module.sparse_propagator_wrapper))

    def gen_operator(self, *args: Any) -> None:  # noqa: ANN401
        """Generate the sparse operator and plan the propagators."""
        for propagator in self.propagators:
            propagator.plan(self.context)

        self.planned = True  # Mark as planned so destroy() is called during cleanup
        super().gen_operator(*args)

        # Unpack operator result - may have 3 or 4 elements depending on source
        if len(self.operator) == 4:
            self.W_row_starts, self.W_col_indexes, self.W_values, self.is_unit_valued = (
                self.operator
            )
        else:
            self.W_row_starts, self.W_col_indexes, self.W_values = self.operator
            # Detect unit-valued from None values
            self.is_unit_valued = self.W_values is None

        for i, propagator in enumerate(self.propagators):
            if self.is_unit_valued:
                # Unit-valued matrix: only pass row_starts and col_indexes
                # Fortran will detect this and set has_values = .false.
                operator_args = [
                    self.W_row_starts[i],
                    self.W_col_indexes[i],
                ]
            else:
                # Full matrix with explicit values
                operator_args = [
                    self.W_row_starts[i],
                    self.W_col_indexes[i],
                    self.W_values[i],
                ]
            propagator.gen_operator(operator_args, prepare_sparse_csr=True)

    def propagate(self, ts: np.ndarray) -> None:
        """Apply sparse propagators with parameters ``ts``."""
        for t, propagator in zip(ts, self.propagators, strict=True):
            propagator.propagate(np.abs(t))

    def destroy(self) -> None:
        """Free sparse propagator resources."""
        for propagator in self.propagators:
            propagator.destroy()
