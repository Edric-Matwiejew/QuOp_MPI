"""Composite unitary propagator for multidimensional circulant operators."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from ..._lib.propagator import Propagator
from ...unitary import UnitaryBase


class Unitary(UnitaryBase):
    """Unitary propagator for composite (multidimensional) circulant operators.

    Handles tensor-product structure of circulant graphs across multiple
    coordinate dimensions.
    """

    def __init__(self, Ns, *args: Any, **kwargs: Any) -> None:  # noqa: N803, ANN001, ANN401
        """Initialise the composite unitary propagator.

        Parameters
        ----------
        Ns : array_like of int
            Number of grid points per coordinate dimension.
        *args
            Forwarded to :class:`~quop_mpi.unitary.UnitaryBase`.
        **kwargs
            Forwarded to :class:`~quop_mpi.unitary.UnitaryBase`.
        """

        self.Ns = np.array(Ns, dtype=np.int32)

        super().__init__(*args, **kwargs)

        self.unitary_type = "composite"

        self.context = None

        self.comm_size_constraints = [np.array(Ns, dtype=np.int32)]

        self.planner = True

    def assign_backend(self, backend: ModuleType) -> None:
        """Assign the compute backend for composite propagation."""
        self.propagator_module = backend.composite_propagator
        self.propagators = [Propagator(self.propagator_module.composite_propagator_wrapper)]

    def gen_operator(self, *args: Any) -> None:  # noqa: ANN401
        """Generate the composite operator and plan the propagator."""
        self.propagators[0].plan(self.context)
        self.planned = True  # Mark as planned so destroy() is called during cleanup
        super().gen_operator(*args)
        self.propagators[0].gen_operator([self.Ns, self.operator.flatten()])

    def propagate(self, t: np.ndarray) -> None:
        """Apply the composite propagator with parameters ``t``."""
        self.propagators[0].propagate(t)

    def destroy(self) -> None:
        """Free composite propagator resources."""
        self.propagators[0].destroy()
