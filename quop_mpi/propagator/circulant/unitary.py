"""Circulant graph unitary propagator."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from ..._lib.propagator import Propagator
from ...unitary import UnitaryBase


class Unitary(UnitaryBase):
    """Unitary propagator for circulant graph operators.

    Uses FFT-based propagation for efficient simulation of circulant
    mixing unitaries.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialise the circulant unitary propagator."""

        super().__init__(*args, **kwargs)

        self.unitary_type = "circulant"

        self.context = None

    def assign_backend(self, backend: ModuleType) -> None:
        """Assign the compute backend for circulant propagation."""
        self.propagator_module = backend.circulant_propagator
        self.propagators = [Propagator(self.propagator_module.circulant_propagator_wrapper)]

    def gen_operator(self, *args: Any) -> None:  # noqa: ANN401
        """Generate the circulant operator and plan the propagator."""
        self.propagators[0].plan(self.context)
        self.planned = True  # Mark as planned so destroy() is called during cleanup
        super().gen_operator(*args)
        self.propagators[0].gen_operator([np.real(self.operator).astype(np.float64)])

    def propagate(self, t: np.ndarray) -> None:
        """Apply the circulant propagator with parameter ``t``."""
        self.propagators[0].propagate(t[0])

    def destroy(self) -> None:
        """Free circulant propagator resources."""
        self.propagators[0].destroy()
