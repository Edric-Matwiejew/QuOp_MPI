"""Circulant graph unitary propagator."""

from __future__ import annotations

import warnings
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
        operator = np.asarray(self.operator)
        compliant = (
            operator.dtype == np.float64
            and operator.flags.c_contiguous
            and not np.iscomplexobj(operator)
        )
        if not compliant:
            warnings.warn(
                "Circulant operator function returned a non-contiguous or "
                "non-float64 array; expected a contiguous ndarray[float64]. "
                "A local copy will be made.",
                RuntimeWarning,
                stacklevel=2,
            )
            operator = np.ascontiguousarray(np.real(operator), dtype=np.float64)
        self.propagators[0].gen_operator([operator])

    def propagate(self, t: np.ndarray) -> None:
        """Apply the circulant propagator with parameter ``t``."""
        self.propagators[0].propagate(t[0])

    def destroy(self) -> None:
        """Free circulant propagator resources."""
        self.propagators[0].destroy()
