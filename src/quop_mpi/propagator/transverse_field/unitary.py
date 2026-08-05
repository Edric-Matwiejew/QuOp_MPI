"""Transverse-field unitary propagator."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from ..._lib.propagator import Propagator
from ...unitary import UnitaryBase


class Unitary(UnitaryBase):
    """Apply the same single-qubit :math:`R_X(\\theta)` rotation to each qubit.

    The operator is implicit: the backend computes the transverse-field action
    directly on the distributed statevector without constructing an operator
    array.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialise the transverse-field mixer."""
        super().__init__(
            operator_function=lambda: None,
            operator_n_params=0,
            unitary_n_params=1,
            **kwargs,
        )

        self.unitary_type = "transverse_field"
        self.context = None

    def assign_backend(self, backend: ModuleType) -> None:
        """Assign the native backend for transverse-field propagation."""
        try:
            self.propagator_module = backend.transverse_field_propagator
        except AttributeError as exc:
            raise NotImplementedError(
                "The transverse_field propagator is not implemented for this backend."
            ) from exc

        self.propagators = [
            Propagator(self.propagator_module.transverse_field_propagator_wrapper)
        ]

    def gen_operator(self) -> None:
        """Plan the native propagator.

        A tiny placeholder array keeps the wrapper call pattern aligned with
        the other propagators even though this unitary has no explicit
        operator data to generate.
        """
        self.propagators[0].plan(self.context)
        self.planned = True
        self.propagators[0].gen_operator([np.array([0], dtype=np.int32)])

    def propagate(self, theta: np.ndarray) -> None:
        """Apply the transverse-field layer with angle ``theta``."""
        self.propagators[0].propagate(theta[0])

    def destroy(self) -> None:
        """Free native propagator resources."""
        self.propagators[0].destroy()
