"""Transverse-field QAOA implementation."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from ...propagator import transverse_field
from ._qaoa_base import _QAOABase


class _QAOATransverseFieldMixer(transverse_field.Unitary):
    """Internal mixer preserving the standard QAOA beta semantics."""

    def propagate(self, beta: np.ndarray) -> None:
        """Apply the transverse-field layer with ``theta = 2 * beta``."""
        super().propagate(np.array([2.0 * beta[0]], dtype=np.float64))


class QAOATransverseField(_QAOABase):
    """Simulate QAOA using the transverse-field mixer propagator.

    This variant preserves the usual QAOA mixer parameter semantics by mapping
    the QAOA beta parameter to a transverse-field rotation angle
    ``theta = 2 * beta`` internally.

    Note
    ----
    The current transverse-field propagator is the aligned MPI-only
    implementation, so this variant inherits its present communicator/layout
    constraints.
    """

    def __init__(self, system_size: int, MPI_communicator: MPI.Intracomm = MPI.COMM_WORLD) -> None:  # noqa: N803
        """Initialise a transverse-field-mixer QAOA instance."""
        super().__init__(system_size, MPI_communicator)

    def _build_mixer_unitary(self):
        """Create the transverse-field mixer with QAOA beta semantics."""
        return _QAOATransverseFieldMixer(
            parameter_function=self.param_function,
            param_dict=self.param_dict,
        )
