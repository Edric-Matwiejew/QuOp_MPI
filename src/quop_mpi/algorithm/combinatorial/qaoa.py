"""Default QAOA implementation."""

from __future__ import annotations

from .qaoa_transverse_field import QAOATransverseField


class QAOA(QAOATransverseField):
    """Simulate the :ref:`QAOA <QAOA>`.

    Uses the transverse-field mixer propagator, which applies per-qubit
    :math:`R_X(\\theta)` rotations directly on the distributed statevector
    without constructing a sparse operator.  The previous sparse-hypercube
    implementation is available as :class:`QAOASparse`.
    """

    pass
