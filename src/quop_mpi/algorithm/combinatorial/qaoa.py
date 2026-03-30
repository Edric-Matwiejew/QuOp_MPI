"""Default QAOA implementation."""

from __future__ import annotations

from .qaoa_sparse import QAOASparse


class QAOA(QAOASparse):
    """Simulate the :ref:`QAOA <QAOA>`.

    This remains the default sparse-hypercube implementation while the
    transverse-field-backed variant is kept alongside it for benchmarking.
    """

    pass
