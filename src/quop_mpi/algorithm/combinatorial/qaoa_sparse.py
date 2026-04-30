"""Sparse-hypercube QAOA implementation."""

from __future__ import annotations

from mpi4py import MPI

from ...propagator import sparse
from ._qaoa_base import _QAOABase


class QAOASparse(_QAOABase):
    """Simulate QAOA using the sparse hypercube mixer."""

    def __init__(self, system_size: int, MPI_communicator: MPI.Intracomm = MPI.COMM_WORLD) -> None:  # noqa: N803
        """Initialise a sparse-mixer QAOA instance."""
        super().__init__(system_size, MPI_communicator)

    def _build_mixer_unitary(self):
        """Create the legacy sparse hypercube mixer."""
        return sparse.Unitary(
            sparse.operator.hypercube,
            parameter_function=self.param_function,
            param_dict=self.param_dict,
        )
