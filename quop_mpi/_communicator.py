# cspell:words subcomm subcomms maxcomm
"""MPI Communicator management mixin for QVA simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

from ._utils._mpi import subcomms

if TYPE_CHECKING:
    from .Ansatz import Ansatz


class Communicator:
    """Mixin providing MPI subcommunicator management for :class:`~quop_mpi.Ansatz`.

    This class is not intended to be instantiated directly. It provides methods
    for creating and managing MPI subcommunicators used in QVA simulation.

    The subcommunicator structure supports both single-communicator execution
    and multi-subcommunicator execution for parallel Jacobian computation.
    """

    # Type hints for attributes provided by Ansatz
    if TYPE_CHECKING:
        MPI_COMM_WORLD: MPI.Intracomm
        MPI_COMM: MPI.Intracomm

    def _init_communicator(self):
        """Initialize communicator-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        # Arguments for subcomms class initialisation
        self.nodes_per_subcomm: int | None = None
        self.processes_per_node: int | None = None
        self.maxcomm: int | None = None

        self.setup_parallel: bool = True
        self.subcomms: subcomms | None = None

    def _gen_parallel(self):
        """Creates MPI subcommunicators for QVA simulation.

        Supports both single-communicator execution and multi-subcommunicator
        execution for parallel Jacobian computation. When multiple subcommunicators
        are created, also creates a Jacobian communicator (JACCOMM) for
        coordinating gradient computation.
        """
        self.subcomms = subcomms(
            self.nodes_per_subcomm,
            self.processes_per_node,
            self.maxcomm,
            self.MPI_COMM_WORLD,
        )

        if self.subcomms.in_subcomm():
            self.MPI_COMM = self.subcomms.SUBCOMM

        if self.subcomms.get_n_subcomms() > 1 and self.subcomms.in_subcomm():
            self.subcomms.create_jaccomm()

    def _post_parallel(self):
        """Free subcommunicators associated with the :class:`~quop_mpi.Ansatz` instance.

        Called on simulation completion or when destroying the Ansatz instance.
        """
        self.subcomms.free()
