# cspell:words subcomm subcomms maxcomm
"""MPI Communicator management mixin for QVA simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

from . import config
from ._scope import scope
from ._utils._comm_size import QuopMpiLayout

if TYPE_CHECKING:
    pass


class Communicator:
    """Mixin providing MPI subcommunicator management for :class:`~quop_mpi.ansatz`.

    Provides read-only properties for partitioning data (``local_i``,
    ``local_i_offset``, ``alloc_local``, ``partition_table``, ``MPI_COMM``)
    that delegate to the underlying :class:`QuopMpiLayout`.  Attempting to
    assign to these attributes on the Ansatz raises ``AttributeError``.
    """

    # Type hints for attributes provided by Ansatz
    if TYPE_CHECKING:
        MPI_COMM_WORLD: MPI.Intracomm

    def _init_communicator(self):
        """Initialize communicator-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        # Number of parallel Jacobian workers (1 = single-communicator mode)
        self.n_jacobian_workers: int = 1

        # Layout is created in _gen_parallel, holds split/topo info before negotiate
        self._layout: QuopMpiLayout | None = None

    # -- Read-only properties backed by QuopMpiLayout ----------------

    @property
    def subcomms(self):
        """Alias for ``_layout`` (backward compatibility)."""
        return self._layout

    @property
    def local_i(self):
        """Number of elements in this rank's partition."""
        layout = self._layout
        if layout is None:
            return 0
        return layout.local_i

    @local_i.setter
    def local_i(self, _val):
        raise AttributeError("local_i is read-only on Ansatz; value comes from QuopMpiLayout")

    @property
    def local_i_offset(self):
        """Global index offset for this rank's partition."""
        layout = self._layout
        if layout is None:
            return 0
        return layout.local_i_offset

    @local_i_offset.setter
    def local_i_offset(self, _val):
        raise AttributeError(
            "local_i_offset is read-only on Ansatz; " "value comes from QuopMpiLayout"
        )

    @property
    def alloc_local(self):
        """Size of the local allocation (may exceed ``local_i`` for padding)."""
        layout = self._layout
        if layout is None:
            return 0
        return layout.alloc_local

    @alloc_local.setter
    def alloc_local(self, _val):
        raise AttributeError(
            "alloc_local is read-only on Ansatz; " "value comes from QuopMpiLayout"
        )

    @property
    def partition_table(self):
        """1-D integer array describing the global partitioning scheme."""
        layout = self._layout
        if layout is None:
            return None
        return layout.partition_table

    @partition_table.setter
    def partition_table(self, _val):
        raise AttributeError(
            "partition_table is read-only on Ansatz; " "value comes from QuopMpiLayout"
        )

    @property
    def MPI_COMM(self):  # noqa: N802
        """MPI subcommunicator for the active worker group."""
        layout = self._layout
        if layout is not None:
            sc = layout.SUBCOMM
            if sc is not None:
                return sc
        return self.MPI_COMM_WORLD

    @MPI_COMM.setter
    def MPI_COMM(self, _val):  # noqa: N802
        raise AttributeError(
            "MPI_COMM is read-only on Ansatz; " "value comes from QuopMpiLayout.SUBCOMM"
        )

    # -- Lifecycle ---------------------------------------------------

    @scope("world")
    def _gen_parallel(self):
        """Creates MPI subcommunicators via Fortran split_workers.

        Uses topology-aware splitting from Fortran:
        - Node-aligned when n_workers <= n_nodes
        - Round-robin fallback when n_workers > n_nodes

        If ``n_jacobian_workers`` exceeds the communicator size it is
        clamped to the number of available ranks so that Fortran
        ``split_workers`` does not reject the request.
        """
        import warnings

        backend_flag = 1 if config.backend == "wavefront" else 0

        n_workers = self.n_jacobian_workers
        comm_size = self.MPI_COMM_WORLD.Get_size()

        if n_workers > comm_size:
            warnings.warn(
                f"n_jacobian_workers={n_workers} exceeds communicator size "
                f"({comm_size}); clamping to {comm_size}.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_workers = comm_size
            self.n_jacobian_workers = n_workers

        # Create layout with worker splitting
        self._layout = QuopMpiLayout.create_workers(
            n_workers,
            self.MPI_COMM_WORLD,
            backend_flag,
        )

    @scope("world")
    def _post_parallel(self):
        """Free subcommunicators associated with the :class:`~quop_mpi.ansatz` instance.

        Called on simulation completion or when destroying the Ansatz instance.
        """
        if self._layout is not None:
            self._layout.free()
            self._layout = None
