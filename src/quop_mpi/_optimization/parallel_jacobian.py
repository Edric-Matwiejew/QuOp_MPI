# cspell:words subcomm subcomms maxcomm jacobian jaccomm neval
"""Parallel Jacobian computation mixin for QVA optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

from .._scope import scope
from .._utils._interface import Interface
from .finite_differences import central, forward_differences

if TYPE_CHECKING:
    from .._utils._comm_size import QuopMpiLayout


JacobianMethod = str | Callable[..., Any]


class Jacobian:
    """Mixin providing parallel Jacobian computation for :class:`~quop_mpi.ansatz`.

    This class is not intended to be instantiated directly. It provides methods
    for parallel computation of the objective function gradient using MPI
    subcommunicators.

    Requires the :class:`Communicator` mixin to be present in the class hierarchy
    to provide subcommunicator management (n_jacobian_workers, subcomms attributes).
    """

    PARALLEL_JAC_COMMAND_EVALUATE = 1
    PARALLEL_JAC_COMMAND_STOP = 2

    # Type hints for attributes provided by Ansatz and Communicator mixin
    if TYPE_CHECKING:
        MPI_COMM_WORLD: MPI.Intracomm
        MPI_COMM: MPI.Intracomm
        subcomms: QuopMpiLayout | None
        variational_parameters: np.ndarray | None
        n_free_params: int
        record_objective: bool
        n_evolutions: int
        stop: bool
        optimiser_args: dict[str, Any] | None
        # From Communicator mixin
        n_jacobian_workers: int

    def _init_jacobian(self) -> None:
        """Initialize jacobian-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        self.jacobian_input: list[JacobianMethod] | None = None
        self.jacobian: Interface | None = None
        self.jac_ranks: list[int] | None = None
        self.h: float = np.sqrt(np.finfo(float).eps)
        self.neval_mpi_jac: int = 0
        self.var: int = -999
        self.var_map: list[list[int]] | None = None
        self.parallel_jacobian_enabled: bool = False
        self._parallel_jacobian_control_active: bool = False

    @scope("world")
    def set_parallel_jacobian(
        self,
        n_workers: int,
        method: JacobianMethod = "forward",
        h: float | None = None,
    ) -> None:
        """Specify :term:`optimisation<optimiser>` of the :term:`variational
        parameters` using parallel computation of the jacobian.

        This creates MPI subcommunicators containing duplicates of the
        :class:`~quop_mpi.ansatz` instance which return partial derivative information to
        the root MPI process during optimisation.

        The Fortran backend uses topology-aware splitting:

        - When ``n_workers <= n_nodes``: whole nodes are assigned to each worker
          to minimise inter-node communication.
        - When ``n_workers > n_nodes``: ranks are distributed round-robin.

        Parameters
        ----------
        n_workers : int
            Number of parallel worker subcommunicators. Each worker computes
            a subset of the Jacobian partial derivatives. Must be >= 1.
        method :{'forward', 'central'} or callable, optional
            'forward' or 'central' to use the forward difference or central
            difference method for numerical approximation of the partial
            derivatives, or a QuOp Jacobian Function, by default 'forward'
        h : float, optional
            step-size used by the forward or central difference methods, by
            default :literal:`np.sqrt(np.finfo(float).eps)`
        """
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")

        from ..ansatz import _Dirty

        previous_workers = self.n_jacobian_workers

        # Jacobian-specific attributes
        self.jacobian_input = [method]
        self.h = h if h is not None else np.sqrt(np.finfo(float).eps)

        # Communicator attribute (from Communicator mixin)
        self.n_jacobian_workers = n_workers

        self._dirty |= _Dirty.OPTIMISER
        if previous_workers != n_workers:
            self._dirty |= _Dirty.WORKER_SPLIT

    @scope("world")
    def _update_var_map(self) -> None:
        """Queries :literal:`Unitary` instances passed to the
        :class:`~quop_mpi.ansatz` instance via the
        :meth:`~quop_mpi.ansatz.set_unitaries` methods to determine the
        number and ordering of QVA variational parameters.
        """
        if self.subcomms.get_n_subcomms() > 1:
            self.var_map = [[] for _ in range(self.subcomms.get_n_subcomms())]
            if self.subcomms.in_subcomm():
                n_params = self.n_free_params
                for var in range(n_params):
                    self.var_map[1:][var % (self.subcomms.get_n_subcomms() - 1)].append(var)
        else:
            self.var_map = None

    @scope("subcomm")
    def _parse_jacobian(self) -> None:
        """Bind a QuOp Jacobian Function to the attributes of an instantiated
        :class:`~quop_mpi.ansatz` instance.
        """
        self.jacobian = Interface([self], self.jacobian_input[0], "jacobian", self.subcomms.SUBCOMM)

    @scope("subcomm")
    def _configure_parallel_jacobian(self) -> bool:
        """Configure parallel jacobian in __gen_optimiser if requested.

        Called from __gen_optimiser in Ansatz.
        """
        if self.jacobian_input is not None and self.subcomms.get_n_subcomms() > 1:
            # Only use parallel jacobian if we actually have multiple subcomms
            if self.jacobian_input[0] == "forward":
                self.jacobian_input = [forward_differences]
            elif self.jacobian_input[0] == "central":
                self.jacobian_input = [central]

            self._parse_jacobian()

            if self.optimiser_args is None:
                raise RuntimeError("optimiser_args must be configured before enabling jacobian.")
            self.optimiser_args["jac"] = self._mpi_jacobian
            self.parallel_jacobian_enabled = True
            return True
        elif self.jacobian_input is not None:
            # User requested parallel jacobian but only 1 subcomm was created
            import warnings

            warnings.warn(
                f"Parallel jacobian requested but only 1 subcommunicator exists "
                f"(n_jacobian_workers={self.n_jacobian_workers}). Falling back to scipy's "
                f"default finite difference jacobian.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.parallel_jacobian_enabled = False
            return False
        self.parallel_jacobian_enabled = False
        return False

    @scope("subcomm")
    def _signal_parallel_jacobian_command(self, command: int) -> None:
        """Signal worker subcommunicators to enter the parallel jacobian path.

        Commands are sent over ROOTCOMM from the optimizer subcomm leader and
        relayed to each worker SUBCOMM leader.
        """
        if not self._parallel_jacobian_control_active:
            return
        if self.subcomms.get_n_subcomms() <= 1:
            return
        if self.subcomms.ROOTCOMM is None:
            raise RuntimeError("ROOTCOMM is required for coordinated parallel jacobian execution.")
        if self.subcomms.get_subcomm_index() != 0 or self.subcomms.SUBCOMM.Get_rank() != 0:
            return

        self.subcomms.ROOTCOMM.bcast(int(command), root=0)

    @scope("subcomm")
    def _await_parallel_jacobian_command(self) -> int | None:
        """Wait for the optimizer leader to request jacobian work or shutdown."""
        if not self._parallel_jacobian_control_active:
            return None
        if self.subcomms.get_subcomm_index() == 0:
            return None
        if self.subcomms.ROOTCOMM is None:
            raise RuntimeError("ROOTCOMM is required for coordinated parallel jacobian execution.")

        if self.subcomms.SUBCOMM.Get_rank() == 0:
            command = self.subcomms.ROOTCOMM.bcast(None, root=0)
        else:
            command = None

        command = self.subcomms.SUBCOMM.bcast(command, root=0)
        return None if command is None else int(command)

    @scope("jaccomm")
    def _mpi_jacobian(self, x: np.ndarray[float] | None) -> np.ndarray[np.float64] | None:
        """Compute the objective function gradient with parallel
        instances of the :class:`~quop_mpi.ansatz` class.

        Parameters
        ----------
        x : ndarray[float] or None
            1-D real array of free variational parameters. Non-root workers may
            pass ``None`` before the broadcast supplies the root value.

        Returns
        -------
        ndarray[float64] or None
            The gathered objective-function gradient on the optimiser
            leader (``JACCOMM`` rank 0, i.e.
            :meth:`~quop_mpi._utils._comm_size.SubComms.optimiser_leader_world_rank`),
            ``None`` on every other rank.  The optimiser leader is not
            guaranteed to coincide with world rank 0.
        """
        # Guard: if JACCOMM is None, fall back to scipy's default jacobian
        if self.subcomms.JACCOMM is None:
            return None

        rank = self.subcomms.JACCOMM.Get_rank()

        self.stop = self.subcomms.JACCOMM.bcast(self.stop, 0)

        if self.stop:
            return

        broadcast_parameters = self.subcomms.JACCOMM.bcast(self.variational_parameters, 0)
        self.variational_parameters = (
            None
            if broadcast_parameters is None
            else np.asarray(broadcast_parameters, dtype=np.float64)
        )

        x = np.asarray(self.subcomms.JACCOMM.bcast(x, 0), dtype=np.float64)

        if rank != 0:
            # When a parameter map is set, x contains the free parameters.
            # We keep variational_parameters as the free params so that
            # the jacobian functions perturb the correct indices.
            # The mapping to full params happens inside evaluate() -> __to_full().
            self.variational_parameters = x

        partials = []
        if rank != 0:
            for var in self.var_map[self.subcomms.get_subcomm_index()]:
                self.var = var
                self.jacobian.update_parameters()
                # Pass the parameter index - jacobian.call computes partial derivative
                partials.append(self.jacobian.call())

        if rank == 0:
            jacobian = np.zeros(self.n_free_params, dtype=np.float64)
            roots = self.subcomms.get_subcomm_roots()
            # roots[0] is the optimizer leader (this rank); skip it when
            # collecting partials.  Use the worker index, not the world rank
            # value, since a worker leader can legitimately be at world rank 0.
            for worker_id, (root, mapping) in enumerate(
                zip(roots, self.var_map, strict=True)
            ):
                if worker_id == 0:
                    continue
                for var in mapping:
                    self.MPI_COMM_WORLD.Recv(
                        [jacobian[var : var + 1], MPI.DOUBLE], source=root, tag=var
                    )

        elif self.subcomms.SUBCOMM.Get_rank() == 0:
            jacobian = None
            # The optimiser leader is at this world rank, not necessarily 0.
            optimizer_world_rank = self.subcomms.optimiser_leader_world_rank()
            for part, mapping in zip(
                partials, self.var_map[self.subcomms.get_subcomm_index()], strict=True
            ):
                self.MPI_COMM_WORLD.Send(
                    [np.array([part]), MPI.DOUBLE], dest=optimizer_world_rank, tag=mapping
                )
        else:
            jacobian = None

        if self.record_objective:
            if rank == 0:
                self.n_evolutions = self.subcomms.JACCOMM.reduce(
                    self.n_evolutions, op=MPI.SUM, root=0
                )
            else:
                self.subcomms.JACCOMM.reduce(self.n_evolutions, op=MPI.SUM, root=0)
                self.n_evolutions = 0

        if rank == 0:

            self.neval_mpi_jac += 1
            return jacobian

        else:
            return None
