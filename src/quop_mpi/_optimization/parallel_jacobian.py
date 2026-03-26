# cspell:words subcomm subcomms maxcomm jacobian jaccomm neval
"""Parallel Jacobian computation mixin for QVA optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from mpi4py import MPI

from .._scope import scope
from .._utils._interface import Interface
from .finite_differences import central, forward_differences

if TYPE_CHECKING:
    from .._utils._comm_size import QuopMpiLayout


class Jacobian:
    """Mixin providing parallel Jacobian computation for :class:`~quop_mpi.ansatz`.

    This class is not intended to be instantiated directly. It provides methods
    for parallel computation of the objective function gradient using MPI
    subcommunicators.

    Requires the :class:`Communicator` mixin to be present in the class hierarchy
    to provide subcommunicator management (n_jacobian_workers, subcomms attributes).
    """

    # Type hints for attributes provided by Ansatz and Communicator mixin
    if TYPE_CHECKING:
        MPI_COMM_WORLD: MPI.Intracomm
        MPI_COMM: MPI.Intracomm
        subcomms: QuopMpiLayout | None
        variational_parameters: np.ndarray
        n_free_params: int
        record_objective: bool
        n_evolutions: int
        stop: bool
        optimiser_args: dict
        reset: bool
        # From Communicator mixin
        n_jacobian_workers: int

    def _init_jacobian(self):
        """Initialize jacobian-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        self.jacobian_input: list | None = None
        self.jacobian: object = None
        self.jac_ranks: list | None = None
        self.h: float = np.sqrt(np.finfo(float).eps)
        self.neval_mpi_jac: int = 0
        self.var : int = -999  # Placeholder value to detect if it's not being set correctly
        self.var_map: list | None = None

    @scope("world")
    def set_parallel_jacobian(
        self,
        n_workers: int,
        method: str | Callable = "forward",
        h: float = None,
    ):
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

        # Jacobian-specific attributes
        self.jacobian_input = [method]
        self.h = h if h is not None else np.sqrt(np.finfo(float).eps)

        # Communicator attribute (from Communicator mixin)
        self.n_jacobian_workers = n_workers

    @scope("world")
    def _update_var_map(self):
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
    def _parse_jacobian(self):
        """Bind a QuOp Jacobian Function to the attributes of an instantiated
        :class:`~quop_mpi.ansatz` instance.
        """
        self.jacobian = Interface([self], self.jacobian_input[0], "jacobian", self.subcomms.SUBCOMM)

    @scope("subcomm")
    def _configure_parallel_jacobian(self):
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

            self.optimiser_args["jac"] = self._mpi_jacobian
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
            return False
        return False

    @scope("jaccomm")
    def _mpi_jacobian(self, x: np.ndarray[float]) -> float | None:
        """Compute the objective function gradient with parallel
        instances of the :class:`~quop_mpi.ansatz` class.

        Parameters
        ----------
        x : ndarray[float]
            1-D real array of free variational parameters

        Returns
        -------
        float or None
            returns the objective function gradient to rank 0 in
            :attr:`~quop_mpi.ansatz.MPI_COMM_WORLD`, None otherwise
        """
        # Guard: if JACCOMM is None, fall back to scipy's default jacobian
        if self.subcomms.JACCOMM is None:
            return None

        self.subcomms.JACCOMM.barrier()
        self.stop = self.subcomms.JACCOMM.bcast(self.stop, 0)

        if self.stop:
            self.subcomms.JACCOMM.barrier()
            return

        self.variational_parameters = self.subcomms.JACCOMM.bcast(self.variational_parameters, 0)

        x = self.subcomms.JACCOMM.bcast(x, 0)

        if self.subcomms.JACCOMM.Get_rank() != 0:
            # When a parameter map is set, x contains the free parameters.
            # We keep variational_parameters as the free params so that
            # the jacobian functions perturb the correct indices.
            # The mapping to full params happens inside evaluate() -> __to_full().
            self.variational_parameters = x

        partials = []
        if self.subcomms.JACCOMM.Get_rank() != 0:
            for var in self.var_map[self.subcomms.get_subcomm_index()]:
                self.var = var
                print(self.var, flush = True)  # Debug print to check if var is being set correctly
                self.jacobian.update_parameters()
                # Pass the parameter index - jacobian.call computes partial derivative
                partials.append(self.jacobian.call())

        if self.subcomms.JACCOMM.Get_rank() == 0:
            jacobian = np.zeros(self.n_free_params, dtype=np.float64)
            roots = self.subcomms.get_subcomm_roots()
            for root, mapping in zip(roots, self.var_map, strict=True):
                if root > 0:
                    for var in mapping:
                        self.MPI_COMM_WORLD.Recv(
                            [jacobian[var : var + 1], MPI.DOUBLE], source=root, tag=var
                        )

        elif self.subcomms.SUBCOMM.Get_rank() == 0:
            jacobian = None
            for part, mapping in zip(
                partials, self.var_map[self.subcomms.get_subcomm_index()], strict=True
            ):
                self.MPI_COMM_WORLD.Send([np.array([part]), MPI.DOUBLE], dest=0, tag=mapping)
        else:
            jacobian = None

        self.subcomms.JACCOMM.barrier()

        if self.record_objective:
            if self.subcomms.JACCOMM.Get_rank() == 0:
                self.n_evolutions = self.subcomms.JACCOMM.reduce(
                    self.n_evolutions, op=MPI.SUM, root=0
                )
            else:
                self.subcomms.JACCOMM.reduce(self.n_evolutions, op=MPI.SUM, root=0)
                self.n_evolutions = 0

        if self.subcomms.JACCOMM.Get_rank() == 0:

            self.neval_mpi_jac += 1
            return jacobian

        else:
            return None
