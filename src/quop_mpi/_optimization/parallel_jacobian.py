# cspell:words subcomm subcomms maxcomm jacobian jaccomm neval
"""Parallel Jacobian computation mixin for QVA optimization."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from time import perf_counter
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
        self.var: int = -999  # Placeholder value to detect if it's not being set correctly
        self.var_map: list[list[int]] | None = None
        self._trace_parallel_jacobian: bool = False
        self._trace_parallel_jacobian_path: Path | None = None
        self._trace_parallel_jacobian_counter: int = 0

        trace_setting = os.getenv("QUOP_TRACE_PARALLEL_JACOBIAN", "").strip()
        if trace_setting:
            self._trace_parallel_jacobian = True
            if trace_setting == "1":
                trace_dir = Path.cwd()
            else:
                trace_dir = Path(trace_setting)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_parallel_jacobian_path = trace_dir / "parallel_jacobian_trace.rank-pending.log"

    def _trace_jacobian_event(self, event: str, **fields: Any) -> None:
        """Append an opt-in trace line for MPI Jacobian progress diagnostics."""
        if not self._trace_parallel_jacobian:
            return

        trace_path = self._trace_parallel_jacobian_path
        if trace_path is None:
            return

        if trace_path.name.endswith("rank-pending.log"):
            world_rank = self.MPI_COMM_WORLD.Get_rank()
            trace_path = trace_path.with_name(f"parallel_jacobian_trace.rank-{world_rank}.log")
            self._trace_parallel_jacobian_path = trace_path

        jaccomm = self.subcomms.JACCOMM
        subcomm = self.subcomms.SUBCOMM
        metadata = {
            "event": event,
            "counter": self._trace_parallel_jacobian_counter,
            "world_rank": self.MPI_COMM_WORLD.Get_rank(),
            "subcomm_index": self.subcomms.get_subcomm_index(),
            "subcomm_rank": subcomm.Get_rank() if subcomm is not None else -1,
            "jac_rank": jaccomm.Get_rank() if jaccomm is not None else -1,
            "stop": int(bool(self.stop)),
        }
        metadata.update(fields)
        line = " ".join(f"{key}={value}" for key, value in metadata.items())

        with trace_path.open("a", encoding="ascii") as handle:
            handle.write(f"{perf_counter():.9f} {line}\n")

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

        # Jacobian-specific attributes
        self.jacobian_input = [method]
        self.h = h if h is not None else np.sqrt(np.finfo(float).eps)

        # Communicator attribute (from Communicator mixin)
        self.n_jacobian_workers = n_workers

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
            returns the objective function gradient to rank 0 in
            :attr:`~quop_mpi.ansatz.MPI_COMM_WORLD`, None otherwise
        """
        # Guard: if JACCOMM is None, fall back to scipy's default jacobian
        if self.subcomms.JACCOMM is None:
            return None

        self._trace_parallel_jacobian_counter += 1
        jac_call = self._trace_parallel_jacobian_counter
        self._trace_jacobian_event(
            "jacobian.enter",
            jac_call=jac_call,
            x_size=-1 if x is None else len(x),
        )

        barrier_start = perf_counter()
        self._trace_jacobian_event("jacobian.barrier.enter", jac_call=jac_call, phase="start")
        self.subcomms.JACCOMM.barrier()
        self._trace_jacobian_event(
            "jacobian.barrier.exit",
            jac_call=jac_call,
            phase="start",
            wait_s=f"{perf_counter() - barrier_start:.6f}",
        )

        bcast_start = perf_counter()
        self.stop = self.subcomms.JACCOMM.bcast(self.stop, 0)
        self._trace_jacobian_event(
            "jacobian.bcast.stop",
            jac_call=jac_call,
            duration_s=f"{perf_counter() - bcast_start:.6f}",
        )

        if self.stop:
            stop_barrier_start = perf_counter()
            self._trace_jacobian_event(
                "jacobian.barrier.enter",
                jac_call=jac_call,
                phase="stop",
            )
            self.subcomms.JACCOMM.barrier()
            self._trace_jacobian_event(
                "jacobian.barrier.exit",
                jac_call=jac_call,
                phase="stop",
                wait_s=f"{perf_counter() - stop_barrier_start:.6f}",
            )
            self._trace_jacobian_event("jacobian.exit", jac_call=jac_call, reason="stop")
            return

        params_bcast_start = perf_counter()
        broadcast_parameters = self.subcomms.JACCOMM.bcast(self.variational_parameters, 0)
        self.variational_parameters = (
            None
            if broadcast_parameters is None
            else np.asarray(broadcast_parameters, dtype=np.float64)
        )
        self._trace_jacobian_event(
            "jacobian.bcast.parameters",
            jac_call=jac_call,
            duration_s=f"{perf_counter() - params_bcast_start:.6f}",
            params_size=-1 if broadcast_parameters is None else len(broadcast_parameters),
        )

        x_bcast_start = perf_counter()
        x = np.asarray(self.subcomms.JACCOMM.bcast(x, 0), dtype=np.float64)
        self._trace_jacobian_event(
            "jacobian.bcast.x",
            jac_call=jac_call,
            duration_s=f"{perf_counter() - x_bcast_start:.6f}",
            x_size=len(x),
        )

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
                self.jacobian.update_parameters()
                # Pass the parameter index - jacobian.call computes partial derivative
                eval_start = perf_counter()
                self._trace_jacobian_event(
                    "jacobian.partial.enter",
                    jac_call=jac_call,
                    var=var,
                )
                partials.append(self.jacobian.call())
                self._trace_jacobian_event(
                    "jacobian.partial.exit",
                    jac_call=jac_call,
                    var=var,
                    duration_s=f"{perf_counter() - eval_start:.6f}",
                )

        if self.subcomms.JACCOMM.Get_rank() == 0:
            jacobian = np.zeros(self.n_free_params, dtype=np.float64)
            roots = self.subcomms.get_subcomm_roots()
            for root, mapping in zip(roots, self.var_map, strict=True):
                if root > 0:
                    for var in mapping:
                        recv_start = perf_counter()
                        self._trace_jacobian_event(
                            "jacobian.recv.enter",
                            jac_call=jac_call,
                            source=root,
                            var=var,
                        )
                        self.MPI_COMM_WORLD.Recv(
                            [jacobian[var : var + 1], MPI.DOUBLE], source=root, tag=var
                        )
                        self._trace_jacobian_event(
                            "jacobian.recv.exit",
                            jac_call=jac_call,
                            source=root,
                            var=var,
                            duration_s=f"{perf_counter() - recv_start:.6f}",
                        )

        elif self.subcomms.SUBCOMM.Get_rank() == 0:
            jacobian = None
            for part, mapping in zip(
                partials, self.var_map[self.subcomms.get_subcomm_index()], strict=True
            ):
                send_start = perf_counter()
                self._trace_jacobian_event(
                    "jacobian.send.enter",
                    jac_call=jac_call,
                    dest=0,
                    var=mapping,
                )
                self.MPI_COMM_WORLD.Send([np.array([part]), MPI.DOUBLE], dest=0, tag=mapping)
                self._trace_jacobian_event(
                    "jacobian.send.exit",
                    jac_call=jac_call,
                    dest=0,
                    var=mapping,
                    duration_s=f"{perf_counter() - send_start:.6f}",
                )
        else:
            jacobian = None

        end_barrier_start = perf_counter()
        self._trace_jacobian_event("jacobian.barrier.enter", jac_call=jac_call, phase="end")
        self.subcomms.JACCOMM.barrier()
        self._trace_jacobian_event(
            "jacobian.barrier.exit",
            jac_call=jac_call,
            phase="end",
            wait_s=f"{perf_counter() - end_barrier_start:.6f}",
        )

        if self.record_objective:
            reduce_start = perf_counter()
            if self.subcomms.JACCOMM.Get_rank() == 0:
                self.n_evolutions = self.subcomms.JACCOMM.reduce(
                    self.n_evolutions, op=MPI.SUM, root=0
                )
            else:
                self.subcomms.JACCOMM.reduce(self.n_evolutions, op=MPI.SUM, root=0)
                self.n_evolutions = 0
            self._trace_jacobian_event(
                "jacobian.reduce.n_evolutions",
                jac_call=jac_call,
                duration_s=f"{perf_counter() - reduce_start:.6f}",
            )

        if self.subcomms.JACCOMM.Get_rank() == 0:

            self.neval_mpi_jac += 1
            self._trace_jacobian_event(
                "jacobian.exit",
                jac_call=jac_call,
                result="root",
                neval_mpi_jac=self.neval_mpi_jac,
            )
            return jacobian

        else:
            self._trace_jacobian_event("jacobian.exit", jac_call=jac_call, result="worker")
            return None
