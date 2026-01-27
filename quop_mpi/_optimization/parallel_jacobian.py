# cspell:words subcomm subcomms maxcomm jacobian jaccomm neval
"""Parallel Jacobian computation mixin for QVA optimization."""

from __future__ import annotations

from typing import Callable, Union, TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from .finite_differences import forward_differences, central
from .._utils._interface import interface

if TYPE_CHECKING:
    from ..Ansatz import Ansatz
    from .._utils._mpi import subcomms


class Jacobian:
    """Mixin providing parallel Jacobian computation for :class:`~quop_mpi.Ansatz`.

    This class is not intended to be instantiated directly. It provides methods
    for parallel computation of the objective function gradient using MPI
    subcommunicators.

    Requires the :class:`Communicator` mixin to be present in the class hierarchy
    to provide subcommunicator management (nodes_per_subcomm, processes_per_node,
    maxcomm, subcomms attributes).
    """

    # Type hints for attributes provided by Ansatz and Communicator mixin
    if TYPE_CHECKING:
        MPI_COMM_WORLD: MPI.Intracomm
        MPI_COMM: MPI.Intracomm
        subcomms: subcomms
        variational_parameters: np.ndarray
        n_free_params: int
        record_objective: bool
        n_evolutions: int
        stop: bool
        optimiser_args: dict
        reset: bool
        # From Communicator mixin
        nodes_per_subcomm: int | None
        processes_per_node: int | None
        maxcomm: int | None

    def _init_jacobian(self):
        """Initialize jacobian-related instance variables.

        Called by :meth:`Ansatz.__init__`.
        """
        self.jacobian_input: list | None = None
        self.jacobian: object = None
        self.var: int | None = None
        self.jac_ranks: list | None = None
        self.h: float = np.sqrt(np.finfo(float).eps)
        self.neval_mpi_jac: int = 0
        self.var_map: list | None = None

    def set_parallel_jacobian(
        self,
        nodes_per_subcomm: int,
        processes_per_node: int,
        maxcomm: int,
        method: Union[str, Callable] = "forward",
        h: float = None,
    ):
        """Specify :term:`optimisation<optimiser>` of the :term:`variational
        parameters` using parallel computation of the jacobian.

        This creates MPI subcommunicators containing duplicates of the
        :class:`~quop_mpi.Ansatz` instance which return partial derivative information to
        the root MPI process during optimisation.

        Parameters
        ----------
        nodes_per_subcomm : int
            MPI nodes per subcommunicator
        processes_per_node : int
            MPI processors associated with each node
        maxcomm : int
            maximum number of created MPI subcommunicators (and :class:`~quop_mpi.Ansatz`
            instance duplicates) if `nodes_per_subcomm > 1`, or the maximum
            number of MPI subcommunicators per node if `nodes_per_subcomm = 1`
        method :{'forward', 'central'} or callable, optional
            'forward' or 'central' to use the forward difference or central
            difference method for numerical approximation of the partial
            derivatives, or a QuOp Jacobian Function, by default 'forward'
        h : float, optional
            step-size used by the forward or central difference methods, by
            default :literal:`np.sqrt(np.finfo(float).eps)`
        """

        # Jacobian-specific attributes
        self.jacobian_input = [method]
        self.h = h if h is not None else np.sqrt(np.finfo(float).eps)

        # Communicator attributes (from Communicator mixin)
        self.nodes_per_subcomm = nodes_per_subcomm
        self.processes_per_node = processes_per_node
        self.maxcomm = maxcomm

        self.reset = True

    def _update_var_map(self):
        """Queries :literal:`Unitary` instances passed to the :class:`~quop_mpi.Ansatz` instance via the
        :meth:`~quop_mpi.Ansatz.set_unitaries` methods to determine the number and ordering of
        QVA variational parameters.
        """
        if self.subcomms.get_n_subcomms() > 1:
            self.var_map = [[] for _ in range(self.subcomms.get_n_subcomms())]
            if self.subcomms.in_subcomm():
                n_params = self.n_free_params
                for var in range(n_params):
                    self.var_map[1:][var % (self.subcomms.get_n_subcomms() - 1)].append(
                        var
                    )
        else:
            self.var_map = None

    def _parse_jacobian(self):
        """Bind a QuOp Jacobian Function to the attributes of an instantiated
        :class:`~quop_mpi.Ansatz` instance.
        """
        self.jacobian = interface(
            [self], self.jacobian_input[0], "jacobian", self.subcomms.SUBCOMM
        )

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
                f"Parallel jacobian requested but only 1 subcommunicator could be created "
                f"(requested maxcomm={self.maxcomm}). Falling back to scipy's default "
                f"finite difference jacobian.",
                RuntimeWarning,
            )
            return False
        return False

    def _mpi_jacobian(self, x: np.ndarray[float]) -> Union[float, None]:
        """Compute the objective function gradient with parallel
        instances of the :class:`~quop_mpi.Ansatz` class.

        Parameters
        ----------
        x : ndarray[float]
            1-D real array of free variational parameters

        Returns
        -------
        float or None
            returns the objective function gradient to rank 0 in
            :attr:`~quop_mpi.Ansatz.MPI_COMM_WORLD`, None otherwise
        """
        # Guard: if JACCOMM is None, fall back to scipy's default jacobian
        if self.subcomms.JACCOMM is None:
            return None

        self.subcomms.JACCOMM.barrier()
        self.stop = self.subcomms.JACCOMM.bcast(self.stop, 0)

        if self.stop:
            self.subcomms.JACCOMM.barrier()
            return

        self.variational_parameters = self.subcomms.JACCOMM.bcast(
            self.variational_parameters, 0
        )

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
                self.jacobian.update_parameters()
                # Pass the parameter index - jacobian.call computes partial derivative
                partials.append(self.jacobian.call(var))

        if self.subcomms.JACCOMM.Get_rank() == 0:
            jacobian = np.zeros(self.n_free_params, dtype=np.float64)
            for root, mapping in zip(self.subcomms.get_subcomm_roots(), self.var_map):
                if root > 0:
                    for var in mapping:
                        self.MPI_COMM_WORLD.Recv(
                            [jacobian[var : var + 1], MPI.DOUBLE], source=root, tag=var
                        )

        elif self.subcomms.SUBCOMM.Get_rank() == 0:
            jacobian = None
            for part, mapping in zip(
                partials, self.var_map[self.subcomms.get_subcomm_index()]
            ):
                self.MPI_COMM_WORLD.Send(
                    [np.array([part]), MPI.DOUBLE], dest=0, tag=mapping
                )
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
