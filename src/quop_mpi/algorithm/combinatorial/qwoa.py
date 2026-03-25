"""Quantum Walk-based Optimisation Algorithm (QWOA) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

from ..._scope import scope
from ...ansatz import Ansatz
from ...propagator import circulant, diagonal

if TYPE_CHECKING:
    from typing import Callable

    Intracomm = MPI.Intracomm


class QWOA(Ansatz):
    """Simulate the :ref:`QWOA <QWOA>`.

    See :class:`quop_mpi.ansatz`.

    Parameters
    ----------
    system_size : int
        :term:`system size` of the simulated :term:`QVA`
    MPI_COMM : Intracomm, optional
        MPI communicator, default ``mpi4py.MPI.COMM_WORLD``
    """

    def __init__(self, system_size: int, MPI_communicator: Intracomm = MPI.COMM_WORLD) -> None:  # noqa: N803
        """Initialise a QWOA instance.

        Parameters
        ----------
        system_size : int
            Number of quantum basis states.
        MPI_communicator : Intracomm, optional
            MPI communicator, by default ``MPI.COMM_WORLD``.
        """

        super().__init__(system_size, MPI_communicator)

        self.operator_function = None
        self.param_function = None

    def set_qualities(self, function: Callable, observable_dict: dict | None = None) -> None:
        """Define the :term:`observables` and :term:`phase-shift unitary` :term:`operator`

        Parameters
        ----------
        function : Callable
            an :term:`Operator Function`
        observable_dict : FunctionDict, optional
            :term:`FunctionDict` for ``function``
        """
        self.set_observables(function, observable_dict)

    def set_params(self, param_function: Callable, param_dict: dict | None = None) -> None:
        """Define the :term:`Parameter Function` for the :term:`phase-shift
        <phase-shift unitary>` and :term:`mixing <mixing unitary>` unitaries.

        Parameters
        ----------
        param_function : Callable
            a :term:`Parameter Function`
        param_dict : FunctionDict
            :term:`FunctionDict` for ``param_function``
        """
        self.param_function = param_function
        self.param_dict = param_dict

    @scope("world")
    def setup(self) -> None:
        """Configure the QWOA unitaries and prepare the ansatz for execution."""
        if not self.setup_called:

            if self.observable_function is None:
                raise RuntimeError(
                    "Rank {}: Solution qualities not defined.".format(
                        self.MPI_COMM_WORLD.Get_rank()
                    )
                )

            if self.param_function is None:
                from ...param.rand import uniform

                self.set_params(uniform)

            phase_unitary = diagonal.Unitary(
                diagonal.operator.observables,
                parameter_function=self.param_function,
                param_dict=self.param_dict,
            )

            mixer_unitary = circulant.Unitary(
                circulant.operator.complete,
                parameter_function=self.param_function,
                param_dict=self.param_dict,
            )

            self.set_unitaries([phase_unitary, mixer_unitary])

        super().setup()
