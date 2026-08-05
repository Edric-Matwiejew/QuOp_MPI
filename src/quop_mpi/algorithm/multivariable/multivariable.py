"""Multivariable quantum optimisation algorithms (QMOA, QOWE)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

from ...ansatz import Ansatz
from ...param.rand import uniform
from ...propagator import composite, diagonal, momentum
from ...state import position_grid

if TYPE_CHECKING:
    from typing import Callable

    Intracomm = MPI.Intracomm


class Multivariable(Ansatz):
    """Base class for simulation of the :ref:`QMOA <QMOA>` and :ref:`QOWE`
    algorithm."""

    def set_params(self, param_function: Callable, param_dict: dict | None = None) -> None:
        """Define the :term:`Parameter Function` for the
        :term:`phase-shift <phase-shift unitary>` and
        :term:`mixing <mixing unitary>` unitaries.

        Parameters
        ----------
        param_function : Callable
            a :term:`Parameter Function`
        param_dict : FunctionDict
            :term:`FunctionDict` for :literal:`param_function`
        """
        self.UQ.param_function = param_function
        self.UQ.param_dict = param_dict
        self.UW.param_function = param_function
        self.UW.param_dict = param_dict

    def set_independent_t(self, independent: bool) -> None:
        """Specify simulation with or without independent :term:`unitary
        parameters <unitary parameter>` (walk times) over each coordinate
        dimension.

        Parameters
        ----------
        independent : bool
            simulate a unique walk time in each coordinate dimension if ``True``,
            all walk times share the same value if ``False``
        """
        if independent:
            self.UW_n_params = len(self.Ns)
            self.UW.unitary_n_params = len(self.Ns)
        else:
            self.UW_n_params = 1
            self.UW.unitary_n_params = 1

        # Recalculate n_params (operator_n_params + unitary_n_params)
        self.UW.n_params = self.UW.operator_n_params + self.UW.unitary_n_params

        self.set_unitaries([self.UQ, self.UW])

    def set_qualities(self, function: Callable, operator_dict: dict | None = None) -> None:
        """Define the :term:`observables` and :term:`phase-shift unitary` :term:`operator`

        Parameters
        ----------
        function : Callable
            an :term:`Operator Function`
        operator_dict : FunctionDict, optional
            :term:`FunctionDict` for ``function``
        """
        self.set_observables(function, operator_dict)

    def grid_point_from_index(self, index: int) -> np.ndarray[np.float64]:
        """Retrieve the corresponding coordinate point from a global index of the
        :term:`system state`.

        Parameters
        ----------
        index : int
            global index of the system state

        Returns
        -------
        ndarray[float64]
            a 1-D real array containing a grid point in Cartesian coordinates
        """

        if self.COMM_OPT.Get_rank() == 0:

            inds = self.unitaries[1].fCQAOA.continuous.get_index(
                index + 1, self.Ns, self.UW.strides
            )
            grid_points = (inds - 1) * self.deltas
            grid_points += self.mins
            return grid_points


class QMOA(Multivariable):
    """Simulate the :ref:`QMOA <QMOA>`.

    A :term:`QVA` for the optimisation of continuous multivariable
    functions.

    Parameters
    ----------
    Ns : list[int]
        the number of grid points in each each coordinate dimension
    MPI_communicator : Intracomm, optional
        MPI Intracomm, by default MPI.COMM_WORLD
    """

    def __init__(self, Ns: list[int], MPI_COMM: Intracomm = MPI.COMM_WORLD) -> None:  # noqa: N803
        """Initialise a QMOA instance.

        Parameters
        ----------
        Ns : list[int]
            Number of grid points per coordinate dimension (as powers of 2).
        MPI_COMM : Intracomm, optional
            MPI communicator, by default ``MPI.COMM_WORLD``.
        """
        self.Ns = [2**N for N in Ns]

        system_size = 1
        for N in self.Ns:  # noqa: N806
            system_size *= N

        super().__init__(system_size, MPI_COMM)

        self.continuous_function = None  # must be defined using set_qualities
        self.graphs = Ns  # complete graphs by default
        self.UW_n_params = len(Ns)

        self.UQ = diagonal.Unitary(
            diagonal.operator.observables,
            parameter_function=uniform,
        )

        self.UQ.Ns = self.Ns

        self.UW = composite.Unitary(
            self.Ns,
            composite.operator.ith,
            operator_dict={
                "args": [],
                "kwargs": {"Cs": self.Ns},
            },
            parameter_function=uniform,
            unitary_n_params=self.UW_n_params,
        )

        self.set_unitaries([self.UQ, self.UW])

    def set_mixer(self, Cs: list[int]) -> None:  # noqa: N803
        """Set the circulant :term:`mixing unitary` :term:`operator` in each
        coordinate dimension.

        See Also
        --------
        :func:`quop_mpi.propagator.composite.operator.ith`

        Parameters
        ----------
        Cs : list[int]
            specifies the "i-th" symmetric circulant matrix with edges weights
            ``1``,  ``Cs[j] == 1`` cycle graph,  ``Cs[j] > system_size // 2``
            complete graph
        """
        self.UW.operator_dict = {"args": [], "kwargs": {"Cs": Cs}}


class QOWE(Multivariable):
    """Simulate the Quantum Walk Optimisation by Eigenvalues (QOWE) algorithm.

    A :term:`QVA` for optimisation of continuous multivariable functions
    using momentum-space mixing unitaries.
    """

    def __init__(
        self,
        Ns: list[int],  # noqa: N803
        deltas: list[float],
        mins: list[float],
        MPI_COMM: Intracomm = MPI.COMM_WORLD,  # noqa: N803
    ) -> None:
        """Simulate the :ref:`QMOA <QMOA>`.

        A :term:`QVA` for the optimisation of continuous multivariable
        functions.

        Parameters
        ----------
        Ns : list[int]
            the number of grid points in each each coordinate dimension
        deltas : list[float]
            the step size of each Cartesian coordinate in position space
        mins : list[float]
            the minimum value of each Cartesian coordinate in position space
        MPI_communicator : Intracomm, optional
            MPI Intracomm, by default MPI.COMM_WORLD
        """
        self.Ns = [2**N for N in Ns]
        self.deltas = deltas
        self.mins = mins

        system_size = 1
        for N in self.Ns:  # noqa: N806
            system_size *= N

        super().__init__(system_size, MPI_COMM)

        self.continuous_function = None  # must be defined using set_qualities
        self.graphs = Ns  # complete graphs by default
        self.UW_n_params = len(Ns)

        self.deltask = np.array(
            [2 * np.pi / (n * delta) for (delta, n) in zip(self.deltas, self.Ns, strict=True)],
            dtype=np.float64,
        )

        self.minsk = np.array(
            [-(n / 2) * delta for (delta, n) in zip(self.deltask, self.Ns, strict=True)],
            dtype=np.float64,
        )

        self.UQ = diagonal.Unitary(
            diagonal.operator.observables,
            parameter_function=uniform,
        )

        self.UQ.Ns = self.Ns
        self.UQ.mins = mins
        self.UQ.deltas = self.deltas

        self.UW = momentum.Unitary(
            self.Ns,
            self.mins,
            self.minsk,
            self.deltas,
            self.deltask,
            parameter_function=uniform,
            unitary_n_params=len(self.Ns),
        )

        self.set_unitaries([self.UQ, self.UW])

        self.set_initial_state(
            position_grid, {"args": [None]}  # function=None uses default Gaussian
        )
