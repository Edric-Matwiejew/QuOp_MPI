"""Implements the :ref:`QOWE <QOWE>` :term:`mixing unitary` using momentum-space
propagation via FFT.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from ..._lib.propagator import Propagator
from ...unitary import UnitaryBase


class Unitary(UnitaryBase):
    """Implements the :ref:`QOWE <QOWE>` :term:`mixing unitary`.

    This propagator uses Fourier transforms to evolve the quantum state
    in momentum space, applying kinetic energy evolution.

    .. warning::

        ``unitary`` instances of type ``'momentum`` require that the ``size`` of
        the MPI communicator associated with :class:`quop_mpi.ansatz` class be
        a factor of the first grid dimension (``Ns[0] % size == 0``).

    **Inheritance Diagram:**

        .. graphviz::

            digraph "sphinx-ext-graphviz" {
                rankdir="LR"; node [fontsize="10"];
                Unitary[label="quop_mpi.unitary", shape="rectangle"];
                unitary[label="quop_mpi.propagator.momentum.unitary",
                shape="rectangle"];

                Unitary -> unitary;
            }

    See :class:`quop_mpi.unitary`.

    Attributes
    ----------
    unitary_type
        ``'momentum'``
    planner
        ``True``
    unitary_n_params
        ``len(Ns)``

    Parameters
    ----------
    Ns : list[int]
        the number of grid points in each dimension of the Cartesian grid
    minsq : list[float]
        the minimum of each Cartesian coordinate in position space
    minsk : list[float]
        the minimum of each Cartesian coordinate in momentum space
    deltasq : list[float]
        the step-size in each Cartesian coordinate in position space
    deltask : list[float]
        the step-size in each Cartesian coordinate in momentum space
    *args and **kwargs:
        passed to the initialisation method of :class:`quop_mpi.unitary`
    """

    def __init__(
        self,
        Ns: list[int],  # noqa: N803
        minsq: list[float],
        minsk: list[float],
        deltasq: list[float],
        deltask: list[float],
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialise the momentum unitary propagator.

        The momentum propagator does not accept an :term:`Operator Function`.
        Momentum-space eigenvalues are computed internally by the Fortran
        backend from the grid parameters.
        """
        self.Ns = np.array(Ns, dtype=np.int32)
        self.minsq = np.array(minsq, dtype=np.float64)
        self.minsk = np.array(minsk, dtype=np.float64)
        self.deltasq = np.array(deltasq, dtype=np.float64)
        self.deltask = np.array(deltask, dtype=np.float64)

        # No operator function — eigenvalues are computed by the Fortran backend.
        super().__init__(operator_function=lambda: None, **kwargs)

        self.unitary_type = "momentum"
        self.context = None
        self.comm_size_constraints = [np.array(Ns, dtype=np.int32)]
        self.planner = True

    def assign_backend(self, backend: ModuleType) -> None:
        """Assign the Fortran backend for momentum propagation."""
        self.propagator_module = backend.momentum_propagator
        self.propagators = [Propagator(self.propagator_module.momentum_propagator_wrapper)]

    def gen_operator(self) -> None:
        """Generate the momentum-space operator.

        Sets up the FFTW plans and passes grid parameters to the Fortran
        backend, which computes the phase factors and momentum-space
        eigenvalues internally.
        """
        self.propagators[0].plan(self.context)
        self.planned = True  # Mark as planned so destroy() is called during cleanup

        self.propagators[0].gen_operator(
            [self.Ns, self.minsq, self.minsk, self.deltasq, self.deltask]
        )

    def propagate(self, t: np.ndarray) -> None:
        """Apply momentum-space evolution.

        Parameters
        ----------
        t : ndarray
            Evolution times for each dimension
        """
        self.propagators[0].propagate(t)

    def destroy(self) -> None:
        """Clean up FFTW plans and deallocate memory."""
        self.propagators[0].destroy()
