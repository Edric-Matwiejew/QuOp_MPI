"""Implements the :ref:`QOWE <QOWE>` :term:`mixing unitary` using momentum-space
propagation via FFT.
"""
from importlib import import_module
import numpy as np
from ... import config
from ...Unitary import Unitary
from ...__lib.propagator import propagator


class unitary(Unitary):
    """Implements the :ref:`QOWE <QOWE>` :term:`mixing unitary`.

    This propagator uses Fourier transforms to evolve the quantum state
    in momentum space, applying kinetic energy evolution.

    .. warning::

        ``unitary`` instances of type ``'momentum`` require that the ``size`` of
        the MPI communicator associated with :class:`quop_mpi.Ansatz` class be
        a factor of the first grid dimension (``Ns[0] % size == 0``). 

    **Inheritance Diagram:**

        .. graphviz::

            digraph "sphinx-ext-graphviz" {
                rankdir="LR"; node [fontsize="10"];
                Unitary[label="quop_mpi.Unitary", shape="rectangle"];
                unitary[label="quop_mpi.propagator.momentum.unitary",
                shape="rectangle"];
    
                Unitary -> unitary;
            }

    See :class:`quop_mpi.Unitary`.

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
        passed to the initialisation method of :class:`quop_mpi.Unitary`
    """

    def __init__(
        self,
        Ns: list[int],
        minsq: list[float],
        minsk: list[float],
        deltasq: list[float],
        deltask: list[float],
        *args,
        **kwargs
    ):
        self.Ns = np.array(Ns, dtype=np.int32)
        self.minsq = np.array(minsq, dtype=np.float64)
        self.minsk = np.array(minsk, dtype=np.float64)
        self.deltasq = np.array(deltasq, dtype=np.float64)
        self.deltask = np.array(deltask, dtype=np.float64)

        super().__init__(*args, **kwargs)

        self.unitary_type = "momentum"
        self.context = None
        self.comm_size_constraints = [np.array(Ns, dtype=np.int32)]
        self.planner = True

    def assign_backend(self, backend):
        """Assign the Fortran backend for momentum propagation."""
        self.propagator_module = backend.momentum_propagator
        self.propagators = [propagator(self.propagator_module.momentum_propagator_wrapper)]

    def plan(self, system_size, MPI_COMM):
        """Compute local partition size for this rank.
        
        Parameters
        ----------
        system_size : int
            Total size of the system state
        MPI_COMM : Intracomm
            MPI communicator
            
        Returns
        -------
        tuple[int, int]
            (local_i, alloc_local) - local partition size and allocation size
        """
        size = MPI_COMM.Get_size()
        rank = MPI_COMM.Get_rank()

        local_i = int(system_size // size + np.ceil((system_size % size) // (rank + 1) / size))

        return local_i, local_i

    def copy_plan(self, ex_unitary):
        """Copy planning information from another unitary."""
        pass

    def gen_operator(self, *args):
        """Generate the momentum-space operator.
        
        Sets up the FFTW plans and computes the phase factors and
        momentum-space eigenvalues needed for propagation.
        """
        self.propagators[0].plan(self.context)
        super().gen_operator(*args)
        
        # Pass grid parameters to the Fortran propagator
        # The operator function returns eigenvalues, but the momentum propagator
        # computes its own based on grid parameters
        self.propagators[0].gen_operator([
            self.Ns,
            self.minsq,
            self.minsk,
            self.deltasq,
            self.deltask
        ])

    def propagate(self, t):
        """Apply momentum-space evolution.
        
        Parameters
        ----------
        t : ndarray
            Evolution times for each dimension
        """
        self.propagators[0].propagate(t)

    def destroy(self):
        """Clean up FFTW plans and deallocate memory."""
        self.propagators[0].destroy()
