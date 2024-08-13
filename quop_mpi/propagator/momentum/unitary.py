import numpy as np
from ...Unitary import Unitary
#from ...__lib import fCQAOA

class unitary(Unitary):
    """Implements the :ref:`QOWE <QOWE>` :term:`mixing unitary`.

    .. warning::

        ``unitary`` instances of type ``'momentum`` require that the ``size`` of
        the MPI communicator assocaited with :class:`quop_mpi.Ansatz` class be
        a factor of the first grid dimension (``Ns[0] % size == 0``). 

    **Inhertance Diagram:**

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
        the number of grid points in each dimension of the Cartesian grid in position and momentum space
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

        super().__init__(
            *args,
            **kwargs
        )

        # check Ns against N
        self.evolve = fCQAOA.fcqaoa.evolve_ft
        
        # should be system_size
        self.Ns = Ns
        self.minsq = minsq
        self.minsk = minsk
        self.deltasq = deltasq
        self.deltask = deltask
        
        self.unitary_type = "continuous"
        self.planner = True
        self.planned = False
        self.N = 1

        for dim in Ns:
            self.N *= dim

        self.dummy_mixer = np.empty((np.max(self.Ns), len(self.Ns)), dtype=np.float64)
        self.local_n0 = None
        self.local_n0_offset = None
        self.strides = None
        self.pk = None
        self.pq = None

    def __fftw_plan(self):

        """Calls FFTW subroutines which set up the ancillary data structures
        needed to efficiently perform 1D parallel Fourier and inverse Fourier
        transforms."""
       
        self.evolve(
                self.N,
                self.Ns,
                self.local_i_offset,
                self.local_n0,
                self.strides,
                np.array([0 for i in range(len(self.Ns))]),
                self.dummy_mixer,
                self.pk,
                self.pq,
                self.initial_state,
                self.MPI_COMM.py2f(), 
                1)
    
        self.planned = True

    def plan(self, system_size, MPI_COMM):

        self.MPI_COMM = MPI_COMM
        part = fCQAOA.fcqaoa.plan_partition(self.Ns, self.MPI_COMM.py2f())
       
        self.alloc_local = part[0]
        self.local_i = part[1]
        self.local_i_offset = part[2]
        self.local_n0 = part[3] 
        self.local_n0_offset = part[4]
        self.strides = part[5]

        return self.local_i, self.alloc_local

    def copy_plan(self, ex_unitary):

        # CHECK FOR CORRECTNESS
        try:

            self.local_n0 = ex_unitary.local_n0
            self.local_n0_offset = ex_unitary.local_n0_offset
            self.strides = ex_unitary.strides

        except:

            raise ValueError("Input unitary does not propagate using FFTW")

    def gen_operator(self, *args):

        if not self.planned:
            self.__fftw_plan()

        def phase_k(x):
            return np.exp(-1.0j*np.sum(x*self.minsq))
        
        def phase_q(x):
            return np.exp(1.0j*np.sum(x*self.minsk))

        self.pk = np.empty(shape = [self.local_i], dtype = np.complex128)
        self.pq = np.empty(shape = [self.local_i], dtype = np.complex128)
        
        fCQAOA.fcqaoa.dist_vector(
                phase_k,
                self.Ns,
                self.strides,
                self.deltask,
                self.minsk,
                self.local_i_offset,
                self.pk)
        
        fCQAOA.fcqaoa.dist_vector(
                phase_q,
                self.Ns,
                self.strides,
                self.deltasq,
                self.minsq,
                self.local_i_offset,
                self.pq)

        super().gen_operator(*args)

    def propagate(self, x):

        self.evolve(
                self.N,
                self.Ns,
                self.local_i_offset,
                self.local_n0,
                self.strides,
                np.abs(x),
                self.operator,
                self.pk,
                self.pq,
                self.initial_state,
                self.MPI_COMM.py2f(), 
                0)

        # check efficiency
        self.final_state[:] = self.initial_state

    def destroy(self):

        if self.planned:

            self.evolve(
                    self.N,
                    self.Ns,
                    self.local_i_offset,
                    self.local_n0,
                    self.strides,
                    np.array([0 for i in range(len(self.Ns))]),
                    self.operator,
                    self.pk,
                    self.pq,
                    self.initial_state,
                    self.MPI_COMM.py2f(), 
                    -1)
    
            self.planned = False
