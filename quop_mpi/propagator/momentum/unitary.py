from importlib import import_module
import numpy as np
from ...Unitary import Unitary


class unitary(Unitary):
    """Implements a mixing unitary with a circulant matrix operator exponent.

    See :class:`Unitary` for more information.
    """

    def __init__(
        self,
        Ns,
        minsq,
        minsk,
        deltasq,
        deltask,
        operator_function,
        operator_n_params=0,
        operator_kwargs=None,
        parameter_function=None,
        parameter_kwargs=None,
        unitary_n_params = None,
    ):

        super().__init__(
            operator_function,
            operator_n_params,
            operator_kwargs,
            parameter_function,
            parameter_kwargs,
            unitary_n_params =  unitary_n_params,
        )

        # check Ns against N
        self.fCQAOA = import_module("quop_mpi.__lib.fCQAOA")
        self.evolve = self.fCQAOA.continuous.evolve_ft
        
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
        part = self.fCQAOA.continuous.plan_partition(self.Ns, self.MPI_COMM.py2f())
       
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
        
        self.fCQAOA.continuous.dist_vector(
                phase_k,
                self.Ns,
                self.strides,
                self.deltask,
                self.minsk,
                self.local_i_offset,
                self.pk)
        
        self.fCQAOA.continuous.dist_vector(
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
