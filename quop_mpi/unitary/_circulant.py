import numpy as np
from mpi4py import MPI
import numpy as np
from mpi4py import MPI
import quop_mpi.fqwoa_mpi as fqwoa_mpi

I = np.complex(0, 1)

class unitary(object):

    def __init__(
            self,
            system_size,
            COMM):

        self.system_size = system_size
        self.COMM = COMM

        self.rank = self.COMM.Get_rank()

        self.initial_state = None
        self.final_state = None
        self.variational_operator_function = None
        self.variational_operator_function_call = None
        self.planned = False

    def call_graph_array(self, function, variational_parameters = None, **kwargs):

        if not variational_parameters is None:
            self.graph_array = function(
                    N,
                    variational_parameters
                    **kwargs)
        else:
            self.graph_array = function(
                    N,
                    **kwargs)

    def gen_lambdas(self):

        self.lambdas = fqwoa_mpi.graph_eigenvalues(
                self.graph_array,
                self.local_o,
                self.local_o_offset)

    def _fftw_plan(self):

        # self.final state must be defined 
        self.initial_state = np.empty(self.final_state.shape, np.complex128)

        self.dummy_lambdas = np.empty(1, dtype = np.float64)

        """
        Calls FFTW subroutines which set up the ancillary data structures needed to
        efficiently perform 1D parallel Fourier and inverse Fourier transforms.
        """

        self.final_state = np.empty(self.alloc_local, np.complex128)
        self.initial_state = self.final_state

        fqwoa_mpi.evolve_circulant(
                self.system_size,
                self.local_i,
                0,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.COMM.py2f(),
                1)

    def copy_plan(self, _evolve_circulant):

        self.alloc_local = _evolve_circulant.alloc_local
        self.local_i = _evolve_circulant.local_i
        self.local_i_offset = _evolve_circulant.local_i_offset
        self.local_o = _evolve_circulant.local_o
        self.local_o_offset = _evolve_circulant.local_o_offset

        if not self.planned:
            self._fftw_plan()

    def plan(self):

        if not self.planned:

            local_sizes = fqwoa_mpi.mpi_local_size(self.system_size, self.COMM.py2f())

            self.alloc_local = local_sizes[0]
            self.local_i = local_sizes[1]
            self.local_i_offset = local_sizes[2]
            self.local_o = local_sizes[3]
            self.local_o_offset = local_sizes[4]

            self.final_state = np.empty(self.alloc_local, np.complex128)

            self._fftw_plan()

    def update(self):
        self._gen_lambdas()

    def propagate(self, t):

        # Class variables initial_state and final_state
        # need to be assigned directly. Expected size
        # is self.local_alloc.

        fqwoa_mpi.evolve_circulant(
                self.system_size,
                self.local_i,
                np.array([t, np.float64]),
                self.lambdas,
                self.initial_state,
                self.final_state,
                self.COMM.py2f(),
                0)

    def reset(self):
        pass

    def destroy(self):

        fqwoa_mpi.evolve_circulant(
                self.system_size,
                self.local_i,
                0,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.COMM.py2f(),
                -1)
