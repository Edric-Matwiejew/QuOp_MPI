from importlib import import_module
import numpy as np
from ...Unitary import Unitary


class unitary(Unitary):
    """Implements a mixing unitary with a circulant maxtrix operator exponent.

    See :class:`Unitary` for more information.
    """

    def __init__(
        self,
        operator_function,
        operator_n_params=0,
        operator_kwargs=None,
        parameter_function=None,
        parameter_kwargs=None,
    ):

        super().__init__(
            operator_function,
            operator_n_params,
            operator_kwargs,
            parameter_function,
            parameter_kwargs,
        )

        self.fqwoa_mpi = import_module("quop_mpi.__lib.fqwoa_mpi")
        self.evolve_circulant = self.fqwoa_mpi.evolve_circulant

        self.unitary_type = "circulant"
        self.planner = True
        self.planned = False

        self.dummy_eigs = np.empty(1, dtype=np.float64)

    def __fftw_plan(self):

        """Calls FFTW subroutines which set up the ancillary data structures
        needed to efficiently perform 1D parallel Fourier and inverse Fourier
        transforms."""

        self.initial_state = self.final_state

        self.evolve_circulant(
            self.system_size,
            self.local_i,
            0,
            self.dummy_eigs,
            self.initial_state,
            self.final_state,
            self.MPI_COMM.py2f(),
            1,
        )

        self.planned = True

    def plan(self, system_size, MPI_COMM):

        local_sizes = self.fqwoa_mpi.mpi_local_size(system_size, MPI_COMM.py2f())

        self.local_o = local_sizes[3]
        self.local_o_offset = local_sizes[4]

        return local_sizes[1], local_sizes[0]

    def copy_plan(self, ex_unitary):

        try:

            self.local_o = ex_unitary.local_o
            self.local_o_offset = ex_unitary.local_o_offset

        except:

            raise ValueError("Input unitary does not propagate using FFTW")

    def gen_operator(self, *args):

        if not self.planned:
            self.__fftw_plan()

        super().gen_operator(*args)

    def propagate(self, x):

        self.evolve_circulant(
            self.system_size,
            self.local_i,
            np.abs(x, dtype=np.float64),
            self.operator,
            self.initial_state,
            self.final_state,
            self.MPI_COMM.py2f(),
            0,
        )

    def destroy(self):

        if self.planned:

            self.evolve_circulant(
                self.system_size,
                self.local_i,
                0,
                self.dummy_eigs,
                self.initial_state,
                self.final_state,
                self.MPI_COMM.py2f(),
                -1,
            )

            self.planned = False
