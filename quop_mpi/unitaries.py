from importlib import import_module
import numpy as np
from mpi4py import MPI
from quop_mpi.__utils.__interface import interface

I = np.complex(0, 1)

class __unitary(object):

    def __init__(
            self,
            operator_function,
            operator_n_params = 0,
            operator_kwargs = {},
            parameter_function = None,
            parameter_kwargs = {},
            unitary_n_params = 1):

        self.operator_function = operator_function
        self.operator_n_params = operator_n_params
        self.operator_kwargs = operator_kwargs
        self.parameter_function = parameter_function
        self.parameter_kwargs = parameter_kwargs
        self.unitary_n_params = unitary_n_params

        self.unitary_type = None
        self.planner = False

        self.operator_parameters = [
                    'operator_function',
                    'system_size',
                    'local_alloc',
                    'local_i',
                    'local_i_offset',
                    'local_o',
                    'local_o_offset',
                    'partition_table',
                    'lb',
                    'ub',
                    'variational_parameters',
                    'seed',
                    'MPI_COMM',
                ]

        self.parameter_function_parameters = [
                'system_size',
                'operator',
                'n_params',
                'seed',
                'MPI_COMM'
                ]

        self.system_size = None
        self.operator = None
        self.n_params = 0
        self.seed = 0
        self.initial_parameters = None
        self.initial_state = None
        self.final_state = None
        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None
        self.partition_table = None
        self.lb = None
        self.ub = None
        self.variational_parameters = None
        self.planned = False

        self.n_params += operator_n_params + unitary_n_params


    def parse_operator_function(self):

        self.parsed_operator_function = interface(
                self,
                self.operator_function,
                self.operator_parameters,
                "operator",
                self.MPI_COMM,
                )

    def parse_parameter_function(self):

        if self.parameter_function is None:
            from quop_mpi.params import uniform
            self.parameter_function = uniform

        self.parsed_parameter_function = interface(
                self,
                self.parameter_function,
                self.parameter_function_parameters,
                "initial parameters",
                self.MPI_COMM
                )

    def plan(self, system_size, MPI_COMM):

        self.system_size = system_size
        self.MPI_COMM = MPI_COMM

        self.rank = self.MPI_COMM.Get_rank()
        self.size = self.MPI_COMM.Get_size()

    def copy_plan(self, unitary):

        self.system_size = unitary.system_size
        self.MPI_COMM = unitary.MPI_COMM

        self.rank = self.MPI_COMM.Get_rank()
        self.size = self.MPI_COMM.Get_size()

    def get_initial_params(self):
        self.parsed_parameter_function.update_parameters()
        return self.parsed_parameter_function.call(**self.parameter_kwargs)

    def gen_operator(self):
        if self.variational_parameters is not None:
            self.parsed_operator_function.update_parameters()
        self.operator = self.parsed_operator_function.call(**self.operator_kwargs)

    def propagate(self, x):
        raise NotImplementedError("Rank {}: Method 'propagate' not implemented by child class".format(self.rank))

    def reset(self):
        pass

    def destroy(self):
        pass

class __quop_partitioned(__unitary):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    #def __init__(
    #        self,
    #        operator_function,
    #        operator_n_params = 0,
    #        operator_kwargs = {},
    #        parameter_function = None,
    #        parameter_kwargs = {}
    #        ):

    #    super().__init__(
    #            operator_function,
    #            operator_n_params,
    #            operator_kwargs,
    #            parameter_function,
    #            parameter_kwargs,
    #            )

    def copy_plan(self, unitary):

        self.system_size = unitary.system_size
        self.MPI_COMM = unitary.MPI_COMM
        self.local_i = unitary.local_i
        self.local_i_offset = unitary.local_i_offset
        self.alloc_local = unitary.alloc_local
        self.final_state = unitary.final_state
        self.initial_state = unitary.initial_state

        self.rank = self.MPI_COMM.Get_rank()

        self.partition_table = np.zeros(self.MPI_COMM.Get_size() + 1, dtype = np.int32)
        self.partition_table[self.rank + 1] = self.local_i

        self.partition_table[1:] = self.MPI_COMM.allgather(self.partition_table[self.rank + 1])

        self.partition_table[0] = 1

        for i in range(1, self.MPI_COMM.Get_size() + 1):
            self.partition_table[i] += self.partition_table[i - 1]

        self.lb = self.partition_table[self.rank] - 1
        self.ub = self.partition_table[self.rank + 1] - 1

        self.parse_operator_function()
        self.parse_parameter_function()


    def plan(self, system_size, MPI_COMM):

        super().plan(system_size, MPI_COMM)

        self.partition_table = np.zeros(self.size + 1, dtype = np.int32)
        for i in range(self.size + 1):
            self.partition_table[i] = i * self.system_size / self.size + 1

        remainder = self.system_size - self.partition_table[self.size]

        for i in range(remainder):
            self.partition_table[self.size - i % self.size : self.size + 1] += 1

        self.lb = self.partition_table[self.rank] - 1
        self.ub = self.partition_table[self.rank + 1] - 1

        self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
        self.local_i_offset = self.partition_table[self.rank] - 1

        self.alloc_local = self.local_i

        self.final_state = np.empty(self.alloc_local, np.complex128)
        self.initial_state = np.empty(self.alloc_local, dtype = np.complex128)

        self.parse_operator_function()
        self.parse_parameter_function()

class circulant(__unitary):


    def __init__(
            self,
            operator_function,
            operator_n_params = 0,
            operator_kwargs = {},
            parameter_function = None,
            parameter_kwargs = {}
            ):

        super().__init__(
                operator_function,
                operator_n_params,
                operator_kwargs,
                parameter_function,
                parameter_kwargs,
                )

        self.fqwoa_mpi = import_module('quop_mpi.__lib.fqwoa_mpi')
        self.evolve_circulant = self.fqwoa_mpi.evolve_circulant

        self.unitary_type = "circulant"
        self.planner = True
        self.planned = False

        self.dummy_eigs = np.empty(1, dtype = np.float64)

    def __fftw_plan(self):

        """
        Calls FFTW subroutines which set up the ancillary data structures needed to
        efficiently perform 1D parallel Fourier and inverse Fourier transforms.
        """

        self.initial_state = self.final_state

        self.evolve_circulant(
                self.system_size,
                self.local_i,
                0,
                self.dummy_eigs,
                self.initial_state,
                self.final_state,
                self.MPI_COMM.py2f(),
                1)

    def __gen_partition_table(self):

       self.partition_table = np.zeros(self.MPI_COMM.Get_size() + 1, dtype = np.int32)
       self.partition_table[self.rank + 1] = self.local_i

       self.partition_table[1:] = self.MPI_COMM.allgather(self.partition_table[self.rank + 1])

       self.partition_table[0] = 1

       for i in range(1, self.MPI_COMM.Get_size() + 1):
           self.partition_table[i] += self.partition_table[i - 1]

    def copy_plan(self, unitary):

        super().copy_plan(unitary)

        try:

            self.local_o = unitary.local_o
            self.local_o_offset = unitary.local_o_offset

        except:

            raise ValueError("Rank {}: Input unitary does not propagate using FFTW".format(self.rank))

        self.alloc_local = unitary.alloc_local
        self.local_i = unitary.local_i
        self.local_i_offset = unitary.local_i_offset
        self.final_state = unitary.final_state
        self.initial_state = unitary.initial_state
        self.__gen_partition_table()

        self.parse_operator_function()
        self.parse_parameter_function()

        if not self.planned:
            self.__fftw_plan()

    def plan(self, system_size, MPI_COMM):

        super().plan(system_size, MPI_COMM)

        local_sizes = self.fqwoa_mpi.mpi_local_size(self.system_size, self.MPI_COMM.py2f())

        self.alloc_local = local_sizes[0]
        self.local_i = local_sizes[1]
        self.local_i_offset = local_sizes[2]
        self.local_o = local_sizes[3]
        self.local_o_offset = local_sizes[4]

        self.final_state = np.empty(self.alloc_local, np.complex128)
        self.initial_state = np.empty(self.alloc_local, dtype = np.complex128)

        self.__gen_partition_table()
        self.__fftw_plan()

        self.parse_operator_function()
        self.parse_parameter_function()

        self.planned = True

    def propagate(self, x):

        # Class variables initial_state and final_state
        # need to be assigned directly. Expected size
        # is self.local_alloc.

        self.evolve_circulant(
                self.system_size,
                self.local_i,
                np.abs(x, dtype = np.float64),
                self.operator,
                self.initial_state,
                self.final_state,
                self.MPI_COMM.py2f(),
                0)

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
                    -1)

            self.planned = False

class diagonal(__quop_partitioned):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.unitary_type="diagonal"

        if self.unitary_n_params > 1:
            self.propagate = self.evolve_group
        else:
            self.propagate = self.evolve_single

    def evolve_single(self, x):

        # Class variables initial_state and final_state
        # need to be assigned directly. Expected size
        # is self.local_i, but may be of size self.local_alloc
        # so slice these arrays just in case.

        self.final_state[:self.local_i] = np.exp(-I * x * self.operator[:self.local_i]) * self.initial_state[:self.local_i]

    def evolve_group(self, x):

        self.final_state = self.initial_state

        for operator, param in zip(self.operator, x):
            self.final_state[:self.local_i] = np.exp(-I * param * operator[:self.local_i]) * self.final_state[:self.local_i]

class sparse(__quop_partitioned):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.unitary_n_params > 1:
            self.propagate = self.evolve_group
        else:
            self.propagate = self.evolve_single

        self.fMPI = import_module('quop_mpi.__lib.fMPI')

        self.precision = "dp"

        self.W_row_starts = None
        self.W_col_indexes = None
        self.W_values = None

        self.one_norms = None
        self.num_norms = None
        self.W_num_rec_inds = None
        self.W_rec_disps = None
        self.W_num_send_inds = None
        self.W_send_disps = None
        self.W_local_col_inds = None
        self.W_rhs_send_inds = None

    def gen_operator(self, *args):

        super().gen_operator(*args)

        self.W_row_starts, self.W_col_indexes, self.W_values = self.operator

        for i in range(len(self.W_values)):
            self.W_values[i][:] = -I*self.W_values[i]

        self.W_num_rec_inds = []
        self.W_rec_disps = []
        self.W_num_send_inds = []
        self.W_send_disps = []
        self.W_local_col_inds = []
        self.W_rhs_send_inds = []
        self.one_norms = []
        self.num_norms = []

        for w_row_starts, w_col_indexes, w_values in zip(self.W_row_starts, self.W_col_indexes, self.W_values):

            w_num_rec_inds, w_rec_disps, w_num_send_inds, w_send_disps = self.fMPI.rec_a(
                    self.system_size,
                    w_row_starts,
                    w_col_indexes,
                    self.partition_table,
                    self.MPI_COMM.py2f())

            self.W_num_rec_inds.append(w_num_rec_inds)
            self.W_rec_disps.append(w_rec_disps)
            self.W_num_send_inds.append(w_num_send_inds)
            self.W_send_disps.append(w_send_disps)

            w_local_col_inds, w_rhs_send_inds = self.fMPI.rec_b(
                    self.system_size,
                    np.sum(self.W_num_send_inds),
                    w_row_starts,
                    w_col_indexes,
                    w_num_rec_inds,
                    w_rec_disps,
                    w_num_send_inds,
                    w_send_disps,
                    self.partition_table,
                    self.MPI_COMM.py2f())

            self.W_local_col_inds.append(w_local_col_inds)
            self.W_rhs_send_inds.append(w_rhs_send_inds)

            one_norms, num_norms = self.fMPI.one_norm_series(
                    self.system_size,
                    w_row_starts,
                    w_col_indexes,
                    w_values,
                    w_num_rec_inds,
                    w_rec_disps,
                    w_num_send_inds,
                    w_send_disps,
                    w_local_col_inds,
                    w_rhs_send_inds,
                    self.partition_table,
                    self.MPI_COMM.py2f())

            self.one_norms.append(one_norms)
            self.num_norms.append(num_norms)

    def evolve_single(self, x):
        """
        Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """

        for i in range(len(self.W_row_starts)):

            self.final_state[:self.local_i] = self.fMPI.step(
                    self.system_size,
                    self.local_i,
                    self.W_row_starts[i],
                    self.W_col_indexes[i],
                    self.W_values[i],
                    self.W_num_rec_inds[i],
                    self.W_rec_disps[i],
                    self.W_num_send_inds[i],
                    self.W_send_disps[i],
                    self.W_local_col_inds[i],
                    self.W_rhs_send_inds[i],
                    np.abs(x),
                    self.initial_state[:self.local_i],
                    self.partition_table,
                    self.num_norms[i],
                    self.one_norms[i],
                    self.MPI_COMM.py2f(),
                    self.precision)

            self.initial_state[:] = self.final_state

    def evolve_group(self, x):
        """
        Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """

        for i in range(len(self.W_row_starts)):

            self.final_state[:self.local_i] = self.fMPI.step(
                    self.system_size,
                    self.local_i,
                    self.W_row_starts[i],
                    self.W_col_indexes[i],
                    self.W_values[i],
                    self.W_num_rec_inds[i],
                    self.W_rec_disps[i],
                    self.W_num_send_inds[i],
                    self.W_send_disps[i],
                    self.W_local_col_inds[i],
                    self.W_rhs_send_inds[i],
                    np.abs(x[i]),
                    self.initial_state[:self.local_i],
                    self.partition_table,
                    self.num_norms[i],
                    self.one_norms[i],
                    self.MPI_COMM.py2f(),
                    self.precision)

            self.initial_state[:] = self.final_state
