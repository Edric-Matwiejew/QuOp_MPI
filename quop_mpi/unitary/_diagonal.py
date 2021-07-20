import numpy as np
from mpi4py import MPI
import quop_mpi.fMPI as fMPI

I = np.complex(0, 1)

class unitary(object):

    def __init__(
            self,
            system_size,
            COMM):

        self.system_size = system_size
        self.COMM = COMM

        self.rank = self.COMM.Get_rank()

        self.partition_table = None
        self.initial_state = None
        self.final_state = None
        self.variational_operator_function = None
        self.variational_operator_function_call = None

    def generate_partition_table(self):

        size = self.COMM.Get_size()

        self.partition_table = np.zeros(size + 1, dtype = np.int32)
        for i in range(size + 1):
            self.partition_table[i] = i * self.system_size / size + 1

        remainder = self.system_size - self.partition_table[size]

        for i in range(remainder):
            self.partition_table[size - i % size : size + 1] += 1

        self.lb = self.partition_table[self.rank] - 1
        self.ub = self.partition_table[self.rank + 1] - 1

        self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
        self.local_i_offset = self.partition_table[self.rank] - 1

        self.alloc_local = self.local_i

    def copy_plan(self, evolve_circulant):

       self.local_i = evolve_circulant.local_i
       self.local_i_offset = evolve_circulant.local_i_offset
       self.alloc_local = evolve_circulant.alloc_local

       self.partition_table = np.zeros(self.COMM.Get_size() + 1, dtype = np.int32)
       self.partition_table[self.rank + 1] = self.local_i

       self.partition_table[1:] = self.COMM.allgather(self.partition_table[self.rank + 1])

       self.partition_table[0] = 1

       for i in range(1, self.COMM.Get_size() + 1):
           self.partition_table[i] += self.partition_table[i - 1]
       self.lb = self.partition_table[self.rank] - 1
       self.ub = self.partition_table[self.rank + 1] - 1

    def call_mpi_diag(self, function, variational_parameters = None, **kwargs):

            if not variational_parameters is None:
                self.local_diag = function(
                        self.system_size,
                        self.local_i,
                        self.local_i_offset,
                        variational_parameters,
                        **kwargs)
            else:

                self.local_diag = function(
                        self.system_size,
                        self.local_i,
                        self.local_i_offset,
                        **kwargs)

    def call_local_diag(self, function, variational_parameters = None, **kwargs):

            if not variational_parameters is None:
                self.local_diag = function(
                        self.system_size,
                        variational_parameters,
                        **kwargs)[self.local_i_offset:self.local_i_offset + self.local_i]
            else:

                self.local_diag = function(
                        self.system_size,
                        **kwargs)[self.local_i_offset:self.local_i_offset + self.local_i]

    def plan(self):
        pass

    def update(self):
        pass

    def propagate(self, gamma):

        # Class variables initial_state and final_state
        # need to be assigned directly. Expected size
        # is self.local_i, but may be of size self.local_alloc
        # so slice these arrays just in case.


        self.final_state[:self.local_i] = np.multiply(np.exp(-I * gamma * self.local_diag), self.initial_state[:self.local_i])

    def reset(self):
        self.local_diag = None

    def destroy(self):
        self.reset()
