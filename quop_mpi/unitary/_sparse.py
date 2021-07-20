import numpy as np
from mpi4py import MPI
from scipy import sparse
import quop_mpi.fMPI as fMPI

I = np.complex(0, 1)

class unitary(object):

    def __init__(
            self,
            system_size,
            COMM)

        self.system_size = system_size
        self.COMM = COMM
        self.rank = self.COMM.Get_rank()
        self.precision = precision

        self.rank = self.COMM.Get_rank()

        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None

        self.partition_table = None
        self.initial_state = None
        self.final_state = None
        self.variational_operator_function = None
        self.variational_operator_function_call = None

        self.planned = False

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

