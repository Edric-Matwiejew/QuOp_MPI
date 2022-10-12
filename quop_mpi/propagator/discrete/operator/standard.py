from importlib import import_module
from quop_mpi.__lib import fCQAOA
from mpi4py import MPI
import numpy as np

def grid(system_size, local_i, local_i_offset, Ns, strides, deltas, mins, MPI_COMM, function = None, scale = None):

    x = fCQAOA.continuous.gen_local_grid(   system_size,
                                            Ns,
                                            strides,
                                            deltas,
                                            mins,
                                            local_i_offset,
                                            local_i)
    f = []
    for point in x:
        f.append(function(point))

    f = np.array(f, dtype = np.float64)

    if scale is not None:
        f_max = MPI_COMM.allreduce(np.max(f), op = MPI.MAX) 
        f_min = MPI_COMM.allreduce(np.min(f), op = MPI.MIN) 

        f = scale*(f - f_min)/(f_max - f_min)

    return f


