from time import time
import sys
import resource
from os.path import exists
from importlib import import_module
from mpi4py import MPI
import numpy as np

def evolution(
        simulation_time,
        output_filepath,
        function):

    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    
    if COMM.Get_rank() == 0:
        if not exists(output_filepath):
            log = open(output_filepath, 'a')
            log.write('comm_size,system_size,time,peak_memory\n')
            log.close()
    
    elapsed = 0
    last_time = 0
    qubits = 10
    
    while elapsed + last_time < simulation_time:
    
        qubits += 1
    
        start = time()
    
        system_size = 2**qubits
    
        final_state = function(system_size, COMM)
    
        finish = time()
        last_time = finish - start
        last_time = COMM.allreduce(last_time, op = MPI.MAX)
        elapsed += last_time
    
        peak_mem = COMM.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, MPI.SUM,0)
    
        if final_state is None:
            final_state = 0

        norm = COMM.reduce(np.sum(np.abs(final_state)**2), root = 0, op = MPI.SUM)
    
        if COMM.Get_rank() == 0:
    
            peak_mem = peak_mem*(1024**(-2))
    
            log = open(output_filepath, 'a')
            log.write('{},{},{},{},{},{}\n'.format(
            COMM.Get_size(),
            qubits,
            system_size,
            norm,
            elapsed,
            peak_mem))
            log.close()

