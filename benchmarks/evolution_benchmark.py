from time import time
import sys
import resource
from os.path import exists
from importlib import import_module
from mpi4py import MPI
import numpy as np

simulation_time = float(sys.argv[1])
output_filepath = sys.argv[2]
test_program = import_module(sys.argv[3])

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()

if COMM.Get_rank() == 0:
    if not exists(output_filepath):
        log = open(output_filepath, 'a')
        log.write('comm_size,system_size,time,peak_memory\n')
        log.close()

elapsed = 0
last_time = 0
qubits = 1

while elapsed + last_time < simulation_time:

    qubits += 1

    start = time()

    system_size = 2**qubits

    final_state = test_program.function(system_size, COMM)

    finish = time()
    last_time = finish - start
    last_time = COMM.allreduce(last_time, op = MPI.MAX)
    elapsed += last_time

    peak_mem = COMM.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, MPI.SUM,0)

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
