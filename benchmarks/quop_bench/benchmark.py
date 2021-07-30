from time import time
import sys
import resource
from os.path import exists
from importlib import import_module
from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD

def evolution(
        simulation_time,
        output_filepath,
        function):

    rank = COMM.Get_rank()

    if COMM.Get_rank() == 0:
        if not exists(output_filepath):
            log = open(output_filepath, 'a')
            log.write('comm_size,qubits,system_size,norm,time,peak_memory\n')
            log.close()

    elapsed = 0
    last_time = 0
    qubits = 1

    while elapsed + last_time < simulation_time:

        qubits += 1

        start = time()

        system_size = 2**qubits

        COMM.barrier()

        local_i, final_state = function(system_size, COMM)

        finish = time()
        last_time = finish - start
        last_time = COMM.allreduce(last_time, op = MPI.MAX)
        elapsed += last_time

        peak_mem = COMM.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, MPI.SUM,0)

        if final_state is None:
            final_state = 0
        else:
            final_state = final_state[:local_i]

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

def execute(
        simulation_time,
        qubits,
        output_filepath,
        bench_log_name,
        quop_log_name,
        function
        ):

    rank = COMM.Get_rank()

    if COMM.Get_rank() == 0:
        bench_log = "{}/{}".format(output_filepath  bench_log_name)
        if not exists(bench_log):
            log = open(bench_log, 'a')
            log.write('comm_size,qubits,system_size,depth,time,peak_memory\n')
            log.close()

    elapsed = 0
    last_time = 0
    depth = 0

    quop_log = "{}/{}".format(output_filepath, quop_log)

    while elapsed + last_time < simulation_time:

        depth += 1

        start = time()

        system_size = 2**qubits

        function(system_size, depth, quop_log, COMM)
        finish = time()
        last_time = finish - start

        COMM.barrier()

        last_time = COMM.allreduce(last_time, op = MPI.MAX)
        elapsed += last_time

        peak_mem = COMM.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, MPI.SUM,0)

        if COMM.Get_rank() == 0:

            peak_mem = peak_mem*(1024**(-2))

            log = open(bench_log, 'a')
            log.write('{},{},{},{},{},{}\n'.format(
            COMM.Get_size(),
            qubits,
            system_size,
            depth,
            elapsed,
            peak_mem))
            log.close()


COMM.barrier()
