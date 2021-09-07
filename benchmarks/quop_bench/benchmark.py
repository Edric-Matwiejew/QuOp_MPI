import sys
sys.path.insert(0,'../')
from time import time
import sys
import resource
from os.path import exists
from copy import copy, deepcopy
from importlib import import_module
from mpi4py import MPI
import numpy as np
import h5py as h5

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

def execute_depth(
        simulation_time,
        qubits,
        output_filepath,
        bench_log_name,
        quop_log_name,
        function
        ):

    rank = COMM.Get_rank()

    if COMM.Get_rank() == 0:
        bench_log = "{}/{}".format(output_filepath,  bench_log_name)
        if not exists(bench_log):
            log = open(bench_log, 'a')
            log.write('comm_size,qubits,system_size,depth,time\n')
            log.close()

    elapsed = 0
    last_time = 0
    depth = 0

    quop_log = "{}/{}".format(output_filepath, quop_log_name)

    while elapsed + last_time < simulation_time:

        depth += 1

        start = time()

        system_size = 2**qubits

        
        COMM.barrier()

        function(system_size, depth, quop_log, COMM)
        finish = time()
        last_time = finish - start

        COMM.barrier()

        last_time = COMM.allreduce(last_time, op = MPI.MAX)
        elapsed += last_time

        if COMM.Get_rank() == 0:

            log = open(bench_log, 'a')
            log.write('{},{},{},{},{}\n'.format(
            COMM.Get_size(),
            qubits,
            system_size,
            depth,
            elapsed))
            log.close()

def execute(
        depth,
        qubits,
        output_filepath,
        bench_log_name,
        quop_log_name,
        function
        ):

    rank = COMM.Get_rank()

    if COMM.Get_rank() == 0:
        bench_log = "{}/{}".format(output_filepath,  bench_log_name)
        if not exists(bench_log):
            log = open(bench_log, 'a')
            log.write('comm_size,qubits,system_size,depth,time,peak_memory\n')
            log.close()

    quop_log = "{}/{}".format(output_filepath, quop_log_name)

    system_size = 2**qubits

    start = time()

    function(system_size, depth, quop_log, COMM)

    finish = time()

    time_log = COMM.allreduce(finish - start, op = MPI.MAX)

    peak_mem = COMM.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, MPI.SUM,0)

    if COMM.Get_rank() == 0:

        peak_mem = peak_mem*(1024**(-2))

        log = open(bench_log, 'a')
        log.write('{},{},{},{},{},{}\n'.format(
        COMM.Get_size(),
        qubits,
        system_size,
        depth,
        time_log,
        peak_mem))
        log.close()


def optimisers(
        min_qubits,
        max_qubits,
        min_depth,
        max_depth,
        algs,
        alg_names,
        qualities,
        backends,
        method_names,
        options,
        log_filename,
        objective_history_filename):

    def target(expectation, target_val = None):
        return np.abs(expectation - target_val)

    for alg, alg_name in zip(algs, alg_names):

        for n_qubits in range(min_qubits, max_qubits + 1):

            system_size = 2**n_qubits

            test_alg = alg(system_size)
            test_alg.set_parallel('jacobian')
            test_alg.verbose_objective = True

            test_alg.set_qualities(qualities)

            test_alg.record_objective = True

            thetas = []
            fmin = []

            for depth in range(min_depth, max_depth + 1):

                for seed in range(1, 6):

                    test_alg.seed = seed
                    test_alg.set_depth(depth)

                    theta = test_alg.gen_initial_params(depth)

                    thetas.append(deepcopy(theta))

                    fmins = []

                    for i, backend in enumerate(backends):

                        for option, method_name, in zip(options[i], method_names[i]):

                            test_alg.set_log(
                                    "{}/{}".format(log_filename, alg_name),
                                    'baseline_{}'.format(method_name),
                                    action = 'a')

                            test_alg.set_optimiser(
                                    backend,
                                    copy(option),
                                    ['fun','nfev','success', 'message'])


                            if not 'jac' in option:
                                test_alg.optimiser_args['jac'] = None

                            test_alg.execute(theta)

                            test_alg.print_optimiser_result()

                        if test_alg.COMM.Get_rank() == 0:
                            fmins.append(test_alg.result['fun'])

                            data = np.array([test_alg.total_n_evolutions, test_alg.objective_history], dtype = np.float64)
                            data_label =  '{}_{}_{}_{}_{}_{}'.format(method_name, alg_name, 'baseline', seed, n_qubits, depth)
                            np.save(objective_history_filename + '/' + data_label + '.npy', data)

						test_alg.COMM.barrier()
  
                    if test_alg.COMM.Get_rank() == 0:
                        fmin.append(np.min(fmins))

            fmin = test_alg.COMM.bcast(fmin, 0)

            for i, backend in enumerate(backends):

                for option, method_name, in zip(options[i], method_names[i]):

                    if test_alg.COMM.Get_rank() == 0:
                        print(method_name, flush = True)
                    
                    test_alg.set_log(
                            "{}/{}".format(log_filename, alg_name),
                            '{}'.format(method_name),
                            action = 'a')

                    test_alg.set_optimiser(
                            backend,
                            copy(option),
                            ['fun','nfev','success', 'message'])

                    if not 'jac' in option:
                        test_alg.optimiser_args['jac'] = None

                    for j, theta in enumerate(thetas):

                        test_alg.set_objective_map(target,{'target_val':fmin[j]})

                        test_alg.execute(theta)

                        test_alg.print_optimiser_result()

                        if test_alg.COMM.Get_rank() == 0:
                            data = np.array([test_alg.total_n_evolutions, test_alg.objective_history], dtype = np.float64)
                            data_label =  '{}_{}_{}_{}_{}'.format(alg_name, method_name, j, n_qubits, len(theta)//test_alg.total_params)
                            np.save(objective_history_filename + '/' + data_label + '.npy', data)
