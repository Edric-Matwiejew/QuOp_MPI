import numpy as np
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt
import h5py
import qwao_mpi as qw
import os

def MPI_std(local_array, global_N, MPI_communicator):
    mean = MPI_communicator.allreduce(np.sum(local_array), op = MPI.SUM) / np.float64(global_N)
    return np.sqrt(MPI_communicator.allreduce(np.sum((local_array - mean)**2) / np.float(global_N - 1), op = MPI.SUM))

def MPI_median(local_array, MPI_communicator):

    if MPI_communicator.Get_rank() == 0:
        pivot = np.random.choice(local_array)
    else:
        pivot = 0

    pivot = MPI_communicator.bcast(pivot, root = 0)

    upper = []
    lower = []
    for element in local_array:
        if element <= pivot:
            lower.append(element)
        else:
            upper.append(element)

    local_lower_n = len(lower)
    local_upper_n = len(upper)

    total_lower = MPI_communicator.allreduce(local_lower_n, op = MPI.SUM)
    total_upper = MPI_communicator.allreduce(local_upper_n, op = MPI.SUM)


    while (abs(total_lower - total_upper) > 1):

        if total_lower < total_upper:

            array = upper

            pivots = np.empty(MPI_communicator.Get_size(), dtype = np.float64)

            if len(array) > 0:
                pivot = np.random.choice(upper)
            else:
                pivot = 0

            pivots[MPI_communicator.Get_rank()] = pivot

            pivots = MPI_communicator.reduce(pivots, op = MPI.SUM, root = 0)

            if MPI_communicator.Get_rank() == 0:
                nz = np.where(pivots > 0)[0]
                pivot = pivots[np.random.choice(nz)]

            pivot = MPI_communicator.bcast(pivot, root = 0)

            lower = []
            upper = []

            for element in array:

                if element <= pivot:
                    lower.append(element)
                    local_lower_n += 1
                    local_upper_n -= 1
                else:
                    upper.append(element)

        else:

            array = lower

            pivots = np.empty(MPI_communicator.Get_size(), dtype = np.float64)
            if len(array) > 0:
                pivot = np.random.choice(lower)
            else:
                pivot = 0

            pivots[MPI_communicator.Get_rank()] = pivot

            pivots = MPI_communicator.reduce(pivots, op = MPI.SUM, root = 0)

            if MPI_communicator.Get_rank() == 0:
                nz = np.where(pivots > 0)[0]
                pivot = pivots[np.random.choice(nz)]

            pivot = MPI_communicator.bcast(pivot, root = 0)

            lower = []
            upper = []

            for element in array:

                if element <= pivot:
                    lower.append(element)
                else:
                    upper.append(element)
                    local_lower_n -= 1
                    local_upper_n += 1

        total_lower_prior = total_lower
        total_upper_prior = total_upper

        total_lower = MPI_communicator.allreduce(local_lower_n, op = MPI.SUM)
        total_upper = MPI_communicator.allreduce(local_upper_n, op = MPI.SUM)

    return pivot

def MPI_print(text, MPI_communicator, **kwargs):
    if MPI_communicator.Get_rank() == 0:
        print(text, **kwargs)

comm = MPI.COMM_WORLD
logname = "results.csv"
if comm.Get_rank() == 0:
    if os.path.exists(logname):
        log = open(logname, 'a')
    else:
        log = open(logname, 'w')
        log.write('qubits, repeat, p, total_outliers, best\n')


n_qubits = int(sys.argv[1])
file_name = str(sys.argv[2])

MPI_print("Commencing for n_qubits = " + str(n_qubits), comm)

qwao = qw.MPI.qwao(n_qubits, comm)
qwao.graph(qw.graph_array.complete(qwao.size))

qwao.plan()

for i in range(1,11):

    MPI_print('Test ' + str(i) + ' of 5:', comm)

    qwao.set_qualities(qw.qualities.random_integers, seed = i)

    for p in range(1,101):

        MPI_print(p, comm, file = sys.stderr, flush = True)

        success = False

        config_name = "q_" + str(n_qubits) + "_p_" + str(p) + '_' + str(i)

        np.random.seed(p)
        x0 = np.pi*np.random.rand(2*p) - np.pi
        qwao.execute(x0)

        probs = np.abs(qwao.final_state)**2
        approx_median = MPI_median(probs, comm)
        outliers = np.where(np.abs(probs - approx_median)/approx_median > 3)

        total_out = 0
        for outlier in outliers:
            for out in outlier:
                total_out += 1
                if qwao.qualities[out] == qwao.max_quality:
                    print(qwao.qualities[out], qwao.max_quality)
                success = True

        success = comm.allreduce(success, op = MPI.LOR)
        total_outliers = comm.allreduce(total_out, op = MPI.SUM)

        if total_outliers > 0:
            MPI_print(str(n_qubits) + " qubits converged at p = " + str(p), comm)
            MPI_print("Total outliers: " + str(total_outliers)  + ', ' + str(100*total_outliers/float(qwao.size)) + "%", comm)
            p_con = p
            break
        elif p == 100:
            p_con = 100
            MPI_print(str(n_qubits) + " did not converge by p = 200", comm)

    qwao.save(file_name, config_name, action = "a")

    if comm.Get_rank() == 0:
        print(qwao.result)
        f = h5py.File(file_name + ".h5", "r")
        state = np.array(f[config_name]['final_state']).view(np.complex128)
        probs = np.float64(np.multiply(state, np.conj(state)))
        plt.ylim(np.min(probs), np.max(probs))
        qualities = np.array(f[config_name + '/qualities'], dtype = np.float64)
        plt.scatter(qualities, probs, marker = '*')
        plt.savefig('plots/' + config_name)
        plt.cla()
        f.close()

        log.write('{}, {}, {}, {}, {}\n'.format(
            n_qubits, i, p_con, total_outliers, str(success)))

qwao.destroy_plan()

if comm.Get_rank() == 0:
    log.close()

