import numpy as np
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt
import h5py
import qwao_mpi as qw
import os

def MPI_print(text, MPI_communicator, **kwargs):
    if MPI_communicator.Get_rank() == 0:
        print(text, **kwargs)

comm = MPI.COMM_WORLD

logname = "exp_2_results.csv"

if comm.Get_rank() == 0:
    if os.path.exists(logname):
        log = open(logname, 'a')
    else:
        log = open(logname, 'w')
        log.write('qubits, repeat, p\n')


n_qubits = int(sys.argv[1])

file_name = str(sys.argv[2])

MPI_print("Commencing for n_qubits = " + str(n_qubits), comm)

qwao = qw.MPI.qwao(n_qubits, comm)
qwao.graph(qw.graph_array.complete(qwao.size))

qwao.plan()

for i in range(1,11):

    MPI_print('Test ' + str(i) + ' of 10:', comm)

    qwao.set_qualities(qw.qualities.random_integers, seed = i)

    for p in range(1,101):

        MPI_print(p, comm, file = sys.stderr, flush = True)

        success = False

        np.random.seed(i)
        x0 = np.pi*np.random.rand(2*p) - np.pi

        qwao.execute(x0)

        probs = np.abs(qwao.final_state)**2

        good_prob = 0
        for j, prob in enumerate(probs):
            if qwao.qualities[j] / float(qwao.max_quality) >= 0.95:
                good_prob += probs[j]

        if good_prob > 0.5:
            MPI_print(str(n_qubits) + " qubits converged at p = " + str(p), comm)
            p_con = p
            break
        elif p == 100:
            p_con = 100
            MPI_print(str(n_qubits) + " did not converge by p = 100", comm)

    config_name = "q_" + str(n_qubits) + "_p_" + str(p_con) + '_' + str(i)
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

        log.write('{}, {}, {}\n'.format(
            n_qubits, i, p_con))

qwao.destroy_plan()

if comm.Get_rank() == 0:
    log.close()

