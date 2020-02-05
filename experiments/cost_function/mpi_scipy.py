from mpi4py import MPI
import numpy as np
import qwao_mpi as qw

def MPI_print(text, MPI_communicator, **kwargs):
    if MPI_communicator.Get_rank() == 0:
        print(text, **kwargs)

comm = MPI.COMM_WORLD

logname = "costfunction.csv"

if comm.Get_rank() == 0:
    log = open(logname, 'a')
    log.write('p, success_probability\n')

n_qubits = 16

qwao = qw.MPI.qwao(n_qubits, comm)
qwao.graph(qw.graph_array.circle(qwao.size))

qwao.set_qualities(qw.qualities.random_floats,seed = 13)

qwao.initial_state[:] = 0

if comm.Get_rank() == 0:
    qwao.initial_state[0] = 1/np.sqrt(2.0)
    qwao.initial_state[1] = 1/np.sqrt(2.0)

qwao.plan()

rank = comm.Get_rank()

for p in range(1,26):

    MPI_print('Starting p: ' + str(p), comm)

    x0 =  np.zeros(2*p, dtype = np.float64)
    x0[:] = 0

    result = qwao.execute(x0)

    MPI_print(result, comm)

    probs = np.abs(qwao.final_state)**2

    success_prob = 0.0
    qual = qwao.qualities[j]/qwao.max_quality

    for j, prob in enumerate(probs):
        if qual >= 0.9:
            success_prob += prob
            print(qual)

    success_prob = comm.allreduce(success_prob, op = MPI.SUM)

    MPI_print('Success probability: ' + str(success_prob), comm)

    qwao.save('costfunction', '16_qubits_p_' + str(p), action = 'w')

    if comm.Get_rank() == 0:
        log.write('{},{}\n'.format(
            p,
            success_prob))
        log.flush()

qwao.destroy_plan()

if comm.Get_rank() == 0:
    log.close()
