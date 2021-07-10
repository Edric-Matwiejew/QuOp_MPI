from mpi4py import MPI
import numpy as np
import quop_mpi as qu
import time

COMM = MPI.COMM_WORLD

def create_communication_topology(MPI_COMMUNICATOR, variables):

        COMM = MPI_COMMUNICATOR

        size = COMM.Get_size()
        rank = COMM.Get_rank()

        process = MPI.Get_processor_name()

        if rank == 0:

                processes = [process]

                for i in range(1, size):
                        processes.append(COMM.recv(source = i))

                unique_processes = list(set(processes))
                lsts = [[] for _ in range(len(unique_processes))]
                process_dict = dict(zip(unique_processes, lsts))

                for i, proc in enumerate(processes):
                        process_dict[proc].append(i)

                for key in process_dict:
                        process_dict[key] = np.array(process_dict[key])

        else:

                COMM.send(process, dest = 0)
                process_dict = None

        process_dict = COMM.bcast(process_dict)

        n_processors = len(process_dict)
        n_subcomm_processes = variables//n_processors

        if n_subcomm_processes == 0:
        
            comm_opt_mapping = [[] for _ in range(variables)]
            for i, key in enumerate(process_dict.keys()):
                for rank in process_dict[key]:
                    comm_opt_mapping[i % variables].append(rank)

        else:
            #print("HI", process_dict, n_subcomm_processes, flush = True)

            comm_opt_mapping = []

            for key in process_dict.keys():

                if n_subcomm_processes == 1:
                    comm_opt_mapping.append(process_dict[key])
                    continue

                if n_subcomm_processes > len(process_dict[key]):
                    for process in process_dict[key]:
                        comm_opt_mapping.append([process])
                    continue

                for part in np.array_split(process_dict[key], n_subcomm_processes):
                    comm_opt_mapping.append(part)


        colours = []
        comm_opt_roots = []
        #print(comm_opt_mapping,flush = True)
        for i, comm in enumerate(comm_opt_mapping):
            #print(i, comm, flush = True)
            comm_opt_roots.append(min(comm))
            for _ in range(len(comm)):
                colours.append(i)


        COMM_OPT = MPI.Comm.Split(
                COMM,
                colours[rank],
                COMM.Get_rank())

        opt_size = COMM_OPT.Get_size()
        #print(opt_size, flush = True)
        var_map = [[] for _ in range(len(comm_opt_mapping))]
        for var in range(variables):
                var_map[var % len(comm_opt_mapping)].append(var)

        return COMM_OPT, var_map, comm_opt_roots, colours


#def jacobian(x, MPI_COMMUNICATOR = COMM_OPT, variable_mapping = var_map):
#    """
#    qwoa = qu.MPI.qwoa(n_qubits, comm, parallel = "jacobian")
#    """


p = 8
n_qubits = 10

rng = np.random.RandomState(1)

def x0(p):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)


COMM_OPT, var_map, comm_opt_roots, colours = create_communication_topology(COMM,2*p)

qwoa = qu.MPI.qwoa(n_qubits, COMM_OPT)
qwoa.set_initial_state(name="equal")
qwoa.set_graph(qu.graph_array.complete(n_qubits))
qwoa.set_qualities(qu.qualities.random_floats)
qwoa.plan()

if colours[COMM.Get_rank()] == 0:
    x = x0(p)
else:
    x = None
 

def mpi_jacobian(x, tol = 1e-8):

    #qwoa.stop = COMM.bcast(qwoa.stop, root = 0)

    start = time.time()
    x_jac_temp = np.empty(len(x))
    partials = []

    #if colours[COMM.Get_rank()] == 0:
    #    expectation = qwoa.expectation()
    #else:
    #    expectation = None
        
    expectation = qwoa.expt
    #if COMM.Get_rank() == 0:
    #    print(expectation, flush = True)

    for var in var_map[colours[COMM.Get_rank()]]:
        x_jac_temp[:] = x
        h =  x_jac_temp[var]*np.sqrt(tol)
        x_jac_temp[var] += h
        xs, ts = np.split(x_jac_temp,2)
        qwoa.evolve_state(xs, ts)
        partials.append((qwoa.expectation() - expectation)/h)
    
    opt_root = comm_opt_roots[colours[COMM_OPT.Get_rank()]]
    
    if COMM.Get_rank() == 0:
        jacobian = np.zeros(2*p, dtype = np.float64)
        for i, var in enumerate(var_map[colours[COMM.Get_rank()]]):
            jacobian[var] = partials[i]
        for root, mapping in zip(comm_opt_roots,var_map):
            if root != 0:
                for var in mapping:
                    COMM.Irecv([jacobian[var:var+1], MPI.DOUBLE],  source = root, tag = root)
    
    elif COMM_OPT.Get_rank() == 0:
        jacobian = None
        for i, var in enumerate(var_map[colours[COMM.Get_rank()]]):
            COMM.Isend([partials[i], MPI.DOUBLE], dest = 0, tag = COMM.Get_rank())
    else:
        jacobian = None
    
    COMM.barrier()
    
    finish = time.time()
    
    if COMM.Get_rank() == 0:
        #print(jacobian)
        return jacobian
    else:
        return None

#if colours[COMM.Get_rank()] == 0:
#    xs, ts = np.split(x,2)
#    qwoa.evolve_state(xs, ts)

#print(mpi_jacobian(x), flush = True)

qwoa.set_optimiser('scipy', {'method':'BFGS','tol':1e-5,'jac':mpi_jacobian},['fun','nfev','success'])

qwoa.comm2 = COMM

qwoa.stop = False
start = time.time()
if colours[COMM.Get_rank()] == 0:
    qwoa.stop = False
    qwoa.execute(x)
    print('execution time', time.time() - start)
    qwoa.print_result()
else:
    qwoa.stop = False
    qwoa.gammas_ts = None
    while not qwoa.stop:
        qwoa.stop = COMM.bcast(qwoa.stop, 0)
        #print("GOT STOP", qwoa.stop, flush = True)
        qwoa.gammas_ts = COMM.bcast(qwoa.gammas_ts, 0)
        #print("truth?", gammas_ts, flush = True)
        if not qwoa.stop:
            qwoa.expt = COMM.bcast(qwoa.expt, 0)
            #print('pre jac', qwoa.gammas_ts, flush = True)
            mpi_jacobian(qwoa.gammas_ts)


#qwoa.set_optimiser('scipy', {'method':'BFGS','tol':1e-5},['fun','nfev','success'])
#
#x = COMM.bcast(x, root = 0)
#
#if colours[COMM.Get_rank()] == 0:
#    start = time.time()
#    qwoa.execute(x)
#    print('execution time', time.time() - start)
#    qwoa.print_result()
#
COMM.barrier()
qwoa.destroy_plan()
