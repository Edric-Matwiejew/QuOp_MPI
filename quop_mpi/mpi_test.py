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
                #print(process_dict, flush = True)

        else:

                COMM.send(process, dest = 0)
                process_dict = None

        process_dict = COMM.bcast(process_dict)

        n_processors = len(process_dict)
        n_subcomm_processes = (variables + 1)//n_processors

        if n_subcomm_processes == 0:
        # If the number of MPI processes (physical nodes) is greater than the number of 
        # variables + 1 distribute state evolution across multiple physical nodes.

            comm_opt_mapping = [[] for _ in range(variables + 1)]
            for i, key in enumerate(process_dict.keys()):
                for rank in process_dict[key]:
                    comm_opt_mapping[i % variables].append(rank)

        else:
        # If variables + 1 is greater than the number of MPI processes (physical nodes)

            comm_opt_mapping = []

            for key in process_dict.keys():
            
                if n_subcomm_processes == 1:
                # If variables + 1 is equal to the number of MPI processes, then each
                # create a sub-communicator per physical node consisting of multiple ranks.
                    comm_opt_mapping.append(process_dict[key])
                    continue

                if n_subcomm_processes > len(process_dict[key]):
                # If variables + 1 is greater than the number of physical MPI processes
                # then create subcommunicators consisting of 1 rank. 
                    for process in process_dict[key]:
                        comm_opt_mapping.append([process])
                    continue

                for part in np.array_split(process_dict[key], n_subcomm_processes):
                    comm_opt_mapping.append(part)

    # In the following we create COMM_OPT, the root subcommunicator and the subcommunicators used
    # to calculate the jacobian, and COMM_JAC, the root process in the root subcommunicator and
    # and the sub communicators used to caluclate the gradients.

        # Colours specify membership to a particular COMM_OPT
        # comm_opt is the COMM rank of the root process in each COMM_OPT
        colours = []
        comm_opt_roots = []
        #print(comm_opt_mapping,flush = True)
        for i, comm in enumerate(comm_opt_mapping):
            #print(i, comm, flush = True)
            comm_opt_roots.append(min(comm))
            for _ in range(len(comm)):
                colours.append(i)

        comm_opt_roots.sort()

        #print(colours, flush = True)

        COMM_OPT = MPI.Comm.Split(
                COMM,
                colours[COMM.Get_rank()],
                COMM.Get_rank())

        var_map = [[] for _ in range(len(comm_opt_mapping))]
        for var in range(variables):
        # The subcommunicator container COMM rank 0 is used to compute the objective function
        # it is not assigned variables for gradient calculations.
            var_map[1:][var % len(comm_opt_mapping) - 1].append(var)

        # Create COMM_JAC, a communicator containing the COMM_OPT used to calculate the gradient
        # values and the root process of the sub communicator responsible for calls to the
        # objective function.
        jac_ranks = [rank for rank in range(COMM.Get_size()) if colours[rank] != 0]
        jac_ranks.insert(0,0)
        #print('jack ranks', jac_ranks, flush = True)
        world_group = MPI.Comm.Get_group(COMM)
        jac_group = MPI.Group.Incl(world_group, jac_ranks)
        COMM_JAC = COMM.Create_group(jac_group)

        # Shift COMM rank of the roots to be relaitive to COMM_JAC rank is sent negative
        #rank_shift = COMM_JAC.Get_size() - COMM.Get_size()
        #for i in range(COMM_JAC.Get_size()):
        #    comm_opt_roots[i] -= rank_shift
        
        #print("ROOTS", comm_opt_roots, flush = True)

        return COMM_OPT, var_map, comm_opt_roots, colours, COMM_JAC


#def jacobian(x, MPI_COMMUNICATOR = COMM_OPT, variable_mapping = var_map):
#    """
#    qwoa = qu.MPI.qwoa(n_qubits, comm, parallel = "jacobian")
#    """


p = 16
n_qubits = 20

rng = np.random.RandomState(1)

def x0(p):
    return rng.uniform(low = 0, high = 2*np.pi, size = 2 * p)


COMM_OPT, var_map, comm_opt_roots, colours, COMM_JAC = create_communication_topology(COMM,2*p)

#if COMM.Get_rank() == 0:
    #print(var_map, comm_opt_roots, colours, flush = True)

if colours[COMM.Get_rank()] == 0:
    x = x0(p)
    #print("X",x,flush=True)
else:
    x = None

qwoa = qu.MPI.qwoa(n_qubits, COMM_OPT)
qwoa.set_initial_state(name="equal")
qwoa.set_graph(qu.graph_array.complete(n_qubits))
qwoa.set_qualities(qu.qualities.random_floats)
#if COMM.Get_rank() == 0:
#    print(qwoa.qualities[0], qwoa.qualities[13], qwoa.qualities[69])
qwoa.plan()

def mpi_jacobian(x, tol = 1e-8):

    #qwoa.stop = COMM.bcast(qwoa.stop, root = 0)
    #print("jac enter", COMM.Get_rank(), flush = True)
    COMM_JAC.barrier()

    qwoa.stop = COMM_JAC.bcast(qwoa.stop, 0)

    if qwoa.stop:
        #print(COMM_JAC.Get_rank(), 'OUTOUTOUT', flush = True)
        COMM_JAC.barrier()
        return

    x = COMM_JAC.bcast(x, 0)
    qwoa.expt = COMM_JAC.bcast(qwoa.expt, 0)
    #print(COMM.Get_rank(), qwoa.expt,flush=True)

    start = time.time()
    #x = qwoa.gammas_ts
    x_jac_temp = np.empty(len(x))
    partials = []

    #if colours[COMM.Get_rank()] == 0:
    #    expectation = qwoa.expectation()
    #else:
    #    expectation = None
        
    expectation = qwoa.expt
    #if COMM.Get_rank() == 0:
    #    print(expectation, flush = True)

    if  COMM.Get_rank() != 0:
        #print(x, flush = True)
        #print(expectation, flush = True)
        xs, ts = np.split(x,2)
        qwoa.evolve_state(xs, ts)
        expectation = qwoa.expectation()
        h = 1.4901161193847656e-08    #np.max(np.abs(x_jac_temp))*np.sqrt(tol)
        for var in var_map[colours[COMM.Get_rank()]]:
            x_jac_temp[:] = x
            #print("WHO DO WHAT", var, COMM.Get_rank(), flush = True)
            #print(h,flush=True)
            x_jac_temp[var] += h
            xs, ts = np.split(x_jac_temp,2)
            qwoa.evolve_state(xs, ts)
            partials.append((qwoa.expectation() - expectation)/h)
    
    opt_root = comm_opt_roots[colours[COMM.Get_rank()]]
    
    if COMM.Get_rank() == 0:
        jacobian = np.zeros(2*p, dtype = np.float64)
        #for i, var in enumerate(var_map[colours[COMM.Get_rank()]]):
        #    jacobian[var] = partials[i]
        reqs = []
        for root, mapping in zip(comm_opt_roots,var_map):
            if root > 0:
                for var in mapping:
                    #print('Irecv', COMM.Get_rank(), 'source', root, 'tag', var, jacobian[var:var+1], comm_opt_roots,  flush = True)
                    #reqs.append(COMM.Recv([jacobian[var:var+1], MPI.DOUBLE],  source = root, tag = var))
                    COMM.Recv([jacobian[var:var+1], MPI.DOUBLE],  source = root, tag = var)
                    #print("onjac",jacobian, flush = True)
 
        #for req in reqs:
        #    req.wait()
        #    print(jacobian, flush = True)

   
    elif COMM_OPT.Get_rank() == 0:
        reqs = []
        jacobian = None
        for part, mapping in zip(partials, var_map[colours[COMM.Get_rank()]]):
            #print('Isend', 'sender', COMM.Get_rank(), COMM_OPT.Get_rank(), 'tag', mapping, np.array([part]), partials, var_map,mapping, flush = True)
            #reqs.append(COMM.Send([np.array([part]), MPI.DOUBLE], dest = 0, tag = mapping))
            COMM.Send([np.array([part]), MPI.DOUBLE], dest = 0, tag = mapping)

        #for req in reqs:
        #    req.wait()

    else:
        jacobian = None
    finish = time.time()
    COMM_JAC.barrier()
    #if COMM.Get_rank() == 0:
        #print()
        #for jac in jacobian:
            #print(jac, flush = True)
        #print()
    #COMM.barrier()
    if COMM.Get_rank() == 0:
        #print(jacobian, flush = True)
        return jacobian
    else:
        #print("OUT", COMM.Get_rank())
        return None

#if colours[COMM.Get_rank()] == 0:
#    xs, ts = np.split(x,2)
#    qwoa.evolve_state(xs, ts)

#print(mpi_jacobian(x), flush = True)

qwoa.set_optimiser('scipy', {'method':'BFGS','tol':1e-8,'jac':mpi_jacobian},['fun','nfev','success'])

#qwoa.comm2 = COMM

qwoa.stop = False
start = time.time()
if colours[COMM.Get_rank()] == 0:
    qwoa.stop = False
    qwoa.execute(x)
    qwoa.stop = True
    #print('root in jac out')
    #qwoa.stop = COMM_JAC.bcast(qwoa.stop, 0)
    if COMM.Get_rank() == 0:
        mpi_jacobian(qwoa.gammas_ts)
    #print('root out jac out')
    #print('execution time', time.time() - start)
    qwoa.print_result()
else:
    qwoa.stop = False
    qwoa.gammas_ts = None
    cnt = 0
    while not qwoa.stop:
        mpi_jacobian(qwoa.gammas_ts)
        #print(qwoa.expt, flush = True)
        cnt += 1
        #print("CNT", cnt, flush = True)

#qwoa.set_optimiser('scipy', {'method':'BFGS','tol':1e-5},['fun','nfev','success'])


#if colours[COMM.Get_rank()] == 0:
#    x = COMM_OPT.bcast(x, root = 0)
#    start = time.time()
#    qwoa.execute(x)
#    #print('execution time', time.time() - start)
#    qwoa.print_result()
#
COMM.barrier()
qwoa.destroy_plan()
