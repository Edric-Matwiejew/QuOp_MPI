from mpi4py import MPI
import numpy as np

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

	comm_opt_mapping = []
	for key in process_dict.keys():

		n_subcomms = variables//n_processors

		if n_subcomms > len(process_dict[key]):
			comm_opt_mapping.append(process_dict[key])
			continue

		for part in np.split(process_dict[key], n_subcomms):
			comm_opt_mapping.append(part)
		
	colours = []
	for i, comm in enumerate(comm_opt_mapping):
		for _ in range(len(comm)):
			colours.append(i)

	COMM_OPT = MPI.Comm.Split(COMM, colours[rank], colours[rank])

	opt_size = COMM_OPT.Get_size()

	var_map = [[] for _ in range(opt_size)]
	for var in range(variables):
		var_map[var % opt_size].append(var)
	
	if COMM.Get_rank() == 0:
		print(COMM.Get_rank(), COMM_OPT.Get_rank())

	return COMM_OPT, var_map


#def jacobian(x, MPI_COMMUNICATOR = COMM_OPT, variable_mapping = var_map):

create_communication_topology(COMM, 8)


