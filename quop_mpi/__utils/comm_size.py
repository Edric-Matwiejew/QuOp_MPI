import numpy as np

def __max_sizes(unitaries, system_size, comm_size, COMM):
    return [unitary.propagators[-1].max_comm_size(system_size, comm_size, unitary.comm_size_constraints, COMM) for unitary in unitaries]

def max_compatible_size(unitaries, system_size, available_ranks, COMM):
    max_sizes = __max_sizes(unitaries, system_size, available_ranks, COMM)
    while max_sizes.count(max_sizes[0]) != len(max_sizes):
        max_sizes = __max_sizes(unitaries, system_size, available_ranks, COMM)
        available_ranks = min(max_sizes)
    return max_sizes[0]

def vector_partitioning(system_size, MPI_COMM):

    comm_size = MPI_COMM.Get_size()
    rank = MPI_COMM.Get_rank()

    partition_table = [system_size//comm_size for _ in range(comm_size)]
    remainder = system_size - sum(partition_table)

    for i in range(remainder):
        partition_table[i % comm_size + 1] += 1

    partition_table = [1] + partition_table

    for i in range(1, comm_size + 1):
        partition_table[i] += partition_table[i - 1]

    local_i = partition_table[rank + 1] - partition_table[rank]
    local_i_offset = partition_table[rank] - 1

    alloc_local = local_i

    return local_i, local_i_offset, alloc_local, np.array(partition_table, dtype = np.int32)
