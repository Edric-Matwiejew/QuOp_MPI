import numpy as np
from mpi4py import MPI

def __scatter_1D_array(array, partition_table, MPI_COMM, dtype):

    rank = MPI_COMM.Get_rank()
    local_i = partition_table[rank + 1] - partition_table[rank]
    operator = np.empty(local_i, dtype)

    counts = partition_table[1:] - partition_table[:-1]
    disps = partition_table[:-1] - 1

    if dtype == np.complex128:
        send_type = MPI.DOUBLE_COMPLEX
    elif dtype == np.float64:
        send_type = MPI.DOUBLE

    MPI_COMM.Scatterv([array, counts, disps, send_type], operator[:local_i], 0)

    return operator

def __scatter_sparse(row_starts, col_indexes, values, partition_table, MPI_COMM):

    rank = MPI_COMM.Get_rank()
    size = MPI_COMM.Get_size()

    lb = partition_table[rank] - 1
    ub = partition_table[rank + 1] - 1

    if rank == 0:
        n_terms = MPI_COMM.bcast(len(row_starts), 0)
    else:
        n_terms = MPI_COMM.bcast(None, 0)

    W_row_starts = []
    W_col_indexes = []
    W_values = []

    for i in range(n_terms):

        n_local_rows = partition_table[rank + 1] - partition_table[rank]

        W_row_starts.append(np.empty(n_local_rows + 1, np.int32))

        counts = partition_table[1:] - partition_table[0:-1] + 1
        disps = partition_table[:-1] - 1

        if rank == 0:
            sends = [row_starts[i], counts, disps, MPI.INT]
        else:
            sends = None #[None, counts, disps, MPI.INT]

        MPI_COMM.Scatterv(sends, W_row_starts[-1], 0)

        n_local_nnz = W_row_starts[-1][-1] - W_row_starts[-1][0]

        W_col_indexes.append(np.empty(n_local_nnz, np.int32))
        W_values.append(np.empty(n_local_nnz, np.complex128))

        counts = np.zeros(size, int)
        counts[rank] = n_local_nnz

        for j in range(size):
            counts[j] = MPI_COMM.bcast(n_local_nnz, j)

        disps = [0 for _ in range(size)]
        for j in range(1, size):
            disps[j] = disps[j - 1] + counts[j - 1]


        if rank == 0:
            send_indexes = [col_indexes[i], counts, disps, MPI.INT]
            send_values = [values[i], counts, disps, MPI.DOUBLE_COMPLEX]
        else:
            send_indexes = None #[None, counts, disps, MPI.INT]
            send_values = None #[None, counts, disps, MPI.DOUBLE_COMPLEX]

        MPI_COMM.Scatterv(send_indexes, W_col_indexes[-1], 0)
        MPI_COMM.Scatterv(send_values, W_values[-1], 0)

        #MPI_COMM.barrier()
        #exit()

    return W_row_starts, W_col_indexes, W_values

def shrink_communicator(newsize, colours, COMM_OPT):

    subcolours = []
    for i in range(COMM_OPT.Get_size()):
        if i < newsize:
            subcolours.append(0)
        else:
            colours[i] = -1
            subcolours.append(MPI.UNDEFINED)

    COMM_OPT_NEW = MPI.Comm.Split(
            COMM_OPT,
            subcolours[COMM_OPT.Get_rank()],
            COMM_OPT.Get_rank())

    return colours, COMM_OPT_NEW

def gather_array(array, partition_table, COMM_OPT):

    if COMM_OPT.Get_rank() == 0:
        gathered_array = np.empty(partition_table[-1] - 1, type(array[0]))
    else:
        gathered_array = None

    counts = partition_table[1:] - partition_table[:-1]

    COMM_OPT.Gatherv(array, [gathered_array, counts], 0)

    return gathered_array

