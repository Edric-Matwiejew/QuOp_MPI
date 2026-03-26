import numpy as np
from mpi4py import MPI

MPI_COMM_type = type(MPI.COMM_WORLD)

# Portable 64-bit integer MPI type
_MPI_INT64 = getattr(MPI, "INT64_T", MPI.LONG)


def __scatter_1d_array(array, partition_table, MPI_COMM, dtype):  # noqa: N803

    rank = MPI_COMM.Get_rank()
    local_i = partition_table[rank + 1] - partition_table[rank]
    operator = np.empty(local_i, dtype)

    counts = partition_table[1:] - partition_table[:-1]
    disps = partition_table[:-1] - 1

    if dtype == np.complex128:
        send_type = MPI.DOUBLE_COMPLEX
    elif dtype == np.float64:
        send_type = MPI.DOUBLE
    else:
        raise ValueError(f"Unsupported dtype for scatter: {dtype}")

    MPI_COMM.Scatterv([array, counts, disps, send_type], operator[:local_i], 0)

    return operator


def __scatter_sparse(row_starts, col_indexes, values, partition_table, MPI_COMM):  # noqa: N803
    """Scatter sparse CSR matrix data to all ranks.

    Parameters
    ----------
    row_starts : list[ndarray] or None
        List of row_starts arrays (one per matrix term), only on rank 0
    col_indexes : list[ndarray] or None
        List of col_indexes arrays, only on rank 0
    values : list[ndarray] or None
        List of values arrays, only on rank 0. If None or contains None,
        the matrix is unit-valued (all non-zeros are 1.0).
    partition_table : ndarray
        Partition table for distribution
    MPI_COMM : MPI.Comm
        MPI communicator

    Returns
    -------
    tuple
        (W_row_starts, W_col_indexes, W_values, is_unit_valued)
        W_values will be None if is_unit_valued is True
    """
    rank = MPI_COMM.Get_rank()
    size = MPI_COMM.Get_size()

    if rank == 0:
        n_terms = MPI_COMM.bcast(len(row_starts), 0)
        # Check if unit-valued: values is None or all values are 1.0
        if values is None:
            is_unit_valued = True
        else:
            is_unit_valued = all(v is None or np.allclose(v, 1.0) for v in values)
        is_unit_valued = MPI_COMM.bcast(is_unit_valued, 0)
    else:
        n_terms = MPI_COMM.bcast(None, 0)
        is_unit_valued = MPI_COMM.bcast(None, 0)

    W_row_starts = []  # noqa: N806
    W_col_indexes = []  # noqa: N806
    W_values = [] if not is_unit_valued else None  # noqa: N806

    for i in range(n_terms):

        n_local_rows = partition_table[rank + 1] - partition_table[rank]

        W_row_starts.append(np.empty(n_local_rows + 1, dtype=np.int64))

        counts = partition_table[1:] - partition_table[0:-1] + 1
        disps = partition_table[:-1] - 1

        if rank == 0:
            sends = [row_starts[i].astype(np.int64), counts, disps, _MPI_INT64]
        else:
            sends = None  # [None, counts, disps, MPI.INT]

        MPI_COMM.Scatterv(sends, W_row_starts[-1], 0)

        n_local_nnz = int(W_row_starts[-1][-1] - W_row_starts[-1][0])

        W_col_indexes.append(np.empty(n_local_nnz, dtype=np.int64))
        if not is_unit_valued:
            W_values.append(np.empty(n_local_nnz, np.complex128))

        counts = np.array(MPI_COMM.allgather(n_local_nnz), dtype=np.int64)

        disps = [0 for _ in range(size)]
        for j in range(1, size):
            disps[j] = disps[j - 1] + counts[j - 1]

        if rank == 0:
            send_indexes = [col_indexes[i].astype(np.int64), counts, disps, _MPI_INT64]
        else:
            send_indexes = None

        MPI_COMM.Scatterv(send_indexes, W_col_indexes[-1], 0)

        # Only scatter values if not unit-valued
        if not is_unit_valued:
            if rank == 0:
                v = values[i]
                if v is None:
                    raise ValueError(
                        f"values[{i}] is None but matrix is not unit-valued; "
                        "pass all-None values or provide arrays for every term"
                    )
                send_values = [
                    v.astype(np.complex128),
                    counts,
                    disps,
                    MPI.DOUBLE_COMPLEX,
                ]
            else:
                send_values = None
            MPI_COMM.Scatterv(send_values, W_values[-1], 0)

    return W_row_starts, W_col_indexes, W_values, is_unit_valued


def gather_array(array, partition_table, comm):

    if comm.Get_rank() == 0:
        gathered_array = np.empty(partition_table[-1] - 1, dtype=array.dtype)
    else:
        gathered_array = None

    counts = partition_table[1:] - partition_table[:-1]
    send_count = counts[comm.Get_rank()]

    comm.Gatherv(array[:send_count], [gathered_array, counts], 0)

    return gathered_array
