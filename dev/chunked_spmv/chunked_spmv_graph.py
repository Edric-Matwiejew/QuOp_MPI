"""
Memory-efficient chunked SpMV using MPI graph communicators.

Key optimizations:
1. Unit-valued matrices: No values array stored (all 1s)
2. Graph communicators: MPI optimizes routing via Neighbor_alltoallv
3. O(unique_remote) storage: Binary search instead of O(nnz) mapping
4. Chunked communication: Bounds memory usage per SpMV call

Memory complexity:
- Old approach: O(nnz) for col_to_remote_pos mapping
- New approach: O(unique_remote) for sorted recv_indices + binary search

At 34 qubits with 4096 ranks:
- Old: ~1.1 GB per rank (would crash)
- New: ~0.09 GB per rank (manageable)

Run with: mpirun -np 4 python chunked_spmv_graph.py [n_qubits] [chunk_size] [n_iters]
"""

import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import time


@dataclass
class GraphCommData:
    """
    Communication data structure for chunked SpMV with O(unique_remote) storage.
    
    Memory layout (all O(unique_remote) or O(neighbors)):
    - recv_indices_sorted: Sorted unique remote column indices we need
    - recv_perm: Permutation to unsort (restore neighbor order for Neighbor_alltoallv)
    - recv_counts/disps: Per-neighbor counts and displacements
    - send_counts/disps: What neighbors request from us
    - send_offsets: Offset of each requested index into local array
    
    Optional O(nnz) optimization:
    - nnz_to_recv_pos: Maps each NNZ index to recv buffer position (-1 if local)
      Trades O(nnz) int64 for O(1) lookup instead of O(log N) binary search.
    """
    graph_comm: MPI.Comm
    in_neighbors: np.ndarray   # Ranks that send to us
    out_neighbors: np.ndarray  # Ranks we send to
    
    # Receiving (what we need from others) - O(unique_remote)
    recv_indices_sorted: np.ndarray  # Sorted for binary search
    recv_perm: np.ndarray            # Permutation to unsort
    recv_counts: np.ndarray          # Per-neighbor counts
    recv_disps: np.ndarray           # Per-neighbor displacements
    total_recv: int
    
    # Sending (what others need from us) - O(unique_remote of neighbors)
    send_offsets: np.ndarray    # Local offsets of requested elements
    send_counts: np.ndarray     # Per-neighbor counts
    send_disps: np.ndarray      # Per-neighbor displacements
    total_send: int
    
    # Optional O(nnz) precomputed mapping (if precompute_nnz_map=True)
    nnz_to_recv_pos: Optional[np.ndarray] = None  # -1 for local, else recv_buf position
    
    # Partition info for fast local check
    lb: int = 0
    ub: int = 0


def generate_partition_table(system_size: int, comm: MPI.Comm) -> np.ndarray:
    """Generate partition table (0-based, exclusive end)."""
    size = comm.Get_size()
    table = np.zeros(size + 1, dtype=np.int64)
    base = system_size // size
    rem = system_size % size
    for i in range(size):
        table[i + 1] = table[i] + base + (1 if i < rem else 0)
    return table


def find_owner(col: int, partition_table: np.ndarray) -> int:
    """Find which rank owns a column (binary search)."""
    # partition_table[r] <= col < partition_table[r+1] means rank r owns col
    left, right = 0, len(partition_table) - 2
    while left < right:
        mid = (left + right + 1) // 2
        if partition_table[mid] <= col:
            left = mid
        else:
            right = mid - 1
    return left


def build_hypercube_csr(n_qubits: int, partition_table: np.ndarray, 
                        rank: int, sort_rows: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Build local CSR for hypercube adjacency (unit-valued).
    
    Args:
        sort_rows: If True, sort column indices within each row for efficient lookup.
    """
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    n_local = ub - lb
    
    row_starts = np.zeros(n_local + 1, dtype=np.int64)
    col_indexes = np.zeros(n_local * n_qubits, dtype=np.int64)
    
    idx = 0
    for i in range(n_local):
        row_starts[i] = idx
        global_row = lb + i
        for k in range(n_qubits):
            col_indexes[idx] = global_row ^ (1 << k)
            idx += 1
        
        # Sort columns within this row
        if sort_rows:
            col_indexes[row_starts[i]:idx].sort()
    
    row_starts[n_local] = idx
    
    return row_starts, col_indexes


def sort_csr_rows(row_starts: np.ndarray, col_indexes: np.ndarray) -> None:
    """Sort column indices within each row in-place."""
    n_local = len(row_starts) - 1
    for i in range(n_local):
        start, end = row_starts[i], row_starts[i + 1]
        col_indexes[start:end].sort()


def setup_graph_comm(row_starts: np.ndarray, col_indexes: np.ndarray,
                     partition_table: np.ndarray, comm: MPI.Comm) -> GraphCommData:
    """
    Set up graph communicator with O(unique_remote) storage.
    
    Key: Uses binary search into sorted recv_indices instead of O(nnz) mapping.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    
    # Step 1: Find unique remote columns grouped by owner
    remote_by_rank: dict = {r: set() for r in range(size)}
    
    for col in col_indexes:
        if col < lb or col >= ub:
            owner = find_owner(col, partition_table)
            remote_by_rank[owner].add(col)
    
    # Convert to sorted lists per rank
    remote_cols_per_rank = {r: sorted(remote_by_rank[r]) for r in range(size)}
    
    # Step 2: Identify neighbors (ranks we actually communicate with)
    out_neighbors = sorted([r for r in range(size) 
                           if r != rank and remote_cols_per_rank[r]])
    
    # Exchange to find in_neighbors
    all_out = comm.allgather(out_neighbors)
    in_neighbors = sorted([r for r in range(size) if rank in all_out[r]])
    
    out_neighbors = np.array(out_neighbors, dtype=np.intc)
    in_neighbors = np.array(in_neighbors, dtype=np.intc)
    
    # Step 3: Create graph communicator
    graph_comm = comm.Create_dist_graph_adjacent(
        in_neighbors, out_neighbors,
        sourceweights=None, destweights=None,
        info=MPI.INFO_NULL, reorder=False
    )
    
    # Step 4: Build recv arrays (what we need from out_neighbors)
    # Concatenate in neighbor order for Neighbor_alltoallv
    all_recv_indices = []
    recv_counts = np.zeros(len(out_neighbors), dtype=np.intc)
    
    for i, r in enumerate(out_neighbors):
        cols = remote_cols_per_rank[int(r)]
        recv_counts[i] = len(cols)
        all_recv_indices.extend(cols)
    
    all_recv_indices = np.array(all_recv_indices, dtype=np.int64)
    total_recv = len(all_recv_indices)
    
    recv_disps = np.zeros(len(out_neighbors), dtype=np.intc)
    for i in range(1, len(out_neighbors)):
        recv_disps[i] = recv_disps[i-1] + recv_counts[i-1]
    
    # Create sorted version + permutation for binary search
    if total_recv > 0:
        sort_perm = np.argsort(all_recv_indices)
        recv_indices_sorted = all_recv_indices[sort_perm]
        # Inverse permutation to unsort
        recv_perm = np.empty_like(sort_perm)
        recv_perm[sort_perm] = np.arange(total_recv)
    else:
        recv_indices_sorted = np.array([], dtype=np.int64)
        recv_perm = np.array([], dtype=np.int64)
    
    # Step 5: Exchange to set up send side
    send_counts = np.zeros(len(in_neighbors), dtype=np.intc)
    graph_comm.Neighbor_alltoall(recv_counts, send_counts)
    
    send_disps = np.zeros(len(in_neighbors), dtype=np.intc)
    for i in range(1, len(in_neighbors)):
        send_disps[i] = send_disps[i-1] + send_counts[i-1]
    
    total_send = int(np.sum(send_counts))
    
    # Exchange indices that neighbors request from us
    requested_indices = np.zeros(max(total_send, 1), dtype=np.int64)
    graph_comm.Neighbor_alltoallv(
        [all_recv_indices, (recv_counts, recv_disps)],
        [requested_indices, (send_counts, send_disps)]
    )
    
    # Convert to local offsets
    send_offsets = requested_indices[:total_send] - lb
    
    return GraphCommData(
        graph_comm=graph_comm,
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        recv_indices_sorted=recv_indices_sorted,
        recv_perm=recv_perm,
        recv_counts=recv_counts,
        recv_disps=recv_disps,
        total_recv=total_recv,
        send_offsets=send_offsets,
        send_counts=send_counts,
        send_disps=send_disps,
        total_send=total_send,
        nnz_to_recv_pos=None,
        lb=lb,
        ub=ub
    )


def precompute_nnz_mapping(col_indexes: np.ndarray, gcd: GraphCommData) -> np.ndarray:
    """
    Precompute O(nnz) mapping from NNZ index to recv buffer position.
    
    This trades O(nnz) * 8 bytes for O(1) lookup instead of O(log N) binary search.
    
    Returns:
        nnz_to_recv_pos: Array where nnz_to_recv_pos[j] = -1 if col_indexes[j] is local,
                         else the position in recv_buf after Neighbor_alltoallv.
    """
    local_nnz = len(col_indexes)
    nnz_to_recv_pos = np.full(local_nnz, -1, dtype=np.int64)
    
    lb, ub = gcd.lb, gcd.ub
    
    for j in range(local_nnz):
        col = col_indexes[j]
        if col < lb or col >= ub:
            # Binary search in sorted recv_indices
            pos = np.searchsorted(gcd.recv_indices_sorted, col)
            # Map to neighbor-order position
            nnz_to_recv_pos[j] = gcd.recv_perm[pos]
    
    return nnz_to_recv_pos


def spmv_unit_valued(row_starts: np.ndarray, col_indexes: np.ndarray,
                     u: np.ndarray, gcd: GraphCommData,
                     partition_table: np.ndarray, rank: int,
                     scalar: complex = 1.0+0j) -> np.ndarray:
    """
    Distributed SpMV for unit-valued matrix using graph communicator.
    
    Memory: O(unique_remote) for communication, O(local_nnz) temporary for values.
    Uses binary search for remote column lookup (or O(1) if nnz_to_recv_pos precomputed).
    
    
    Args:
        row_starts: Local CSR row pointers
        col_indexes: Local CSR column indices (global)
        u: Local portion of input vector
        gcd: Graph communicator data
        partition_table: Partition table
        rank: This rank's ID
        scalar: Scalar multiplier (e.g., -1j)
    
    Returns:
        v: Local portion of output vector
    """
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    n_local = ub - lb
    
    v = np.zeros(n_local, dtype=np.complex128)
    
    # Step 1: Pack send buffer
    send_buf = np.empty(max(gcd.total_send, 1), dtype=np.complex128)
    for i in range(gcd.total_send):
        send_buf[i] = u[gcd.send_offsets[i]]
    
    # Step 2: Exchange via Neighbor_alltoallv
    recv_buf = np.empty(max(gcd.total_recv, 1), dtype=np.complex128)
    gcd.graph_comm.Neighbor_alltoallv(
        [send_buf, (gcd.send_counts, gcd.send_disps)],
        [recv_buf, (gcd.recv_counts, gcd.recv_disps)]
    )
    
    # Step 3: Compute SpMV with binary search for remote columns
    for i in range(n_local):
        row_sum = 0.0 + 0.0j
        for j in range(row_starts[i], row_starts[i + 1]):
            col = col_indexes[j]
            if lb <= col < ub:
                # Local access
                row_sum += u[col - lb]
            else:
                # Remote: binary search in sorted recv_indices
                pos = np.searchsorted(gcd.recv_indices_sorted, col)
                # recv_perm maps sorted position back to neighbor-order position
                recv_pos = gcd.recv_perm[pos]
                row_sum += recv_buf[recv_pos]
        v[i] = scalar * row_sum
    
    return v


def spmv_optimized(row_starts: np.ndarray, col_indexes: np.ndarray,
                   u: np.ndarray, gcd: GraphCommData,
                   partition_table: np.ndarray, rank: int,
                   scalar: complex = 1.0+0j,
                   all_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Optimized SpMV that pre-gathers all values to avoid binary search per NNZ.
    
    Trades O(nnz) temporary memory for O(1) lookup per NNZ.
    The all_values buffer can be reused across SpMV calls.
    
    Args:
        all_values: Pre-allocated buffer of size >= local_nnz, or None to allocate
    """
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    n_local = ub - lb
    local_nnz = row_starts[n_local]
    
    # Allocate or reuse all_values buffer
    if all_values is None:
        all_values = np.empty(local_nnz, dtype=np.complex128)
    
    # Step 1: Pack send buffer
    send_buf = np.empty(max(gcd.total_send, 1), dtype=np.complex128)
    for i in range(gcd.total_send):
        send_buf[i] = u[gcd.send_offsets[i]]
    
    # Step 2: Exchange
    recv_buf = np.empty(max(gcd.total_recv, 1), dtype=np.complex128)
    gcd.graph_comm.Neighbor_alltoallv(
        [send_buf, (gcd.send_counts, gcd.send_disps)],
        [recv_buf, (gcd.recv_counts, gcd.recv_disps)]
    )
    
    # Step 3: Pre-gather all column values into contiguous array
    for j in range(local_nnz):
        col = col_indexes[j]
        if lb <= col < ub:
            all_values[j] = u[col - lb]
        else:
            # Binary search once per unique remote column
            pos = np.searchsorted(gcd.recv_indices_sorted, col)
            recv_pos = gcd.recv_perm[pos]
            all_values[j] = recv_buf[recv_pos]
    
    # Step 4: Compute SpMV with direct array access
    v = np.zeros(n_local, dtype=np.complex128)
    for i in range(n_local):
        row_sum = 0.0 + 0.0j
        for j in range(row_starts[i], row_starts[i + 1]):
            row_sum += all_values[j]
        v[i] = scalar * row_sum
    
    return v


def spmv_with_precomputed_map(row_starts: np.ndarray, col_indexes: np.ndarray,
                               u: np.ndarray, gcd: GraphCommData,
                               nnz_to_recv_pos: np.ndarray,
                               scalar: complex = 1.0+0j,
                               send_buf: Optional[np.ndarray] = None,
                               recv_buf: Optional[np.ndarray] = None) -> np.ndarray:
    """
    SpMV with O(nnz) precomputed mapping for O(1) lookup.
    
    Setup cost: O(nnz) * 8 bytes for nnz_to_recv_pos
    SpMV cost: O(1) lookup per NNZ (no binary search)
    
    This is the fastest option when memory permits. Buffers can be reused.
    """
    lb, ub = gcd.lb, gcd.ub
    n_local = ub - lb
    local_nnz = row_starts[n_local]
    
    # Reuse or allocate buffers
    if send_buf is None:
        send_buf = np.empty(max(gcd.total_send, 1), dtype=np.complex128)
    if recv_buf is None:
        recv_buf = np.empty(max(gcd.total_recv, 1), dtype=np.complex128)
    
    # Pack send buffer
    for i in range(gcd.total_send):
        send_buf[i] = u[gcd.send_offsets[i]]
    
    # Exchange
    gcd.graph_comm.Neighbor_alltoallv(
        [send_buf, (gcd.send_counts, gcd.send_disps)],
        [recv_buf, (gcd.recv_counts, gcd.recv_disps)]
    )
    
    # Compute SpMV with O(1) lookup via precomputed map
    v = np.zeros(n_local, dtype=np.complex128)
    for i in range(n_local):
        row_sum = 0.0 + 0.0j
        for j in range(row_starts[i], row_starts[i + 1]):
            recv_pos = nnz_to_recv_pos[j]
            if recv_pos < 0:
                # Local
                col = col_indexes[j]
                row_sum += u[col - lb]
            else:
                # Remote - O(1) lookup
                row_sum += recv_buf[recv_pos]
        v[i] = scalar * row_sum
    
    return v


@dataclass
class ChunkedGraphCommData:
    """
    Chunked communication data - bounds memory usage for very large systems.
    
    Instead of storing all recv_indices at once, we process in chunks.
    Each chunk exchanges a bounded number of elements per neighbor.
    """
    graph_comm: MPI.Comm
    in_neighbors: np.ndarray
    out_neighbors: np.ndarray
    
    # Per-neighbor lists of remote columns (not concatenated)
    recv_cols_per_neighbor: List[np.ndarray]  # What we need from each out_neighbor
    send_cols_per_neighbor: List[np.ndarray]  # What each in_neighbor needs from us
    
    # Partition info
    lb: int
    ub: int
    
    # Chunk configuration
    chunk_size: int  # Max elements per neighbor per chunk


def setup_chunked_graph_comm(row_starts: np.ndarray, col_indexes: np.ndarray,
                              partition_table: np.ndarray, comm: MPI.Comm,
                              chunk_size: int = 1024*1024) -> ChunkedGraphCommData:
    """
    Set up graph communicator for chunked communication.
    
    Unlike setup_graph_comm, this doesn't concatenate all indices - 
    it keeps them per-neighbor for chunked processing.
    
    Args:
        chunk_size: Max elements to exchange per neighbor per chunk (default 1M = 16MB)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    
    # Find unique remote columns grouped by owner
    remote_by_rank: dict = {r: set() for r in range(size)}
    
    for col in col_indexes:
        if col < lb or col >= ub:
            owner = find_owner(col, partition_table)
            remote_by_rank[owner].add(col)
    
    # Identify neighbors
    out_neighbors = sorted([r for r in range(size) 
                           if r != rank and remote_by_rank[r]])
    
    all_out = comm.allgather(out_neighbors)
    in_neighbors = sorted([r for r in range(size) if rank in all_out[r]])
    
    out_neighbors_arr = np.array(out_neighbors, dtype=np.intc)
    in_neighbors_arr = np.array(in_neighbors, dtype=np.intc)
    
    # Create graph communicator
    graph_comm = comm.Create_dist_graph_adjacent(
        in_neighbors_arr, out_neighbors_arr,
        sourceweights=None, destweights=None,
        info=MPI.INFO_NULL, reorder=False
    )
    
    # Store per-neighbor recv columns (sorted for binary search)
    recv_cols_per_neighbor = [np.array(sorted(remote_by_rank[r]), dtype=np.int64) 
                               for r in out_neighbors]
    
    # Exchange counts to set up send side
    recv_counts = np.array([len(cols) for cols in recv_cols_per_neighbor], dtype=np.intc)
    send_counts = np.zeros(len(in_neighbors), dtype=np.intc)
    graph_comm.Neighbor_alltoall(recv_counts, send_counts)
    
    # Exchange all indices to know what neighbors need from us
    # (This is done once during setup, not per SpMV)
    recv_disps = np.zeros(len(out_neighbors), dtype=np.intc)
    send_disps = np.zeros(len(in_neighbors), dtype=np.intc)
    for i in range(1, len(out_neighbors)):
        recv_disps[i] = recv_disps[i-1] + recv_counts[i-1]
    for i in range(1, len(in_neighbors)):
        send_disps[i] = send_disps[i-1] + send_counts[i-1]
    
    total_recv = int(np.sum(recv_counts))
    total_send = int(np.sum(send_counts))
    
    all_recv = np.concatenate(recv_cols_per_neighbor) if recv_cols_per_neighbor else np.array([], dtype=np.int64)
    all_send = np.zeros(max(total_send, 1), dtype=np.int64)
    
    graph_comm.Neighbor_alltoallv(
        [all_recv, (recv_counts, recv_disps)],
        [all_send, (send_counts, send_disps)]
    )
    
    # Split send indices per neighbor
    send_cols_per_neighbor = []
    for i, count in enumerate(send_counts):
        start = send_disps[i]
        send_cols_per_neighbor.append(all_send[start:start+count].copy())
    
    return ChunkedGraphCommData(
        graph_comm=graph_comm,
        in_neighbors=in_neighbors_arr,
        out_neighbors=out_neighbors_arr,
        recv_cols_per_neighbor=recv_cols_per_neighbor,
        send_cols_per_neighbor=send_cols_per_neighbor,
        lb=lb,
        ub=ub,
        chunk_size=chunk_size
    )


def spmv_chunked(row_starts: np.ndarray, col_indexes: np.ndarray,
                 u: np.ndarray, cgcd: ChunkedGraphCommData,
                 scalar: complex = 1.0+0j) -> np.ndarray:
    """
    SpMV with chunked communication for bounded memory usage.
    
    Memory per chunk: O(chunk_size * n_neighbors) instead of O(total_unique_remote)
    
    At 35 qubits with 4096 ranks and chunk_size=1M:
    - Without chunking: recv_buf = 128 MB
    - With chunking: recv_buf = 16 MB per chunk
    
    Trade-off: Multiple Neighbor_alltoallv calls per SpMV.
    """
    lb, ub = cgcd.lb, cgcd.ub
    n_local = ub - lb
    chunk_size = cgcd.chunk_size
    
    v = np.zeros(n_local, dtype=np.complex128)
    
    # First: accumulate local contributions (no communication)
    for i in range(n_local):
        start = row_starts[i]
        end = row_starts[i + 1]
        row_sum = 0.0 + 0.0j
        for j in range(start, end):
            col = col_indexes[j]
            if lb <= col < ub:
                row_sum += u[col - lb]
        v[i] = row_sum  # Will multiply by scalar at the end
    
    # Determine max chunks needed
    max_recv = max((len(cols) for cols in cgcd.recv_cols_per_neighbor), default=0)
    max_send = max((len(cols) for cols in cgcd.send_cols_per_neighbor), default=0)
    n_chunks = max(1, (max(max_recv, max_send) + chunk_size - 1) // chunk_size)
    
    n_out = len(cgcd.out_neighbors)
    n_in = len(cgcd.in_neighbors)
    
    # Process in chunks
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size
        
        # Build chunk counts and data
        recv_chunk_counts = np.zeros(n_out, dtype=np.intc)
        send_chunk_counts = np.zeros(n_in, dtype=np.intc)
        
        recv_chunk_indices = []
        for i, cols in enumerate(cgcd.recv_cols_per_neighbor):
            chunk_cols = cols[chunk_start:chunk_end]
            recv_chunk_counts[i] = len(chunk_cols)
            recv_chunk_indices.append(chunk_cols)
        
        send_chunk_indices = []
        for i, cols in enumerate(cgcd.send_cols_per_neighbor):
            chunk_cols = cols[chunk_start:chunk_end]
            send_chunk_counts[i] = len(chunk_cols)
            send_chunk_indices.append(chunk_cols)
        
        # Skip if nothing to exchange this chunk
        if np.sum(recv_chunk_counts) == 0 and np.sum(send_chunk_counts) == 0:
            continue
        
        # Build displacements
        recv_chunk_disps = np.zeros(n_out, dtype=np.intc)
        send_chunk_disps = np.zeros(n_in, dtype=np.intc)
        for i in range(1, n_out):
            recv_chunk_disps[i] = recv_chunk_disps[i-1] + recv_chunk_counts[i-1]
        for i in range(1, n_in):
            send_chunk_disps[i] = send_chunk_disps[i-1] + send_chunk_counts[i-1]
        
        total_recv_chunk = int(np.sum(recv_chunk_counts))
        total_send_chunk = int(np.sum(send_chunk_counts))
        
        # Pack send buffer for this chunk
        send_buf = np.empty(max(total_send_chunk, 1), dtype=np.complex128)
        idx = 0
        for i, cols in enumerate(send_chunk_indices):
            for col in cols:
                send_buf[idx] = u[col - lb]
                idx += 1
        
        # Exchange values
        recv_buf = np.empty(max(total_recv_chunk, 1), dtype=np.complex128)
        cgcd.graph_comm.Neighbor_alltoallv(
            [send_buf, (send_chunk_counts, send_chunk_disps)],
            [recv_buf, (recv_chunk_counts, recv_chunk_disps)]
        )
        
        # Build sorted array for binary search (Fortran-compatible)
        # Concatenate all chunk columns in sorted order with their values
        all_chunk_cols = np.concatenate(recv_chunk_indices) if recv_chunk_indices else np.array([], dtype=np.int64)
        
        # Sort for binary search
        if len(all_chunk_cols) > 0:
            sort_perm = np.argsort(all_chunk_cols)
            chunk_cols_sorted = all_chunk_cols[sort_perm]
            chunk_vals_sorted = recv_buf[sort_perm]
        else:
            chunk_cols_sorted = np.array([], dtype=np.int64)
            chunk_vals_sorted = np.array([], dtype=np.complex128)
        
        # Accumulate remote contributions from this chunk using binary search
        for i in range(n_local):
            start = row_starts[i]
            end = row_starts[i + 1]
            for j in range(start, end):
                col = col_indexes[j]
                if col < lb or col >= ub:
                    # Binary search in chunk's sorted columns
                    pos = np.searchsorted(chunk_cols_sorted, col)
                    if pos < len(chunk_cols_sorted) and chunk_cols_sorted[pos] == col:
                        v[i] += chunk_vals_sorted[pos]
    
    # Apply scalar
    v *= scalar
    
    return v


def spmv_sorted_rows(row_starts: np.ndarray, col_indexes: np.ndarray,
                     u: np.ndarray, gcd: GraphCommData,
                     scalar: complex = 1.0+0j,
                     send_buf: Optional[np.ndarray] = None,
                     recv_buf: Optional[np.ndarray] = None) -> np.ndarray:
    """
    SpMV using sorted column indices within each row.
    
    Memory: O(unique_remote) - NO O(nnz) mapping needed!
    Cost per NNZ: O(log(unique_remote)) binary search in recv_indices_sorted
    
    Requires: col_indexes sorted within each row (see sort_csr_rows or build with sort_rows=True)
    
    This is the most memory-efficient non-chunked approach.
    For bounded memory at extreme scale, use spmv_chunked instead.
    """
    lb, ub = gcd.lb, gcd.ub
    n_local = ub - lb
    
    # Reuse or allocate buffers
    if send_buf is None:
        send_buf = np.empty(max(gcd.total_send, 1), dtype=np.complex128)
    if recv_buf is None:
        recv_buf = np.empty(max(gcd.total_recv, 1), dtype=np.complex128)
    
    # Pack send buffer
    for i in range(gcd.total_send):
        send_buf[i] = u[gcd.send_offsets[i]]
    
    # Exchange
    gcd.graph_comm.Neighbor_alltoallv(
        [send_buf, (gcd.send_counts, gcd.send_disps)],
        [recv_buf, (gcd.recv_counts, gcd.recv_disps)]
    )
    
    # Compute SpMV using sorted columns
    # For each row, columns are sorted, so we can:
    # 1. Binary search for first column >= lb (start of local range)
    # 2. Binary search for first column >= ub (end of local range)
    # 3. Process remote columns before lb, local columns [lb,ub), remote columns >= ub
    
    v = np.zeros(n_local, dtype=np.complex128)
    
    for i in range(n_local):
        start = row_starts[i]
        end = row_starts[i + 1]
        row_cols = col_indexes[start:end]
        
        row_sum = 0.0 + 0.0j
        
        # Find boundaries in sorted row
        # local_start: first column >= lb
        # local_end: first column >= ub
        local_start = np.searchsorted(row_cols, lb)
        local_end = np.searchsorted(row_cols, ub)
        
        # Remote columns < lb
        for j in range(local_start):
            col = row_cols[j]
            pos = np.searchsorted(gcd.recv_indices_sorted, col)
            recv_pos = gcd.recv_perm[pos]
            row_sum += recv_buf[recv_pos]
        
        # Local columns [lb, ub)
        for j in range(local_start, local_end):
            col = row_cols[j]
            row_sum += u[col - lb]
        
        # Remote columns >= ub
        for j in range(local_end, len(row_cols)):
            col = row_cols[j]
            pos = np.searchsorted(gcd.recv_indices_sorted, col)
            recv_pos = gcd.recv_perm[pos]
            row_sum += recv_buf[recv_pos]
        
        v[i] = scalar * row_sum
    
    return v


def estimate_memory(n_qubits: int, n_ranks: int) -> dict:
    """Estimate memory usage for different approaches."""
    n_rows = 2 ** n_qubits
    local_rows = n_rows // n_ranks
    local_nnz = local_rows * n_qubits
    
    # Assume roughly local_rows unique remote columns (conservative for hypercube)
    unique_remote_approx = local_rows
    
    # Full pre-computed (current QuOp_MPI approach)
    full_values = local_nnz * 16        # complex128 values array
    full_comm_indices = local_nnz * 8   # col_to_remote_pos
    full_total = full_values + full_comm_indices
    
    # Unit-valued with O(nnz) mapping (old approach)
    old_no_values = 0
    old_nnz_mapping = local_nnz * 8     # col_to_remote_pos per nnz
    old_total = old_no_values + old_nnz_mapping
    
    # Unit-valued with O(unique_remote) (new approach)
    new_no_values = 0
    new_unique_mapping = unique_remote_approx * 8 * 3  # sorted, perm, send_offsets
    new_total = new_no_values + new_unique_mapping
    
    # Temporary during SpMV
    spmv_temp = local_nnz * 16  # all_values gather buffer (reusable)
    
    return {
        'n_qubits': n_qubits,
        'n_ranks': n_ranks,
        'local_rows': local_rows,
        'local_nnz': local_nnz,
        'unique_remote': unique_remote_approx,
        'full_precomputed_gb': full_total / 1e9,
        'old_nnz_mapping_gb': old_total / 1e9,
        'new_unique_mapping_gb': new_total / 1e9,
        'spmv_temp_gb': spmv_temp / 1e9,
        'savings_vs_full': full_total / new_total if new_total > 0 else float('inf'),
        'savings_vs_old': old_total / new_total if new_total > 0 else float('inf'),
    }


def test_correctness(comm: MPI.Comm, n_qubits: int = 10):
    """Test that SpMV produces correct results."""
    rank = comm.Get_rank()
    
    partition_table = generate_partition_table(2**n_qubits, comm)
    row_starts, col_indexes = build_hypercube_csr(n_qubits, partition_table, rank)
    
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    n_local = ub - lb
    
    gcd = setup_graph_comm(row_starts, col_indexes, partition_table, comm)
    
    # Test 1: All-ones vector
    u = np.ones(n_local, dtype=np.complex128)
    v = spmv_unit_valued(row_starts, col_indexes, u, gcd, partition_table, rank)
    
    expected = np.full(n_local, n_qubits, dtype=np.complex128)
    passed = np.allclose(v, expected)
    
    if rank == 0:
        print(f"Test 1 (all ones): {'PASS' if passed else 'FAIL'}")
    
    # Test 2: Index vector
    u2 = np.arange(lb, ub, dtype=np.complex128)
    v2 = spmv_unit_valued(row_starts, col_indexes, u2, gcd, partition_table, rank)
    
    expected2 = np.zeros(n_local, dtype=np.complex128)
    for i in range(n_local):
        global_row = lb + i
        for k in range(n_qubits):
            expected2[i] += global_row ^ (1 << k)
    
    passed2 = np.allclose(v2, expected2)
    if rank == 0:
        print(f"Test 2 (index vector): {'PASS' if passed2 else 'FAIL'}")
    
    # Test 3: Scalar multiplier
    v3 = spmv_unit_valued(row_starts, col_indexes, u, gcd, partition_table, rank, 
                          scalar=-1j)
    expected3 = -1j * expected
    passed3 = np.allclose(v3, expected3)
    if rank == 0:
        print(f"Test 3 (scalar -i): {'PASS' if passed3 else 'FAIL'}")
    
    # Test 4: Optimized version matches basic version
    v4 = spmv_optimized(row_starts, col_indexes, u2, gcd, partition_table, rank)
    passed4 = np.allclose(v4, v2)
    if rank == 0:
        print(f"Test 4 (optimized matches basic): {'PASS' if passed4 else 'FAIL'}")
    
    # Test 5: Precomputed O(nnz) mapping version
    nnz_to_recv_pos = precompute_nnz_mapping(col_indexes, gcd)
    v5 = spmv_with_precomputed_map(row_starts, col_indexes, u2, gcd, nnz_to_recv_pos)
    passed5 = np.allclose(v5, v2)
    if rank == 0:
        print(f"Test 5 (precomputed map matches basic): {'PASS' if passed5 else 'FAIL'}")
    
    # Test 6: Sorted rows version (most memory efficient)
    v6 = spmv_sorted_rows(row_starts, col_indexes, u2, gcd)
    passed6 = np.allclose(v6, v2)
    if rank == 0:
        print(f"Test 6 (sorted rows matches basic): {'PASS' if passed6 else 'FAIL'}")
    
    # Test 7: Chunked communication version (bounded memory)
    cgcd = setup_chunked_graph_comm(row_starts, col_indexes, partition_table, comm, 
                                     chunk_size=1024)  # Small chunks for testing
    v7 = spmv_chunked(row_starts, col_indexes, u2, cgcd)
    passed7 = np.allclose(v7, v2)
    if rank == 0:
        print(f"Test 7 (chunked comm matches basic): {'PASS' if passed7 else 'FAIL'}")
    
    # Test 8: Chunked with scalar
    v8 = spmv_chunked(row_starts, col_indexes, u, cgcd, scalar=-1j)
    passed8 = np.allclose(v8, expected3)
    if rank == 0:
        print(f"Test 8 (chunked with scalar): {'PASS' if passed8 else 'FAIL'}")
    
    return all([passed, passed2, passed3, passed4, passed5, passed6, passed7, passed8])


def benchmark(comm: MPI.Comm, n_qubits: int, chunk_size: int, n_iters: int):
    """Benchmark SpMV performance."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"\n=== Benchmark: {n_qubits} qubits, {size} ranks, {n_iters} iterations ===")
    
    partition_table = generate_partition_table(2**n_qubits, comm)
    
    # Setup timing
    comm.Barrier()
    t_start = time.perf_counter()
    
    row_starts, col_indexes = build_hypercube_csr(n_qubits, partition_table, rank)
    gcd = setup_graph_comm(row_starts, col_indexes, partition_table, comm)
    
    comm.Barrier()
    t_setup = time.perf_counter() - t_start
    
    lb = partition_table[rank]
    ub = partition_table[rank + 1]
    n_local = ub - lb
    local_nnz = row_starts[n_local]
    
    # Pre-allocate reusable buffers
    all_values = np.empty(local_nnz, dtype=np.complex128)
    send_buf = np.empty(max(gcd.total_send, 1), dtype=np.complex128)
    recv_buf = np.empty(max(gcd.total_recv, 1), dtype=np.complex128)
    
    # Precompute O(nnz) mapping for fastest version
    comm.Barrier()
    t_map_start = time.perf_counter()
    nnz_to_recv_pos = precompute_nnz_mapping(col_indexes, gcd)
    comm.Barrier()
    t_map = time.perf_counter() - t_map_start
    
    u = np.random.randn(n_local) + 1j * np.random.randn(n_local)
    u = u.astype(np.complex128)
    
    # === Benchmark: Binary search version ===
    for _ in range(2):  # Warmup
        v = spmv_optimized(row_starts, col_indexes, u, gcd, partition_table, rank,
                           all_values=all_values)
    
    comm.Barrier()
    t_start = time.perf_counter()
    for _ in range(n_iters):
        v = spmv_optimized(row_starts, col_indexes, u, gcd, partition_table, rank,
                           all_values=all_values)
    comm.Barrier()
    t_bsearch = time.perf_counter() - t_start
    
    # === Benchmark: Precomputed map version ===
    for _ in range(2):  # Warmup
        v2 = spmv_with_precomputed_map(row_starts, col_indexes, u, gcd, nnz_to_recv_pos,
                                       send_buf=send_buf, recv_buf=recv_buf)
    
    comm.Barrier()
    t_start = time.perf_counter()
    for _ in range(n_iters):
        v2 = spmv_with_precomputed_map(row_starts, col_indexes, u, gcd, nnz_to_recv_pos,
                                       send_buf=send_buf, recv_buf=recv_buf)
    comm.Barrier()
    t_precomp = time.perf_counter() - t_start
    
    # === Benchmark: Sorted rows version (most memory efficient) ===
    for _ in range(2):  # Warmup
        v3 = spmv_sorted_rows(row_starts, col_indexes, u, gcd,
                              send_buf=send_buf, recv_buf=recv_buf)
    
    comm.Barrier()
    t_start = time.perf_counter()
    for _ in range(n_iters):
        v3 = spmv_sorted_rows(row_starts, col_indexes, u, gcd,
                              send_buf=send_buf, recv_buf=recv_buf)
    comm.Barrier()
    t_sorted = time.perf_counter() - t_start
    
    # === Benchmark: Chunked communication (bounded memory) ===
    cgcd = setup_chunked_graph_comm(row_starts, col_indexes, partition_table, comm,
                                     chunk_size=chunk_size)
    
    for _ in range(2):  # Warmup
        v4 = spmv_chunked(row_starts, col_indexes, u, cgcd)
    
    comm.Barrier()
    t_start = time.perf_counter()
    for _ in range(n_iters):
        v4 = spmv_chunked(row_starts, col_indexes, u, cgcd)
    comm.Barrier()
    t_chunked = time.perf_counter() - t_start
    
    # Compute chunk memory usage
    max_per_neighbor = max((len(cols) for cols in cgcd.recv_cols_per_neighbor), default=0)
    n_chunks = max(1, (max_per_neighbor + chunk_size - 1) // chunk_size)
    chunk_buf_size = min(chunk_size, max_per_neighbor) * len(cgcd.out_neighbors) * 16 / 1e6
    
    if rank == 0:
        mem = estimate_memory(n_qubits, size)
        print(f"Setup time:           {t_setup*1000:.2f} ms")
        print(f"Precompute map time:  {t_map*1000:.2f} ms")
        print(f"")
        print(f"SpMV (gather+loop):   {t_bsearch/n_iters*1000:.2f} ms/iter  [O(nnz) temp buffer]")
        print(f"SpMV (precomp map):   {t_precomp/n_iters*1000:.2f} ms/iter  [O(nnz) mapping]")
        print(f"SpMV (sorted rows):   {t_sorted/n_iters*1000:.2f} ms/iter   [O(unique) only!]")
        print(f"SpMV (chunked):       {t_chunked/n_iters*1000:.2f} ms/iter   [bounded memory]")
        print(f"")
        print(f"Speedup precomp vs gather: {t_bsearch/t_precomp:.2f}x")
        print(f"Slowdown sorted vs precomp: {t_sorted/t_precomp:.2f}x")
        print(f"Slowdown chunked vs sorted: {t_chunked/t_sorted:.2f}x")
        print(f"")
        print(f"Chunked: {n_chunks} chunks, {chunk_buf_size:.2f} MB buffer/chunk")
        print(f"")
        print(f"Local rows:     {n_local:,}")
        print(f"Local NNZ:      {local_nnz:,}")
        print(f"Total recv:     {gcd.total_recv:,}")
        print(f"Total send:     {gcd.total_send:,}")
        print(f"Neighbors:      in={len(gcd.in_neighbors)}, out={len(gcd.out_neighbors)}")
        print(f"\nMemory per rank:")
        print(f"  Full precomputed:     {mem['full_precomputed_gb']:.4f} GB")
        print(f"  O(unique) storage:    {mem['new_unique_mapping_gb']:.4f} GB")
        print(f"  + nnz_to_recv_pos:    {local_nnz * 8 / 1e9:.4f} GB (optional for speed)")
        print(f"  + gather buffer:      {mem['spmv_temp_gb']:.4f} GB (optional, reusable)")
        print(f"  Minimum (sorted):     {mem['new_unique_mapping_gb']:.4f} GB")
        print(f"  Savings vs full:      {mem['savings_vs_full']:.1f}x")


def print_scaling_analysis():
    """Print memory scaling analysis for large systems."""
    print("\n=== Memory Scaling Analysis ===")
    print("Hypercube adjacency with unit values\n")
    
    print(f"{'Qubits':>8} {'Ranks':>8} {'Full (GB)':>12} {'Old O(nnz)':>12} {'New O(uniq)':>12} {'Savings':>10}")
    print("-" * 70)
    
    for n_qubits in [20, 24, 28, 30, 32, 34]:
        for n_ranks in [64, 256, 1024, 4096]:
            if 2**n_qubits // n_ranks < 1:
                continue
            mem = estimate_memory(n_qubits, n_ranks)
            print(f"{n_qubits:>8} {n_ranks:>8} {mem['full_precomputed_gb']:>12.2f} "
                  f"{mem['old_nnz_mapping_gb']:>12.2f} {mem['new_unique_mapping_gb']:>12.4f} "
                  f"{mem['savings_vs_old']:>9.0f}x")
        print()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Parse command line args
    n_qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    if rank == 0:
        print(f"Chunked SpMV with Graph Communicators")
        print(f"Using O(unique_remote) storage with binary search")
        print(f"MPI ranks: {comm.Get_size()}")
    
    # Run correctness tests
    if test_correctness(comm, min(n_qubits, 14)):
        if rank == 0:
            print("All correctness tests passed!")
    
    # Benchmark
    benchmark(comm, n_qubits, chunk_size, n_iters)
    
    # Print scaling analysis on rank 0
    if rank == 0:
        print_scaling_analysis()
