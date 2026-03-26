// HIP SpMV kernels for distributed sparse matrix-vector multiplication
// Two-phase approach: LOCAL computation overlaps with MPI, then REMOTE contributions

#include "hip_common.hpp"

//==============================================================================
// TWO-PHASE DISTRIBUTED SpMV KERNELS
// These match the CPU implementation in sparse.f90:
//   Phase 1 (LOCAL): Compute contributions from local columns while MPI runs
//   Phase 2 (REMOTE): Add contributions from received remote data
//==============================================================================

//------------------------------------------------------------------------------
// Phase 1: LOCAL contributions only (run while MPI communication proceeds)
// Weighted version with explicit edge values
//------------------------------------------------------------------------------
__global__ void spmv_local_weighted(long* row_starts, long* col_indexes, hipDoubleComplex* values,
                                    hipDoubleComplex* x_local, hipDoubleComplex* y, long lb, long ub,
                                    long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long start_j = row_starts[i];
    long end_j = row_starts[i + 1] - 1;

    hipDoubleComplex row_sum = {0.0, 0.0};

    if (start_j <= end_j) {
      // Find local column range using binary search
      long local_start = lower_bound_dev(col_indexes, start_j, end_j + 1, lb);
      long local_end = upper_bound_dev(col_indexes, start_j, end_j + 1, ub) - 1;

      // Sum contributions from local columns only
      for (long j = local_start; j <= local_end; j++) {
        long col = col_indexes[j];
        long local_col = col - lb; // Convert to 0-based local index
        hipDoubleComplex val = values[j];
        hipDoubleComplex xj = x_local[local_col];
        row_sum.x += val.x * xj.x - val.y * xj.y;
        row_sum.y += val.x * xj.y + val.y * xj.x;
      }
    }

    y[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Phase 1: LOCAL contributions only - Unit weight version
//------------------------------------------------------------------------------
__global__ void spmv_local_unit(long* row_starts, long* col_indexes, hipDoubleComplex* x_local,
                                hipDoubleComplex* y, long lb, long ub, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long start_j = row_starts[i];
    long end_j = row_starts[i + 1] - 1;

    hipDoubleComplex row_sum = {0.0, 0.0};

    if (start_j <= end_j) {
      long local_start = lower_bound_dev(col_indexes, start_j, end_j + 1, lb);
      long local_end = upper_bound_dev(col_indexes, start_j, end_j + 1, ub) - 1;

      for (long j = local_start; j <= local_end; j++) {
        long col = col_indexes[j];
        long local_col = col - lb;
        row_sum.x += x_local[local_col].x;
        row_sum.y += x_local[local_col].y;
      }
    }

    y[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Phase 2: REMOTE contributions (run after MPI communication completes)
// Adds remote contributions to y (which already has local contributions)
// Uses hash table to find position in recv_buf for each remote column
// Weighted version
//------------------------------------------------------------------------------
__global__ void spmv_remote_weighted(long* row_starts, long* col_indexes, hipDoubleComplex* values,
                                     hipDoubleComplex* recv_buf_sorted, long* hash_keys, long* hash_vals,
                                     long hash_size, hipDoubleComplex* y, hipDoubleComplex scalar, long lb,
                                     long ub, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long start_j = row_starts[i];
    long end_j = row_starts[i + 1] - 1;

    hipDoubleComplex row_sum = y[i]; // Start with local contributions

    if (start_j <= end_j) {
      // Check if row has any remote columns
      if (col_indexes[start_j] >= lb && col_indexes[end_j] <= ub) {
        // All columns are local, just apply scalar
        y[i].x = scalar.x * row_sum.x - scalar.y * row_sum.y;
        y[i].y = scalar.x * row_sum.y + scalar.y * row_sum.x;
        continue;
      }

      long local_start = lower_bound_dev(col_indexes, start_j, end_j + 1, lb);
      long local_end = upper_bound_dev(col_indexes, start_j, end_j + 1, ub) - 1;

      // Add contributions from remote columns BEFORE local range
      for (long j = start_j; j < local_start; j++) {
        long col = col_indexes[j];
        long sorted_pos = hash_lookup_dev(col, hash_keys, hash_vals, hash_size);
        if (sorted_pos >= 0) {
          hipDoubleComplex val = values[j];
          hipDoubleComplex xj = recv_buf_sorted[sorted_pos]; // 0-based position
          row_sum.x += val.x * xj.x - val.y * xj.y;
          row_sum.y += val.x * xj.y + val.y * xj.x;
        }
      }

      // Add contributions from remote columns AFTER local range
      for (long j = local_end + 1; j <= end_j; j++) {
        long col = col_indexes[j];
        long sorted_pos = hash_lookup_dev(col, hash_keys, hash_vals, hash_size);
        if (sorted_pos >= 0) {
          hipDoubleComplex val = values[j];
          hipDoubleComplex xj = recv_buf_sorted[sorted_pos];
          row_sum.x += val.x * xj.x - val.y * xj.y;
          row_sum.y += val.x * xj.y + val.y * xj.x;
        }
      }
    }

    // Apply scalar
    y[i].x = scalar.x * row_sum.x - scalar.y * row_sum.y;
    y[i].y = scalar.x * row_sum.y + scalar.y * row_sum.x;
  }
}

//------------------------------------------------------------------------------
// Phase 2: REMOTE contributions - Unit weight version
//------------------------------------------------------------------------------
__global__ void spmv_remote_unit(long* row_starts, long* col_indexes, hipDoubleComplex* recv_buf_sorted,
                                 long* hash_keys, long* hash_vals, long hash_size, hipDoubleComplex* y,
                                 hipDoubleComplex scalar, long lb, long ub, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long start_j = row_starts[i];
    long end_j = row_starts[i + 1] - 1;

    hipDoubleComplex row_sum = y[i];

    if (start_j <= end_j) {
      if (col_indexes[start_j] >= lb && col_indexes[end_j] <= ub) {
        y[i].x = scalar.x * row_sum.x - scalar.y * row_sum.y;
        y[i].y = scalar.x * row_sum.y + scalar.y * row_sum.x;
        continue;
      }

      long local_start = lower_bound_dev(col_indexes, start_j, end_j + 1, lb);
      long local_end = upper_bound_dev(col_indexes, start_j, end_j + 1, ub) - 1;

      for (long j = start_j; j < local_start; j++) {
        long col = col_indexes[j];
        long sorted_pos = hash_lookup_dev(col, hash_keys, hash_vals, hash_size);
        if (sorted_pos >= 0) {
          hipDoubleComplex xj = recv_buf_sorted[sorted_pos]; // 0-based position
          row_sum.x += xj.x;
          row_sum.y += xj.y;
        }
      }

      for (long j = local_end + 1; j <= end_j; j++) {
        long col = col_indexes[j];
        long sorted_pos = hash_lookup_dev(col, hash_keys, hash_vals, hash_size);
        if (sorted_pos >= 0) {
          hipDoubleComplex xj = recv_buf_sorted[sorted_pos];
          row_sum.x += xj.x;
          row_sum.y += xj.y;
        }
      }
    }

    y[i].x = scalar.x * row_sum.x - scalar.y * row_sum.y;
    y[i].y = scalar.x * row_sum.y + scalar.y * row_sum.x;
  }
}

//------------------------------------------------------------------------------
// Reorder recv_buf according to sort_perm (sort_perm is 0-based)
//------------------------------------------------------------------------------
__global__ void reorder_recv_buf(hipDoubleComplex* recv_buf, long* sort_perm,
                                 hipDoubleComplex* recv_buf_sorted, long total_recv) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < total_recv; i += grid_size) {
    recv_buf_sorted[i] = recv_buf[sort_perm[i]]; // sort_perm is 0-based
  }
}

//------------------------------------------------------------------------------
// Pack send buffer: gather values to send to neighbors
// send_buf[i] = x_local[send_offsets[i]] (send_offsets is 0-based)
//------------------------------------------------------------------------------
__global__ void pack_send_buf(hipDoubleComplex* x_local, long* send_offsets, hipDoubleComplex* send_buf,
                              long total_send) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < total_send; i += grid_size) {
    send_buf[i] = x_local[send_offsets[i]]; // send_offsets is 0-based
  }
}

//==============================================================================
// LEGACY SpMM KERNELS (from original hip_kernels.cpp)
// Kept for compatibility with existing wavefront circulant propagator
//==============================================================================

// SpMM with unit edge weights and imaginary scalar alpha
// Computes: vec_R = (i * alpha) * A * vec_L where A has all ones
// alpha is real, the multiplication by i is implicit
__global__ void unity_spmm(double alpha, int* row_starts, int* col_inds, hipDoubleComplex* vec_L,
                           hipDoubleComplex* vec_R, long m, int n, long local_i) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 0; k < n; k++) {
    for (size_t i = idx; i < local_i; i += grid_size) {
      vec_R[k * m + i].x = 0;
      vec_R[k * m + i].y = 0;
      for (long j = row_starts[i]; j < row_starts[i + 1]; j++) {
        vec_R[k * m + i].x -= alpha * vec_L[k * m + col_inds[j]].y;
        vec_R[k * m + i].y += alpha * vec_L[k * m + col_inds[j]].x;
      }
    }
  }
}

// SpMM for regular graphs (constant number of edges per row)
__global__ void regular_unity_spmm(double alpha, int per_row, int* col_inds, hipDoubleComplex* vec_L,
                                   hipDoubleComplex* vec_R, long m, int n, long local_i) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 0; k < n; k++) {
    for (size_t i = idx; i < local_i; i += grid_size) {
      vec_R[k * local_i + i].x = 0;
      vec_R[k * local_i + i].y = 0;
      for (long j = per_row * i; j < per_row * (i + 1); j++) {
        vec_R[k * local_i + i].x -= alpha * vec_L[k * m + col_inds[j]].y;
        vec_R[k * local_i + i].y += alpha * vec_L[k * m + col_inds[j]].x;
      }
    }
  }
}

// SpMM without alpha scaling (unit weights, no scalar)
__global__ void non_scaled_unity_spmm(int* row_starts, int* col_inds, hipDoubleComplex* vec_L,
                                      hipDoubleComplex* vec_R, long m, int n, long local_i) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 0; k < n; k++) {
    for (size_t i = idx; i < local_i; i += grid_size) {
      vec_R[k * m + i].x = 0;
      vec_R[k * m + i].y = 0;
      for (long j = row_starts[i]; j < row_starts[i + 1]; j++) {
        vec_R[k * m + i].x -= vec_L[k * m + col_inds[j]].y;
        vec_R[k * m + i].y += vec_L[k * m + col_inds[j]].x;
      }
    }
  }
}

// SpMM for regular graphs without alpha scaling
__global__ void non_scaled_regular_unity_spmm(int per_row, int* col_inds, hipDoubleComplex* vec_L,
                                              hipDoubleComplex* vec_R, long m, int n, long local_i) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 0; k < n; k++) {
    for (size_t i = idx; i < local_i; i += grid_size) {
      vec_R[k * local_i + i].x = 0;
      vec_R[k * local_i + i].y = 0;
      for (long j = per_row * i; j < per_row * (i + 1); j++) {
        vec_R[k * local_i + i].x -= vec_L[k * m + col_inds[j]].y;
        vec_R[k * local_i + i].y += vec_L[k * m + col_inds[j]].x;
      }
    }
  }
}

// Pack values for MPI send
__global__ void pack_send_values(hipDoubleComplex* send_values, hipDoubleComplex* source, int* RHS_send_inds,
                                 int l, size_t N, int pad, int num_send) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < num_send; i += grid_size) {
    send_values[i] = source[(N + pad) * (l - 1) + RHS_send_inds[i] - 1];
  }
}

// Unpack values from MPI receive
__global__ void unpack_rec_values(hipDoubleComplex* target, hipDoubleComplex* rec_values, int l, size_t N,
                                  int pad, int num_rec) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < num_rec; i += grid_size) {
    target[(l - 1) * (N + pad) + N + i] = rec_values[i];
  }
}

//==============================================================================
// KERNEL LAUNCHERS (extern "C" for Fortran interop)
//==============================================================================

extern "C" {
void launch_spmv_local_weighted_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                       long* row_starts, long* col_indexes, hipDoubleComplex* values,
                                       hipDoubleComplex* x_local, hipDoubleComplex* y, long lb, long ub,
                                       long local_rows) {
  hipLaunchKernelGGL((spmv_local_weighted), *grid, *block, shmem, stream, row_starts, col_indexes, values,
                     x_local, y, lb, ub, local_rows);
}
}

extern "C" {
void launch_spmv_local_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, long* row_starts,
                                   long* col_indexes, hipDoubleComplex* x_local, hipDoubleComplex* y, long lb,
                                   long ub, long local_rows) {
  hipLaunchKernelGGL((spmv_local_unit), *grid, *block, shmem, stream, row_starts, col_indexes, x_local, y, lb,
                     ub, local_rows);
}
}

extern "C" {
void launch_spmv_remote_weighted_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        long* row_starts, long* col_indexes, hipDoubleComplex* values,
                                        hipDoubleComplex* recv_buf_sorted, long* hash_keys, long* hash_vals,
                                        long hash_size, hipDoubleComplex* y, hipDoubleComplex scalar, long lb,
                                        long ub, long local_rows) {
  hipLaunchKernelGGL((spmv_remote_weighted), *grid, *block, shmem, stream, row_starts, col_indexes, values,
                     recv_buf_sorted, hash_keys, hash_vals, hash_size, y, scalar, lb, ub, local_rows);
}
}

extern "C" {
void launch_spmv_remote_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, long* row_starts,
                                    long* col_indexes, hipDoubleComplex* recv_buf_sorted, long* hash_keys,
                                    long* hash_vals, long hash_size, hipDoubleComplex* y,
                                    hipDoubleComplex scalar, long lb, long ub, long local_rows) {
  hipLaunchKernelGGL((spmv_remote_unit), *grid, *block, shmem, stream, row_starts, col_indexes,
                     recv_buf_sorted, hash_keys, hash_vals, hash_size, y, scalar, lb, ub, local_rows);
}
}

extern "C" {
void launch_reorder_recv_buf_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                    hipDoubleComplex* recv_buf, long* sort_perm,
                                    hipDoubleComplex* recv_buf_sorted, long total_recv) {
  hipLaunchKernelGGL((reorder_recv_buf), *grid, *block, shmem, stream, recv_buf, sort_perm, recv_buf_sorted,
                     total_recv);
}
}

extern "C" {
void launch_pack_send_buf_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                 hipDoubleComplex* x_local, long* send_offsets, hipDoubleComplex* send_buf,
                                 long total_send) {
  hipLaunchKernelGGL((pack_send_buf), *grid, *block, shmem, stream, x_local, send_offsets, send_buf,
                     total_send);
}
}

// Legacy SpMM launchers
extern "C" {
void launch_unity_spmm(dim3* grid, dim3* block, int shmem, hipStream_t stream, double alpha, int* row_starts,
                       int* col_inds, hipDoubleComplex* vec_L, hipDoubleComplex* vec_R, long n, int m,
                       long local_i) {
  hipLaunchKernelGGL((unity_spmm), *grid, *block, shmem, stream, alpha, row_starts, col_inds, vec_L, vec_R, n,
                     m, local_i);
}
}

extern "C" {
void launch_pack_send_values_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                    hipDoubleComplex* send_values, hipDoubleComplex* source,
                                    int* RHS_send_inds, int l, size_t N, int pad, int num_send) {
  hipLaunchKernelGGL((pack_send_values), *grid, *block, shmem, stream, send_values, source, RHS_send_inds, l,
                     N, pad, num_send);
}
}

extern "C" {
void launch_unpack_rec_values_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                     hipDoubleComplex* target, hipDoubleComplex* rec_values, int l, size_t N,
                                     int pad, int num_rec) {
  hipLaunchKernelGGL((unpack_rec_values), *grid, *block, shmem, stream, target, rec_values, l, N, pad,
                     num_rec);
}
}
