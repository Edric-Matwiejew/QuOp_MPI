// HIP Chebyshev recurrence kernels for sparse matrix exponential
// Two-phase distributed approach with accumulation

#include "hip_common.hpp"

//==============================================================================
// TWO-PHASE CHEBYSHEV RECURRENCE KERNELS
// For distributed case: T_{k+1} = 2*(A/M)*T_k - T_{k-1}
//==============================================================================

//------------------------------------------------------------------------------
// Chebyshev Phase 1: LOCAL contributions for A*w_k
//------------------------------------------------------------------------------
__global__ void chebyshev_local_weighted(double inv_M, long* row_starts, long* col_indexes,
                                         hipDoubleComplex* values, hipDoubleComplex* w_k_local,
                                         hipDoubleComplex* Aw_k, // output: local contribution to A*w_k
                                         long lb, long ub, long local_rows) {

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
        hipDoubleComplex val = values[j];
        hipDoubleComplex xj = w_k_local[local_col];
        row_sum.x += val.x * xj.x - val.y * xj.y;
        row_sum.y += val.x * xj.y + val.y * xj.x;
      }
    }

    // Store local contribution (will be completed in phase 2)
    Aw_k[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Chebyshev Phase 1: LOCAL - Unit weight version
//------------------------------------------------------------------------------
__global__ void chebyshev_local_unit(double inv_M, long* row_starts, long* col_indexes,
                                     hipDoubleComplex* w_k_local, hipDoubleComplex* Aw_k, long lb, long ub,
                                     long local_rows) {

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
        row_sum.x += w_k_local[local_col].x;
        row_sum.y += w_k_local[local_col].y;
      }
    }

    Aw_k[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Chebyshev Phase 2: Complete recurrence with REMOTE contributions
// w_kp1 = 2 * inv_M * (Aw_k + remote) - w_km1
//------------------------------------------------------------------------------
__global__ void chebyshev_remote_weighted(double inv_M, long* row_starts, long* col_indexes,
                                          hipDoubleComplex* values, hipDoubleComplex* recv_buf_sorted,
                                          long* hash_keys, long* hash_vals, long hash_size,
                                          hipDoubleComplex* Aw_k,  // input: local contribution
                                          hipDoubleComplex* w_km1, // T_{k-1}
                                          hipDoubleComplex* w_kp1, // output: T_{k+1}
                                          long lb, long ub, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long start_j = row_starts[i];
    long end_j = row_starts[i + 1] - 1;

    hipDoubleComplex row_sum = Aw_k[i];

    if (start_j <= end_j) {
      // Skip if all columns are local
      if (!(col_indexes[start_j] >= lb && col_indexes[end_j] <= ub)) {
        long local_start = lower_bound_dev(col_indexes, start_j, end_j + 1, lb);
        long local_end = upper_bound_dev(col_indexes, start_j, end_j + 1, ub) - 1;

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
    }

    // Apply Chebyshev recurrence: T_{k+1} = 2*(A/M)*T_k - T_{k-1}
    w_kp1[i].x = 2.0 * inv_M * row_sum.x - w_km1[i].x;
    w_kp1[i].y = 2.0 * inv_M * row_sum.y - w_km1[i].y;
  }
}

//------------------------------------------------------------------------------
// Chebyshev Phase 2: REMOTE - Unit weight version
//------------------------------------------------------------------------------
__global__ void chebyshev_remote_unit(double inv_M, long* row_starts, long* col_indexes,
                                      hipDoubleComplex* recv_buf_sorted, long* hash_keys, long* hash_vals,
                                      long hash_size, hipDoubleComplex* Aw_k, hipDoubleComplex* w_km1,
                                      hipDoubleComplex* w_kp1, long lb, long ub, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long start_j = row_starts[i];
    long end_j = row_starts[i + 1] - 1;

    hipDoubleComplex row_sum = Aw_k[i];

    if (start_j <= end_j) {
      if (!(col_indexes[start_j] >= lb && col_indexes[end_j] <= ub)) {
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
    }

    w_kp1[i].x = 2.0 * inv_M * row_sum.x - w_km1[i].x;
    w_kp1[i].y = 2.0 * inv_M * row_sum.y - w_km1[i].y;
  }
}

// Accumulate Chebyshev term: C += coeff * w
// coeff is a precomputed Bessel coefficient (complex)
__global__ void chebyshev_accumulate(hipDoubleComplex coeff, hipDoubleComplex* w, hipDoubleComplex* C,
                                     size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  hipDoubleComplex prod;
  for (size_t i = idx; i < N; i += grid_size) {
    prod.x = coeff.x * w[i].x - coeff.y * w[i].y;
    prod.y = coeff.x * w[i].y + coeff.y * w[i].x;
    C[i].x += prod.x;
    C[i].y += prod.y;
  }
}

//==============================================================================
// KERNEL LAUNCHERS (extern "C" for Fortran interop)
//==============================================================================

extern "C" {
void launch_chebyshev_local_weighted_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                            double inv_M, long* row_starts, long* col_indexes,
                                            hipDoubleComplex* values, hipDoubleComplex* w_k_local,
                                            hipDoubleComplex* Aw_k, long lb, long ub, long local_rows) {
  hipLaunchKernelGGL((chebyshev_local_weighted), *grid, *block, shmem, stream, inv_M, row_starts, col_indexes,
                     values, w_k_local, Aw_k, lb, ub, local_rows);
}
}

extern "C" {
void launch_chebyshev_local_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double inv_M,
                                        long* row_starts, long* col_indexes, hipDoubleComplex* w_k_local,
                                        hipDoubleComplex* Aw_k, long lb, long ub, long local_rows) {
  hipLaunchKernelGGL((chebyshev_local_unit), *grid, *block, shmem, stream, inv_M, row_starts, col_indexes,
                     w_k_local, Aw_k, lb, ub, local_rows);
}
}

extern "C" {
void launch_chebyshev_remote_weighted_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                             double inv_M, long* row_starts, long* col_indexes,
                                             hipDoubleComplex* values, hipDoubleComplex* recv_buf_sorted,
                                             long* hash_keys, long* hash_vals, long hash_size,
                                             hipDoubleComplex* Aw_k, hipDoubleComplex* w_km1,
                                             hipDoubleComplex* w_kp1, long lb, long ub, long local_rows) {
  hipLaunchKernelGGL((chebyshev_remote_weighted), *grid, *block, shmem, stream, inv_M, row_starts,
                     col_indexes, values, recv_buf_sorted, hash_keys, hash_vals, hash_size, Aw_k, w_km1,
                     w_kp1, lb, ub, local_rows);
}
}

extern "C" {
void launch_chebyshev_remote_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double inv_M,
                                         long* row_starts, long* col_indexes,
                                         hipDoubleComplex* recv_buf_sorted, long* hash_keys, long* hash_vals,
                                         long hash_size, hipDoubleComplex* Aw_k, hipDoubleComplex* w_km1,
                                         hipDoubleComplex* w_kp1, long lb, long ub, long local_rows) {
  hipLaunchKernelGGL((chebyshev_remote_unit), *grid, *block, shmem, stream, inv_M, row_starts, col_indexes,
                     recv_buf_sorted, hash_keys, hash_vals, hash_size, Aw_k, w_km1, w_kp1, lb, ub,
                     local_rows);
}
}

extern "C" {
void launch_chebyshev_accumulate_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        hipDoubleComplex coeff, hipDoubleComplex* w, hipDoubleComplex* C,
                                        size_t N) {
  hipLaunchKernelGGL((chebyshev_accumulate), *grid, *block, shmem, stream, coeff, w, C, N);
}
}
