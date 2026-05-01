// HIP Chebyshev recurrence kernels for sparse matrix exponential
// Two-phase halo-based approach with accumulation.
//
// See hip_spmv_kernels.cpp for the col_halo / diag_lo / diag_hi conventions.

#include "hip_common.hpp"

//==============================================================================
// TWO-PHASE CHEBYSHEV RECURRENCE KERNELS
// For distributed case: T_{k+1} = 2*(A/M)*T_k - T_{k-1}
//==============================================================================

//------------------------------------------------------------------------------
// Chebyshev Phase 1 LOCAL (weighted): Aw_k[i] = sum over diagonal entries
//------------------------------------------------------------------------------
__global__ void chebyshev_local_weighted(double inv_M, long* row_starts, long* col_halo,
                                         hipDoubleComplex* values, long* diag_lo, long* diag_hi,
                                         hipDoubleComplex* w_k_local, hipDoubleComplex* Aw_k,
                                         long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long lo = diag_lo[i] - 1;
    long hi = diag_hi[i];

    hipDoubleComplex row_sum = {0.0, 0.0};

    for (long j = lo; j < hi; j++) {
      long local_col = col_halo[j];
      hipDoubleComplex val = values[j];
      hipDoubleComplex xj = w_k_local[local_col];
      row_sum.x += val.x * xj.x - val.y * xj.y;
      row_sum.y += val.x * xj.y + val.y * xj.x;
    }

    Aw_k[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Chebyshev Phase 1 LOCAL (unit weight)
//------------------------------------------------------------------------------
__global__ void chebyshev_local_unit(double inv_M, long* row_starts, long* col_halo, long* diag_lo,
                                     long* diag_hi, hipDoubleComplex* w_k_local, hipDoubleComplex* Aw_k,
                                     long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long lo = diag_lo[i] - 1;
    long hi = diag_hi[i];

    hipDoubleComplex row_sum = {0.0, 0.0};

    for (long j = lo; j < hi; j++) {
      long local_col = col_halo[j];
      row_sum.x += w_k_local[local_col].x;
      row_sum.y += w_k_local[local_col].y;
    }

    Aw_k[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Chebyshev Phase 2 REMOTE (weighted): w_kp1 = 2 * inv_M * (Aw_k + remote) - w_km1
//------------------------------------------------------------------------------
__global__ void chebyshev_remote_weighted(double inv_M, long* row_starts, long* col_halo,
                                          hipDoubleComplex* values, long* diag_lo, long* diag_hi,
                                          hipDoubleComplex* recv_buf, hipDoubleComplex* Aw_k,
                                          hipDoubleComplex* w_km1, hipDoubleComplex* w_kp1, long n_local,
                                          long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long row_lo = row_starts[i];
    long row_hi = row_starts[i + 1];
    long diag_first = diag_lo[i] - 1;
    long diag_last = diag_hi[i];

    hipDoubleComplex row_sum = Aw_k[i];

    for (long j = row_lo; j < diag_first; j++) {
      long off = col_halo[j] - n_local;
      hipDoubleComplex val = values[j];
      hipDoubleComplex xj = recv_buf[off];
      row_sum.x += val.x * xj.x - val.y * xj.y;
      row_sum.y += val.x * xj.y + val.y * xj.x;
    }

    for (long j = diag_last; j < row_hi; j++) {
      long off = col_halo[j] - n_local;
      hipDoubleComplex val = values[j];
      hipDoubleComplex xj = recv_buf[off];
      row_sum.x += val.x * xj.x - val.y * xj.y;
      row_sum.y += val.x * xj.y + val.y * xj.x;
    }

    w_kp1[i].x = 2.0 * inv_M * row_sum.x - w_km1[i].x;
    w_kp1[i].y = 2.0 * inv_M * row_sum.y - w_km1[i].y;
  }
}

//------------------------------------------------------------------------------
// Chebyshev Phase 2 REMOTE (unit weight)
//------------------------------------------------------------------------------
__global__ void chebyshev_remote_unit(double inv_M, long* row_starts, long* col_halo, long* diag_lo,
                                      long* diag_hi, hipDoubleComplex* recv_buf, hipDoubleComplex* Aw_k,
                                      hipDoubleComplex* w_km1, hipDoubleComplex* w_kp1, long n_local,
                                      long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long row_lo = row_starts[i];
    long row_hi = row_starts[i + 1];
    long diag_first = diag_lo[i] - 1;
    long diag_last = diag_hi[i];

    hipDoubleComplex row_sum = Aw_k[i];

    for (long j = row_lo; j < diag_first; j++) {
      long off = col_halo[j] - n_local;
      hipDoubleComplex xj = recv_buf[off];
      row_sum.x += xj.x;
      row_sum.y += xj.y;
    }

    for (long j = diag_last; j < row_hi; j++) {
      long off = col_halo[j] - n_local;
      hipDoubleComplex xj = recv_buf[off];
      row_sum.x += xj.x;
      row_sum.y += xj.y;
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
                                            double inv_M, long* row_starts, long* col_halo,
                                            hipDoubleComplex* values, long* diag_lo, long* diag_hi,
                                            hipDoubleComplex* w_k_local, hipDoubleComplex* Aw_k,
                                            long local_rows) {
  hipLaunchKernelGGL((chebyshev_local_weighted), *grid, *block, shmem, stream, inv_M, row_starts, col_halo,
                     values, diag_lo, diag_hi, w_k_local, Aw_k, local_rows);
}
}

extern "C" {
void launch_chebyshev_local_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double inv_M,
                                        long* row_starts, long* col_halo, long* diag_lo, long* diag_hi,
                                        hipDoubleComplex* w_k_local, hipDoubleComplex* Aw_k,
                                        long local_rows) {
  hipLaunchKernelGGL((chebyshev_local_unit), *grid, *block, shmem, stream, inv_M, row_starts, col_halo,
                     diag_lo, diag_hi, w_k_local, Aw_k, local_rows);
}
}

extern "C" {
void launch_chebyshev_remote_weighted_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                             double inv_M, long* row_starts, long* col_halo,
                                             hipDoubleComplex* values, long* diag_lo, long* diag_hi,
                                             hipDoubleComplex* recv_buf, hipDoubleComplex* Aw_k,
                                             hipDoubleComplex* w_km1, hipDoubleComplex* w_kp1, long n_local,
                                             long local_rows) {
  hipLaunchKernelGGL((chebyshev_remote_weighted), *grid, *block, shmem, stream, inv_M, row_starts, col_halo,
                     values, diag_lo, diag_hi, recv_buf, Aw_k, w_km1, w_kp1, n_local, local_rows);
}
}

extern "C" {
void launch_chebyshev_remote_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double inv_M,
                                         long* row_starts, long* col_halo, long* diag_lo, long* diag_hi,
                                         hipDoubleComplex* recv_buf, hipDoubleComplex* Aw_k,
                                         hipDoubleComplex* w_km1, hipDoubleComplex* w_kp1, long n_local,
                                         long local_rows) {
  hipLaunchKernelGGL((chebyshev_remote_unit), *grid, *block, shmem, stream, inv_M, row_starts, col_halo,
                     diag_lo, diag_hi, recv_buf, Aw_k, w_km1, w_kp1, n_local, local_rows);
}
}

extern "C" {
void launch_chebyshev_accumulate_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        hipDoubleComplex coeff, hipDoubleComplex* w, hipDoubleComplex* C,
                                        size_t N) {
  hipLaunchKernelGGL((chebyshev_accumulate), *grid, *block, shmem, stream, coeff, w, C, N);
}
}
