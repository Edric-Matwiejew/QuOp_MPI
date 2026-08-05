// HIP SpMV kernels for distributed sparse matrix-vector multiplication
// Two-phase halo-based approach: LOCAL computation overlaps with MPI, then
// REMOTE contributions are added.
//
// All "col_indexes" arrays passed below contain halo offsets, not raw global
// column indices: col_halo[j] in [0, n_local) -> diagonal entry, indexes
// x_local; col_halo[j] in [n_local, n_local + total_recv) -> off-diagonal
// entry, indexes recv_buf at offset (col_halo[j] - n_local).
//
// row_starts is 0-based half-open ([row_starts[i], row_starts[i+1])) and
// arrays passed from Fortran are 1-indexed; col_halo is 0-based for direct
// vector indexing.  diag_lo/diag_hi are populated as 1-based inclusive
// Fortran indices; the local kernel converts to 0-based via diag_lo - 1 and
// the remote kernel uses the same convention.

#include "hip_common.hpp"

//==============================================================================
// TWO-PHASE DISTRIBUTED SpMV KERNELS (halo-based)
//   Phase 1 (LOCAL): contributions from columns in [diag_lo[i], diag_hi[i]]
//   Phase 2 (REMOTE): contributions from the off-diagonal segments, scaled by
//   `scalar` along with the local contribution.
//==============================================================================

//------------------------------------------------------------------------------
// Phase 1 LOCAL (weighted): y[i] = sum over diagonal entries
//------------------------------------------------------------------------------
__global__ void spmv_local_weighted(long* row_starts, long* col_halo, hipDoubleComplex* values,
                                    long* diag_lo, long* diag_hi, hipDoubleComplex* x_local,
                                    hipDoubleComplex* y, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    // diag_lo / diag_hi are 1-based inclusive; convert to 0-based half-open.
    long lo = diag_lo[i] - 1;
    long hi = diag_hi[i]; // exclusive in 0-based form

    hipDoubleComplex row_sum = {0.0, 0.0};

    for (long j = lo; j < hi; j++) {
      long local_col = col_halo[j]; // 0-based local index into x_local
      hipDoubleComplex val = values[j];
      hipDoubleComplex xj = x_local[local_col];
      row_sum.x += val.x * xj.x - val.y * xj.y;
      row_sum.y += val.x * xj.y + val.y * xj.x;
    }

    y[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Phase 1 LOCAL (unit weight)
//------------------------------------------------------------------------------
__global__ void spmv_local_unit(long* row_starts, long* col_halo, long* diag_lo, long* diag_hi,
                                hipDoubleComplex* x_local, hipDoubleComplex* y, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long lo = diag_lo[i] - 1;
    long hi = diag_hi[i];

    hipDoubleComplex row_sum = {0.0, 0.0};

    for (long j = lo; j < hi; j++) {
      long local_col = col_halo[j];
      row_sum.x += x_local[local_col].x;
      row_sum.y += x_local[local_col].y;
    }

    y[i] = row_sum;
  }
}

//------------------------------------------------------------------------------
// Phase 2 REMOTE (weighted): adds off-diagonal contributions and applies scalar.
//   y[i] = scalar * (y[i] + sum over off-diagonal entries reading recv_buf)
//------------------------------------------------------------------------------
__global__ void spmv_remote_weighted(long* row_starts, long* col_halo, hipDoubleComplex* values,
                                     long* diag_lo, long* diag_hi, hipDoubleComplex* recv_buf,
                                     hipDoubleComplex* y, hipDoubleComplex scalar, long n_local,
                                     long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long row_lo = row_starts[i];           // 0-based first
    long row_hi = row_starts[i + 1];       // 0-based exclusive last
    long diag_first = diag_lo[i] - 1;      // 0-based first diagonal entry
    long diag_last = diag_hi[i];           // 0-based exclusive last diagonal entry

    hipDoubleComplex row_sum = y[i]; // pre-loaded with local contribution

    // Off-lower segment
    for (long j = row_lo; j < diag_first; j++) {
      long off = col_halo[j] - n_local; // 0-based offset into recv_buf
      hipDoubleComplex val = values[j];
      hipDoubleComplex xj = recv_buf[off];
      row_sum.x += val.x * xj.x - val.y * xj.y;
      row_sum.y += val.x * xj.y + val.y * xj.x;
    }

    // Off-upper segment
    for (long j = diag_last; j < row_hi; j++) {
      long off = col_halo[j] - n_local;
      hipDoubleComplex val = values[j];
      hipDoubleComplex xj = recv_buf[off];
      row_sum.x += val.x * xj.x - val.y * xj.y;
      row_sum.y += val.x * xj.y + val.y * xj.x;
    }

    y[i].x = scalar.x * row_sum.x - scalar.y * row_sum.y;
    y[i].y = scalar.x * row_sum.y + scalar.y * row_sum.x;
  }
}

//------------------------------------------------------------------------------
// Phase 2 REMOTE (unit weight)
//------------------------------------------------------------------------------
__global__ void spmv_remote_unit(long* row_starts, long* col_halo, long* diag_lo, long* diag_hi,
                                 hipDoubleComplex* recv_buf, hipDoubleComplex* y,
                                 hipDoubleComplex scalar, long n_local, long local_rows) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    long row_lo = row_starts[i];
    long row_hi = row_starts[i + 1];
    long diag_first = diag_lo[i] - 1;
    long diag_last = diag_hi[i];

    hipDoubleComplex row_sum = y[i];

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

    y[i].x = scalar.x * row_sum.x - scalar.y * row_sum.y;
    y[i].y = scalar.x * row_sum.y + scalar.y * row_sum.x;
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
    send_buf[i] = x_local[send_offsets[i]];
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
                                       long* row_starts, long* col_halo, hipDoubleComplex* values,
                                       long* diag_lo, long* diag_hi, hipDoubleComplex* x_local,
                                       hipDoubleComplex* y, long local_rows) {
  hipLaunchKernelGGL((spmv_local_weighted), *grid, *block, shmem, stream, row_starts, col_halo, values,
                     diag_lo, diag_hi, x_local, y, local_rows);
}
}

extern "C" {
void launch_spmv_local_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, long* row_starts,
                                   long* col_halo, long* diag_lo, long* diag_hi, hipDoubleComplex* x_local,
                                   hipDoubleComplex* y, long local_rows) {
  hipLaunchKernelGGL((spmv_local_unit), *grid, *block, shmem, stream, row_starts, col_halo, diag_lo, diag_hi,
                     x_local, y, local_rows);
}
}

extern "C" {
void launch_spmv_remote_weighted_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        long* row_starts, long* col_halo, hipDoubleComplex* values,
                                        long* diag_lo, long* diag_hi, hipDoubleComplex* recv_buf,
                                        hipDoubleComplex* y, hipDoubleComplex scalar, long n_local,
                                        long local_rows) {
  hipLaunchKernelGGL((spmv_remote_weighted), *grid, *block, shmem, stream, row_starts, col_halo, values,
                     diag_lo, diag_hi, recv_buf, y, scalar, n_local, local_rows);
}
}

extern "C" {
void launch_spmv_remote_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, long* row_starts,
                                    long* col_halo, long* diag_lo, long* diag_hi, hipDoubleComplex* recv_buf,
                                    hipDoubleComplex* y, hipDoubleComplex scalar, long n_local,
                                    long local_rows) {
  hipLaunchKernelGGL((spmv_remote_unit), *grid, *block, shmem, stream, row_starts, col_halo, diag_lo,
                     diag_hi, recv_buf, y, scalar, n_local, local_rows);
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
