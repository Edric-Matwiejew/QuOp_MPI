// HIP reduction kernels for sparse matrix operations
// Gershgorin bounds, column norms, max reduction, dense norms

#include "hip_common.hpp"

//==============================================================================
// REDUCTION KERNELS
//==============================================================================

// 1-norms for multiple vectors (l vectors of length N each)
// l must be 5 or less
__global__ void dense_one_norms(double* result, hipDoubleComplex* X, int N, int l) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  double sum[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < l; i++) {
    for (int j = idx; j < N; j += grid_size) {
      sum[i] += sqrt(X[i * N + j].x * X[i * N + j].x + X[i * N + j].y * X[i * N + j].y);
    }
  }

  __syncthreads();

  __shared__ double local_sum[5 * BLOCKSIZE];

  for (int i = 0; i < l; i++) {
    local_sum[i * BLOCKSIZE + threadIdx.x] = sum[i];
  }
  __syncthreads();

  for (int i = 0; i < l; i++) {
    for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
      if (threadIdx.x < s) {
        local_sum[i * BLOCKSIZE + threadIdx.x] += local_sum[i * BLOCKSIZE + threadIdx.x + s];
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < l; i++) {
      result[i * BLOCKSIZE + blockIdx.x] = local_sum[i * BLOCKSIZE];
    }
  }
}

// Row-wise sum for infinity norm estimation
__global__ void infinity_norm(double* result, hipDoubleComplex* X, int N, int l) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < N; i += grid_size) {
    result[i] = 0;
    for (int j = 0; j < l; j++) {
      result[i] += sqrt(X[j * N + i].x * X[j * N + i].x + X[j * N + i].y * X[j * N + i].y);
    }
  }
}

// Compute column 1-norms for CSR matrix (needed for one_norms.f90)
// Each thread handles one row, atomic add to handle multiple rows per column
__global__ void csr_column_one_norms(long* row_starts, int* col_inds, hipDoubleComplex* values,
                                     double* col_norms, int num_rows, int num_cols) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = row; i < num_rows; i += grid_size) {
    for (long j = row_starts[i]; j < row_starts[i + 1]; j++) {
      int col = col_inds[j];
      double abs_val = sqrt(values[j].x * values[j].x + values[j].y * values[j].y);
      atomicAdd(&col_norms[col], abs_val);
    }
  }
}

// Unit weight version
__global__ void csr_column_one_norms_unit(long* row_starts, int* col_inds, double* col_norms, int num_rows,
                                          int num_cols) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = row; i < num_rows; i += grid_size) {
    for (long j = row_starts[i]; j < row_starts[i + 1]; j++) {
      int col = col_inds[j];
      atomicAdd(&col_norms[col], 1.0);
    }
  }
}

// Gershgorin spectral radius estimation
// Computes max_i(|A_ii| + sum_{j!=i}|A_ij|) for local rows
__global__ void gershgorin_bound(long* row_starts, int* col_inds, hipDoubleComplex* values,
                                 double* row_bounds, int local_rows, int offset) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    double diag = 0.0;
    double off_diag = 0.0;
    int global_row = i + offset;

    for (long j = row_starts[i]; j < row_starts[i + 1]; j++) {
      double abs_val = sqrt(values[j].x * values[j].x + values[j].y * values[j].y);
      if (col_inds[j] == global_row) {
        diag = abs_val;
      } else {
        off_diag += abs_val;
      }
    }

    row_bounds[i] = diag + off_diag;
  }
}

// Unit weight version of Gershgorin bound
__global__ void gershgorin_bound_unit(long* row_starts, int* col_inds, double* row_bounds, int local_rows,
                                      int offset) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_rows; i += grid_size) {
    double diag = 0.0;
    double off_diag = 0.0;
    int global_row = i + offset;

    for (long j = row_starts[i]; j < row_starts[i + 1]; j++) {
      if (col_inds[j] == global_row) {
        diag = 1.0;
      } else {
        off_diag += 1.0;
      }
    }

    row_bounds[i] = diag + off_diag;
  }
}

// Final reduction for Gershgorin bound (find max of row_bounds)
__global__ void reduce_max(double* data, double* result, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ double local_max[BLOCKSIZE];

  double max_val = 0.0;
  for (size_t i = idx; i < N; i += grid_size) {
    max_val = data[i] > max_val ? data[i] : max_val;
  }

  __syncthreads();

  local_max[threadIdx.x] = max_val;
  __syncthreads();

  for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      double l = local_max[threadIdx.x];
      double r = local_max[threadIdx.x + s];
      local_max[threadIdx.x] = l > r ? l : r;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = local_max[0];
  }
}

//==============================================================================
// KERNEL LAUNCHERS (extern "C" for Fortran interop)
//==============================================================================

extern "C" {
void launch_dense_one_norms_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double* result,
                                   hipDoubleComplex* X, int N, int l) {
  hipLaunchKernelGGL((dense_one_norms), *grid, *block, shmem, stream, result, X, N, l);
}
}

extern "C" {
void launch_infinity_norm_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double* result,
                                 hipDoubleComplex* X, int N, int M) {
  hipLaunchKernelGGL((infinity_norm), *grid, *block, shmem, stream, result, X, N, M);
}
}

extern "C" {
void launch_csr_column_one_norms_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        long* row_starts, int* col_inds, hipDoubleComplex* values,
                                        double* col_norms, int num_rows, int num_cols) {
  hipLaunchKernelGGL((csr_column_one_norms), *grid, *block, shmem, stream, row_starts, col_inds, values,
                     col_norms, num_rows, num_cols);
}
}

extern "C" {
void launch_csr_column_one_norms_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                             long* row_starts, int* col_inds, double* col_norms, int num_rows,
                                             int num_cols) {
  hipLaunchKernelGGL((csr_column_one_norms_unit), *grid, *block, shmem, stream, row_starts, col_inds,
                     col_norms, num_rows, num_cols);
}
}

extern "C" {
void launch_gershgorin_bound_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, long* row_starts,
                                    int* col_inds, hipDoubleComplex* values, double* row_bounds,
                                    int local_rows, int offset) {
  hipLaunchKernelGGL((gershgorin_bound), *grid, *block, shmem, stream, row_starts, col_inds, values,
                     row_bounds, local_rows, offset);
}
}

extern "C" {
void launch_gershgorin_bound_unit_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                         long* row_starts, int* col_inds, double* row_bounds, int local_rows,
                                         int offset) {
  hipLaunchKernelGGL((gershgorin_bound_unit), *grid, *block, shmem, stream, row_starts, col_inds, row_bounds,
                     local_rows, offset);
}
}

extern "C" {
void launch_reduce_max_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double* data,
                              double* result, size_t N) {
  hipLaunchKernelGGL((reduce_max), *grid, *block, shmem, stream, data, result, N);
}
}
