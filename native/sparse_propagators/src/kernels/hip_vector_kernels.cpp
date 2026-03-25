// HIP vector operation kernels for sparse matrix exponential
// Operations: norm, copy, scale, axpy

#include "hip_common.hpp"

//==============================================================================
// VECTOR KERNELS
//==============================================================================

// Compute infinity norm of a complex vector
// Returns partial maxima per block, needs final reduction on host or second kernel
__global__ void vector_infinity_norm(double* infnorm, hipDoubleComplex* v, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ double local_max[BLOCKSIZE];

  double abs_val;
  double max_abs = 0.0;

  for (size_t i = idx; i < N; i += grid_size) {
    abs_val = sqrt(v[i].x * v[i].x + v[i].y * v[i].y);
    max_abs = abs_val > max_abs ? abs_val : max_abs;
  }

  __syncthreads();

  local_max[threadIdx.x] = max_abs;

  __syncthreads();

  double l;
  double r;

  for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      l = local_max[threadIdx.x];
      r = local_max[threadIdx.x + s];
      local_max[threadIdx.x] = l < r ? r : l;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    infnorm[blockIdx.x] = local_max[0];
  }
}

// In-place vector sum: X = X + Y
__global__ void inplace_vec_sum(hipDoubleComplex* X, hipDoubleComplex* Y, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < N; i += grid_size) {
    X[i].x += Y[i].x;
    X[i].y += Y[i].y;
  }
}

// Scale vector for Taylor series: X = X / (s * j)
__global__ void b_scale(hipDoubleComplex* X, int s, int j, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  double scale = 1.0 / (s * j);
  for (size_t i = idx; i < N; i += grid_size) {
    X[i].x *= scale;
    X[i].y *= scale;
  }
}

// General complex AXPY: y = alpha * x + y
// alpha is complex
__global__ void complex_axpy(hipDoubleComplex alpha, hipDoubleComplex* x, hipDoubleComplex* y, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  hipDoubleComplex ax;
  for (size_t i = idx; i < N; i += grid_size) {
    // Complex multiplication: (a.x + i*a.y) * (x.x + i*x.y)
    ax.x = alpha.x * x[i].x - alpha.y * x[i].y;
    ax.y = alpha.x * x[i].y + alpha.y * x[i].x;
    y[i].x += ax.x;
    y[i].y += ax.y;
  }
}

// Scale vector by complex scalar: x = alpha * x
__global__ void complex_scale(hipDoubleComplex alpha, hipDoubleComplex* x, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  hipDoubleComplex temp;
  for (size_t i = idx; i < N; i += grid_size) {
    temp.x = alpha.x * x[i].x - alpha.y * x[i].y;
    temp.y = alpha.x * x[i].y + alpha.y * x[i].x;
    x[i] = temp;
  }
}

// Scale vector by real scalar: x = alpha * x
__global__ void real_scale(double alpha, hipDoubleComplex* x, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < N; i += grid_size) {
    x[i].x *= alpha;
    x[i].y *= alpha;
  }
}

// Copy vector: y = x
__global__ void vec_copy(hipDoubleComplex* x, hipDoubleComplex* y, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < N; i += grid_size) {
    y[i] = x[i];
  }
}

//==============================================================================
// KERNEL LAUNCHERS (extern "C" for Fortran interop)
//==============================================================================

extern "C" {
void launch_vector_infinity_norm_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        double* infnorm, hipDoubleComplex* X, size_t N) {
  hipLaunchKernelGGL((vector_infinity_norm), *grid, *block, shmem, stream, infnorm, X, N);
}
}

extern "C" {
void launch_inplace_vec_sum_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                   hipDoubleComplex* X, hipDoubleComplex* Y, size_t N) {
  hipLaunchKernelGGL((inplace_vec_sum), *grid, *block, shmem, stream, X, Y, N);
}
}

extern "C" {
void launch_b_scale_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, hipDoubleComplex* X, int s,
                           int j, size_t N) {
  hipLaunchKernelGGL((b_scale), *grid, *block, shmem, stream, X, s, j, N);
}
}

extern "C" {
void launch_complex_axpy_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                hipDoubleComplex alpha, hipDoubleComplex* x, hipDoubleComplex* y, size_t N) {
  hipLaunchKernelGGL((complex_axpy), *grid, *block, shmem, stream, alpha, x, y, N);
}
}

extern "C" {
void launch_complex_scale_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                 hipDoubleComplex alpha, hipDoubleComplex* x, size_t N) {
  hipLaunchKernelGGL((complex_scale), *grid, *block, shmem, stream, alpha, x, N);
}
}

extern "C" {
void launch_real_scale_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double alpha,
                              hipDoubleComplex* x, size_t N) {
  hipLaunchKernelGGL((real_scale), *grid, *block, shmem, stream, alpha, x, N);
}
}

extern "C" {
void launch_vec_copy_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, hipDoubleComplex* x,
                            hipDoubleComplex* y, size_t N) {
  hipLaunchKernelGGL((vec_copy), *grid, *block, shmem, stream, x, y, N);
}
}
