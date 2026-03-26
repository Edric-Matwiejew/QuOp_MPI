#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

constexpr int BLOCKSIZE = 256;
constexpr double PI = 3.141592653589793;

__global__ void expectation_value(double* result, hipDoubleComplex* a, double* b, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0;
  for (size_t i = idx; i < N; i += grid_size) {
    sum += (a[i].x * a[i].x + a[i].y * a[i].y) * b[i];
  }
  __syncthreads();

  __shared__ double local_sum[BLOCKSIZE];
  local_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      local_sum[threadIdx.x] += local_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = local_sum[0];
  }
}

__global__ void state_norm(double* result, hipDoubleComplex* a, size_t N) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid_size = blockDim.x * gridDim.x;

  double sum = 0;
  for (size_t i = idx; i < N; i += grid_size) {
    sum += a[i].x * a[i].x + a[i].y * a[i].y;
  }
  __syncthreads();

  __shared__ double local_sum[BLOCKSIZE];
  local_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int s = BLOCKSIZE / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      local_sum[threadIdx.x] += local_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = local_sum[0];
  }
}

// compute the action of a phase-shift unitary
__global__ void phase_shift(double gamma, double* diag_operator, hipDoubleComplex* state, size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  hipDoubleComplex phase_shift;
  // required as state[i].x is updated first
  double state_x;

  // Euler's formula: e^(i x) = cos x + i sin x
  // complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
  for (size_t i = idx; i < N; i += grid_size) {
    state_x = state[i].x;
    phase_shift.x = cos(gamma * diag_operator[i]);
    phase_shift.y = sin(-gamma * diag_operator[i]);
    state[i].x = phase_shift.x * state[i].x - phase_shift.y * state[i].y;
    state[i].y = phase_shift.x * state[i].y + phase_shift.y * state_x;
  }
}

__global__ void fourier_normalise(double norm_factor, hipDoubleComplex* state, size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < N; i += grid_size) {
    state[i].x *= norm_factor;
    state[i].y *= norm_factor;
  }
}

__global__ void circulant_eigenvalues(int nnz, long* indexes, double* values, double* eigenvalues, size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // real component only as the eigenvalues are real
  // for symmetric matrices
  double eigen, doubletwoiPiDivN;
  for (size_t i = idx; i < N; i += grid_size) {
    eigen = 0;
    doubletwoiPiDivN = 2.0 * i * PI / (double)N;
    for (int j = 0; j < nnz; j++) {
      eigen += cos(indexes[j] * doubletwoiPiDivN) * values[j];
    }
    eigenvalues[i] = eigen;
  }
}

// Distributed version: computes eigenvalues for local portion with global offset
// global_N is the total system size, local_N is the local portion, offset is the global starting index
// For indices >= global_N (padding region), eigenvalue is set to 0
__global__ void distributed_circulant_eigenvalues(int nnz, long* indexes, double* values, double* eigenvalues,
                                                  size_t local_N, size_t global_N, size_t offset) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // real component only as the eigenvalues are real for symmetric matrices
  double eigen, doubletwoiPiDivN;
  for (size_t i = idx; i < local_N; i += grid_size) {
    size_t global_i = i + offset; // global index for DFT formula

    // Handle padding: indices beyond system_size get zero eigenvalue
    if (global_i >= global_N) {
      eigenvalues[i] = 0.0;
      continue;
    }

    eigen = 0;
    doubletwoiPiDivN = 2.0 * global_i * PI / (double)global_N;
    for (int j = 0; j < nnz; j++) {
      eigen += cos(indexes[j] * doubletwoiPiDivN) * values[j];
    }
    eigenvalues[i] = eigen;
  }
}

// Distributed version for complete graphs
// For indices >= global_N (padding region), eigenvalue is set to 0
__global__ void distributed_complete_graph_eigenvalues(double* eigenvalues, size_t local_N, size_t global_N,
                                                       size_t offset) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < local_N; i += grid_size) {
    size_t global_i = i + offset;

    // Handle padding: indices beyond system_size get zero eigenvalue
    if (global_i >= global_N) {
      eigenvalues[i] = 0.0;
    } else if (global_i == 0) {
      eigenvalues[i] = (double)(global_N - 1);
    } else {
      eigenvalues[i] = (double)(-1);
    }
  }
}

__global__ void complete_graph_eigenvalues(double* eigenvalues, size_t N) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = idx; i < N; i += grid_size) {
    eigenvalues[i] = (double)(-1);
  }
  if (idx == 0) {
    eigenvalues[0] = (double)(N - 1);
  }
}

__global__ void n_dim_circulant_eigenvalues(int n_dim, int Ns_max, int* Ns, double* graph_array,
                                            double* eigenvalues, size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // real component only as the eigenvalues are real
  // for symmetric matrices
  // each column starts at Ns_max * i
  int N_loc;
  double eigenvalue;
  for (int i = 0; i < n_dim; i++) {
    N_loc = Ns[i];
    for (int j = idx; j < Ns_max; j += grid_size) {
      eigenvalue = 0;
      for (int k = 0; k < N_loc; k++) {
        eigenvalue += cos(2 * ((double)j * (double)k) * PI / (double)N_loc) * graph_array[Ns_max * i + k];
      }
      eigenvalues[Ns_max * i + j] = eigenvalue;
      __syncthreads();
    }
  }
}

__global__ void n_dim_complete_graph_eigenvalues(int n_dim, int Ns_max, int* Ns, double* eigenvalues,
                                                 size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  int N_loc;
  for (int i = 0; i < n_dim; i++) {
    N_loc = Ns[i];
    for (int j = idx; j < Ns_max; j += grid_size) {
      eigenvalues[Ns_max * i + j] = (double)(-1);
    }
    __syncthreads();
    if (idx == 0) {
      eigenvalues[Ns_max * i] = (double)(N_loc - 1);
    }
  }
}

__global__ void composite_mixer(int n_dim, int Ns_max, int* Ns, int* strides, double* ts, double* eigenvalues,
                                double* mixer, size_t N, size_t offset) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int indices[20];
  double m;

  // Iterate over local indices (0 to N-1)
  // Use offset to compute the global multi-dimensional indices
  for (size_t local_i = idx; local_i < N; local_i += grid_size) {
    size_t global_i = local_i + offset;

    // Compute multi-dimensional indices from global linear index
    for (int j = 0; j < n_dim; j++) {
      indices[j] = ((int)global_i / strides[j]) % Ns[j];
    }

    // Compute the mixer value from per-dimension eigenvalues
    m = 0;
    for (int j = 0; j < n_dim; j++) {
      int eigen_idx = j * Ns_max + indices[j];
      m += ts[j] * eigenvalues[eigen_idx];
    }

    // Write to local mixer array
    mixer[local_i] = m;
  }
}

__global__ void constant_phase_shift(double* diag_operator, hipDoubleComplex* state, size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  hipDoubleComplex phase_shift;
  // required as state[i].x is updated first
  double state_x;

  // Euler's formula: e^(i x) = cos x + i sin x
  // complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
  for (size_t i = idx; i < N; i += grid_size) {
    state_x = state[i].x;
    phase_shift.x = cos(diag_operator[i]);
    phase_shift.y = sin(-diag_operator[i]);
    state[i].x = phase_shift.x * state[i].x - phase_shift.y * state[i].y;
    state[i].y = phase_shift.x * state[i].y + phase_shift.y * state_x;
  }
}

__global__ void init_x(hipDoubleComplex* X, int N, int l, int seed) {

  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  double x_val = 1.0 / (double)N;

  for (size_t i = idx; i < N; i += grid_size) {
    X[i].x = x_val;
    X[i].y = 0;
  }

  // LCG random number generator, hipRAND is overkill for this
  // parameters from Numerical Recipes
  long int a = 1664525;
  long int c = 1013904223;
  long int m = 4294967296;
  long int r = m / idx % seed;
  double rd;

  r = (a * r + c) % m; // if BLOCKSIZE < N

  if (l > 1) {
    for (int i = N + idx; i < l * N; i += grid_size) {
      r = (a * r + c) % m;
      rd = r / (double)m;
      if (rd < 0.5) {
        X[i].x = x_val;
        X[i].y = 0;
      } else {
        X[i].x = -x_val;
        X[i].y = 0;
      }
    }
  }
}

// Momentum propagator kernels

// Generate momentum-space eigenvalues (k^2 for kinetic energy)
__global__ void n_dim_momentum_eigenvalues(int n_dim, int Ns_max, int* Ns, double* minsk, double* deltask,
                                           double* eigenvalues, int N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  int offset = 0;
  double k_val;

  for (int dim = 0; dim < n_dim; dim++) {
    for (size_t i = idx; i < Ns[dim]; i += grid_size) {
      // k = minsk + i * deltask
      k_val = minsk[dim] + (double)i * deltask[dim];
      eigenvalues[offset + i] = k_val * k_val; // k^2 for kinetic energy
    }
    offset += Ns[dim];
    __syncthreads();
  }
}

// Generate phase factors for position<->momentum transforms
// direction = 0: phase_k = exp(-i * sum(k * minsq))
// direction = 1: phase_q = exp(i * sum(q * minsk))
__global__ void gen_phase_factors(int n_dim, int Ns_max, int* Ns, int* strides, double* mins_target,
                                  double* deltas_source, double* mins_source, hipDoubleComplex* phase_out,
                                  size_t N, size_t offset, int direction) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  double phase_sum;
  double grid_point;
  size_t global_idx;
  int inds[20]; // max 20 dimensions

  for (size_t i = idx; i < N; i += grid_size) {
    global_idx = i + offset;

    // Compute n-dimensional indices
    size_t temp = global_idx;
    for (int dim = 0; dim < n_dim; dim++) {
      inds[dim] = temp / strides[dim];
      temp = temp % strides[dim];
    }

    // Compute phase sum
    phase_sum = 0.0;
    for (int dim = 0; dim < n_dim; dim++) {
      grid_point = mins_source[dim] + (double)inds[dim] * deltas_source[dim];
      phase_sum += grid_point * mins_target[dim];
    }

    // phase_k: exp(-i * sum), phase_q: exp(+i * sum)
    if (direction == 0) {
      phase_out[i].x = cos(-phase_sum);
      phase_out[i].y = sin(-phase_sum);
    } else {
      phase_out[i].x = cos(phase_sum);
      phase_out[i].y = sin(phase_sum);
    }
    __syncthreads();
  }
}

// Apply complex phase multiplication: state = phase * state
__global__ void apply_complex_phase(hipDoubleComplex* phase, hipDoubleComplex* state, size_t N) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  hipDoubleComplex p, s, result;

  for (size_t i = idx; i < N; i += grid_size) {
    p = phase[i];
    s = state[i];
    // Complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
    result.x = p.x * s.x - p.y * s.y;
    result.y = p.x * s.y + p.y * s.x;
    state[i] = result;
    __syncthreads();
  }
}

// Apply checkerboard phase for centered FFT: (-1)^(sum(indices))
__global__ void apply_checkerboard(int n_dim, int Ns_max, int* Ns, int* strides, hipDoubleComplex* state,
                                   size_t N, size_t offset) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t global_idx;
  int inds[20];
  int sum_inds;
  double sign;

  for (size_t i = idx; i < N; i += grid_size) {
    global_idx = i + offset;

    // Compute n-dimensional indices
    size_t temp = global_idx;
    sum_inds = 0;
    for (int dim = 0; dim < n_dim; dim++) {
      inds[dim] = temp / strides[dim];
      temp = temp % strides[dim];
      sum_inds += inds[dim];
    }

    // (-1)^(sum of indices)
    sign = (sum_inds % 2 == 0) ? 1.0 : -1.0;
    state[i].x *= sign;
    state[i].y *= sign;
    __syncthreads();
  }
}

// Generate momentum-space mixer (kinetic energy weighted by t parameters)
__global__ void gen_momentum_mixer(int n_dim, int Ns_max, int* Ns, int* strides, double* ts,
                                   double* eigenvalues, double* mixer, size_t N, size_t offset) {
  size_t grid_size = blockDim.x * gridDim.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t global_idx;
  int inds[20];
  double m;
  int eigenvalue_offset;

  for (size_t i = idx; i < N; i += grid_size) {
    global_idx = i + offset;

    // Compute n-dimensional indices
    size_t temp = global_idx;
    for (int dim = 0; dim < n_dim; dim++) {
      inds[dim] = temp / strides[dim];
      temp = temp % strides[dim];
    }

    // Sum up t[dim] * eigenvalue[dim][index[dim]]
    m = 0.0;
    eigenvalue_offset = 0;
    for (int dim = 0; dim < n_dim; dim++) {
      m += ts[dim] * eigenvalues[eigenvalue_offset + inds[dim]];
      eigenvalue_offset += Ns[dim];
    }

    mixer[i] = m;
    __syncthreads();
  }
}

// Note: dense_one_norms, infinity_norm, vector_infinity_norm, inplace_vec_sum,
// b_scale, unity_spmm, regular_unity_spmm, non_scaled_unity_spmm,
// non_scaled_regular_unity_spmm, pack_send_values, unpack_rec_values
// kernels have been moved to sparse_propagators/src/kernels/

extern "C" {
void launch_expectation_value_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double* dout,
                                     hipDoubleComplex* da, double* db, size_t N) {
  hipLaunchKernelGGL((expectation_value), *grid, *block, shmem, stream, dout, da, db, N);
}
}

extern "C" {
void launch_state_norm_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double* dout,
                              hipDoubleComplex* da, size_t N) {
  hipLaunchKernelGGL((state_norm), *grid, *block, shmem, stream, dout, da, N);
}
}

extern "C" {
void launch_phase_shift_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, double gamma,
                               double* diag_operator, hipDoubleComplex* state, size_t N) {
  hipLaunchKernelGGL((phase_shift), *grid, *block, shmem, stream, gamma, diag_operator, state, N);
}
}

extern "C" {
void launch_fourier_normalise_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                     double norm_factor, hipDoubleComplex* state, size_t N) {
  hipLaunchKernelGGL((fourier_normalise), *grid, *block, shmem, stream, norm_factor, state, N);
}
}

extern "C" {
void launch_circulant_eigenvalues_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, int nnz,
                                         long* indexes, double* values, double* eigenvalues, size_t N) {
  hipLaunchKernelGGL((circulant_eigenvalues), *grid, *block, shmem, stream, nnz, indexes, values, eigenvalues,
                     N);
}
}

extern "C" {
void launch_distributed_circulant_eigenvalues_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                                     int nnz, long* indexes, double* values,
                                                     double* eigenvalues, size_t local_N, size_t global_N,
                                                     size_t offset) {
  hipLaunchKernelGGL((distributed_circulant_eigenvalues), *grid, *block, shmem, stream, nnz, indexes, values,
                     eigenvalues, local_N, global_N, offset);
}
}

extern "C" {
void launch_distributed_complete_graph_eigenvalues_kernel(dim3* grid, dim3* block, int shmem,
                                                          hipStream_t stream, double* eigenvalues,
                                                          size_t local_N, size_t global_N, size_t offset) {
  hipLaunchKernelGGL((distributed_complete_graph_eigenvalues), *grid, *block, shmem, stream, eigenvalues,
                     local_N, global_N, offset);
}
}

extern "C" {
void launch_complete_graph_eigenvalues_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                              double* eigenvalues, size_t N) {
  hipLaunchKernelGGL((complete_graph_eigenvalues), *grid, *block, shmem, stream, eigenvalues, N);
}
}

extern "C" {
void launch_n_dim_circulant_eigenvalues_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                               int n_dim, int Ns_max, int* Ns, double* graph_array,
                                               double* eigenvalues, size_t N) {
  hipLaunchKernelGGL((n_dim_circulant_eigenvalues), *grid, *block, shmem, stream, n_dim, Ns_max, Ns,
                     graph_array, eigenvalues, N);
}
}

extern "C" {
void launch_n_dim_complete_graph_eigenvalues_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                                    int n_dim, int Ns_max, int* Ns, double* eigenvalues,
                                                    size_t N) {
  hipLaunchKernelGGL((n_dim_complete_graph_eigenvalues), *grid, *block, shmem, stream, n_dim, Ns_max, Ns,
                     eigenvalues, N);
}
}

extern "C" {
void launch_gen_composite_mixer_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, int n_dim,
                                       int Ns_max, int* Ns, int* strides, double* ts, double* eigenvalues,
                                       double* mixer, size_t N, size_t offset) {
  hipLaunchKernelGGL((composite_mixer), *grid, *block, shmem, stream, n_dim, Ns_max, Ns, strides, ts,
                     eigenvalues, mixer, N, offset);
}
}

extern "C" {
void launch_constant_phase_shift_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                        double* diag_operator, hipDoubleComplex* state, size_t N) {
  hipLaunchKernelGGL((constant_phase_shift), *grid, *block, shmem, stream, diag_operator, state, N);
}
}

extern "C" {
void launch_init_x_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, hipDoubleComplex* X, int N,
                          int l, int seed) {
  hipLaunchKernelGGL((init_x), *grid, *block, shmem, stream, X, N, l, seed);
}
}

// Momentum propagator launch wrappers

extern "C" {
void launch_n_dim_momentum_eigenvalues_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                              int n_dim, int Ns_max, int* Ns, double* minsk, double* deltask,
                                              double* eigenvalues, int N) {
  hipLaunchKernelGGL((n_dim_momentum_eigenvalues), *grid, *block, shmem, stream, n_dim, Ns_max, Ns, minsk,
                     deltask, eigenvalues, N);
}
}

extern "C" {
void launch_gen_phase_factors_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, int n_dim,
                                     int Ns_max, int* Ns, int* strides, double* mins_target,
                                     double* deltas_source, double* mins_source, hipDoubleComplex* phase_out,
                                     size_t N, size_t offset, int direction) {
  hipLaunchKernelGGL((gen_phase_factors), *grid, *block, shmem, stream, n_dim, Ns_max, Ns, strides,
                     mins_target, deltas_source, mins_source, phase_out, N, offset, direction);
}
}

extern "C" {
void launch_apply_complex_phase_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream,
                                       hipDoubleComplex* phase, hipDoubleComplex* state, size_t N) {
  hipLaunchKernelGGL((apply_complex_phase), *grid, *block, shmem, stream, phase, state, N);
}
}

extern "C" {
void launch_apply_checkerboard_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, int n_dim,
                                      int Ns_max, int* Ns, int* strides, hipDoubleComplex* state, size_t N,
                                      size_t offset) {
  hipLaunchKernelGGL((apply_checkerboard), *grid, *block, shmem, stream, n_dim, Ns_max, Ns, strides, state, N,
                     offset);
}
}

extern "C" {
void launch_gen_momentum_mixer_kernel(dim3* grid, dim3* block, int shmem, hipStream_t stream, int n_dim,
                                      int Ns_max, int* Ns, int* strides, double* ts, double* eigenvalues,
                                      double* mixer, size_t N, size_t offset) {
  hipLaunchKernelGGL((gen_momentum_mixer), *grid, *block, shmem, stream, n_dim, Ns_max, Ns, strides, ts,
                     eigenvalues, mixer, N, offset);
}
}

// Note: launch wrappers for dense_one_norms, infinity_norm, vector_infinity_norm,
// inplace_vec_sum, b_scale, pack_send_values, unpack_rec_values, unity_spmm,
// regular_unity_spmm, non_scaled_unity_spmm, non_scaled_regular_unity_spmm
// have been moved to sparse_propagators/src/kernels/
