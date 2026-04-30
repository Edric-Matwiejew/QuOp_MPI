//------------------------------------------------------------------------------
// HIP kernels for the transverse-field propagator.
//
// Three kernels:
//   1. local_pair_kernel  — in-place (g, g+delta) butterfly update
//   2. remote_update_kernel — psi[j] = c*psi[j] + a*recv[j] after exchange
//   3. pack_send_kernel   — copy segment from psi into contiguous send buffer
//------------------------------------------------------------------------------

#include "hip_transverse_field_common.hpp"

//------------------------------------------------------------------------------
// 1. Local pair kernel: applies the RX butterfly to pairs (g, g+delta)
//    for a contiguous segment [g0, g1] within local memory.
//
//    psi is indexed from local offset 0 (C-style).
//    lb_global is the global index of psi[0].
//------------------------------------------------------------------------------
__global__ void tf_local_pair_kernel(
    hipDoubleComplex* psi,
    long lb_global,
    long g0,
    long count,
    long delta,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag)
{
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    long g = g0 + tid;
    long local_u = g - lb_global;
    long local_v = g + delta - lb_global;

    hipDoubleComplex u = psi[local_u];
    hipDoubleComplex v = psi[local_v];

    psi[local_u] = hipCadd(hipCmul(coeff_diag, u), hipCmul(coeff_offdiag, v));
    psi[local_v] = hipCadd(hipCmul(coeff_offdiag, u), hipCmul(coeff_diag, v));
}

extern "C" void launch_tf_local_pair_kernel(
    hipDoubleComplex* psi,
    long lb_global,
    long g0,
    long count,
    long delta,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag,
    hipStream_t stream)
{
    if (count <= 0) return;
    int threads = TF_BLOCKSIZE;
    int blocks = (int)((count + threads - 1) / threads);
    hipLaunchKernelGGL(tf_local_pair_kernel, dim3(blocks), dim3(threads), 0, stream,
                       psi, lb_global, g0, count, delta, coeff_diag, coeff_offdiag);
}

//------------------------------------------------------------------------------
// 2. Remote update kernel: after MPI_Sendrecv delivers partner data into
//    recvbuf, apply psi[j] = coeff_diag * psi[j] + coeff_offdiag * recvbuf[j]
//    for count elements starting at psi[local0].
//------------------------------------------------------------------------------
__global__ void tf_remote_update_kernel(
    hipDoubleComplex* psi,
    const hipDoubleComplex* recvbuf,
    long local0,
    long count,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag)
{
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    hipDoubleComplex p = psi[local0 + tid];
    hipDoubleComplex r = recvbuf[tid];

    psi[local0 + tid] = hipCadd(hipCmul(coeff_diag, p), hipCmul(coeff_offdiag, r));
}

extern "C" void launch_tf_remote_update_kernel(
    hipDoubleComplex* psi,
    const hipDoubleComplex* recvbuf,
    long local0,
    long count,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag,
    hipStream_t stream)
{
    if (count <= 0) return;
    int threads = TF_BLOCKSIZE;
    int blocks = (int)((count + threads - 1) / threads);
    hipLaunchKernelGGL(tf_remote_update_kernel, dim3(blocks), dim3(threads), 0, stream,
                       psi, recvbuf, local0, count, coeff_diag, coeff_offdiag);
}

//------------------------------------------------------------------------------
// 3. Pack send kernel: copy count elements from psi[local0..] into sendbuf[0..].
//    Used for non-GPU-aware MPI staging (device → device copy before D→H).
//    Also used for GPU-aware MPI to isolate the send region.
//------------------------------------------------------------------------------
__global__ void tf_pack_send_kernel(
    hipDoubleComplex* sendbuf,
    const hipDoubleComplex* psi,
    long local0,
    long count)
{
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= count) return;

    sendbuf[tid] = psi[local0 + tid];
}

extern "C" void launch_tf_pack_send_kernel(
    hipDoubleComplex* sendbuf,
    const hipDoubleComplex* psi,
    long local0,
    long count,
    hipStream_t stream)
{
    if (count <= 0) return;
    int threads = TF_BLOCKSIZE;
    int blocks = (int)((count + threads - 1) / threads);
    hipLaunchKernelGGL(tf_pack_send_kernel, dim3(blocks), dim3(threads), 0, stream,
                       sendbuf, psi, local0, count);
}
