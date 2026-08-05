//------------------------------------------------------------------------------
// HIP kernels for the transverse-field propagator.
//
// Kernels:
//   1. local_pair_qubit_kernel   — fused per-qubit RX butterfly across the
//      entire local array (aligned-layout fast path).
//   2. local_pair_kernel         — segment-scoped (g, g+delta) butterfly
//      (segmented-layout boundary path).
//   3. local_pair_strided_kernel — fused strided butterfly across N aligned
//      2*bit_mask blocks (segmented-layout bulk path).
//   4. remote_update_kernel      — psi[j] = c*psi[j] + a*recv[j] after exchange.
//------------------------------------------------------------------------------

#include "hip_transverse_field_common.hpp"

//------------------------------------------------------------------------------
// 1. Fused local-pair kernel for an entire qubit q.
//
//    Processes n_pairs = local_i / 2 butterflies in a single launch. The
//    partner of local index `low` is `low | (1 << q)`. Mapping a thread id
//    pid -> low is done by inserting a 0 bit at position q.
//
//    Used only in the aligned layout where every pair at qubit q is local.
//------------------------------------------------------------------------------
__global__ void tf_local_pair_qubit_kernel(
    hipDoubleComplex* psi,
    long n_pairs,
    int q,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag)
{
    long pid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    long bit  = 1L << q;
    long mask = bit - 1L;
    long low  = ((pid & ~mask) << 1) | (pid & mask);
    long high = low | bit;

    hipDoubleComplex u = psi[low];
    hipDoubleComplex v = psi[high];

    psi[low]  = hipCadd(hipCmul(coeff_diag, u), hipCmul(coeff_offdiag, v));
    psi[high] = hipCadd(hipCmul(coeff_offdiag, u), hipCmul(coeff_diag, v));
}

extern "C" void launch_tf_local_pair_qubit_kernel(
    hipDoubleComplex* psi,
    long n_pairs,
    int q,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag,
    hipStream_t stream)
{
    if (n_pairs <= 0) return;
    int threads = TF_BLOCKSIZE;
    int blocks = (int)((n_pairs + threads - 1) / threads);
    hipLaunchKernelGGL(tf_local_pair_qubit_kernel, dim3(blocks), dim3(threads), 0, stream,
                       psi, n_pairs, q, coeff_diag, coeff_offdiag);
}

//------------------------------------------------------------------------------
// 2. Segment-scoped local-pair kernel: applies the RX butterfly to pairs
//    (g, g+delta) for a contiguous segment [g0, g0+count-1].
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
// 3. Strided block local-pair kernel: applies the RX butterfly to every
//    pair (low, low + bit_mask) for `n_pairs` consecutive lower-half slots
//    of 2*bit_mask-aligned blocks starting at psi[base_local].
//
//    Layout: each block of size 2*bit_mask has bit_mask lower-half slots
//    followed by bit_mask upper-half slots. Thread pid maps to:
//      low = base_local + (pid / bit_mask) * (2*bit_mask) + (pid % bit_mask)
//      high = low + bit_mask
//
//    Used by the segmented-layout path to fuse the bulk of the local-pair
//    work into a single launch per qubit.
//------------------------------------------------------------------------------
__global__ void tf_local_pair_strided_kernel(
    hipDoubleComplex* psi,
    long base_local,
    long n_pairs,
    long bit_mask,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag)
{
    long pid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;

    long blk    = pid / bit_mask;
    long inblk  = pid - blk * bit_mask;
    long low    = base_local + blk * (2L * bit_mask) + inblk;
    long high   = low + bit_mask;

    hipDoubleComplex u = psi[low];
    hipDoubleComplex v = psi[high];

    psi[low]  = hipCadd(hipCmul(coeff_diag, u), hipCmul(coeff_offdiag, v));
    psi[high] = hipCadd(hipCmul(coeff_offdiag, u), hipCmul(coeff_diag, v));
}

extern "C" void launch_tf_local_pair_strided_kernel(
    hipDoubleComplex* psi,
    long base_local,
    long n_pairs,
    long bit_mask,
    hipDoubleComplex coeff_diag,
    hipDoubleComplex coeff_offdiag,
    hipStream_t stream)
{
    if (n_pairs <= 0) return;
    int threads = TF_BLOCKSIZE;
    int blocks = (int)((n_pairs + threads - 1) / threads);
    hipLaunchKernelGGL(tf_local_pair_strided_kernel, dim3(blocks), dim3(threads), 0, stream,
                       psi, base_local, n_pairs, bit_mask, coeff_diag, coeff_offdiag);
}

//------------------------------------------------------------------------------
// 4. Remote update kernel: after MPI_Sendrecv delivers partner data into
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
