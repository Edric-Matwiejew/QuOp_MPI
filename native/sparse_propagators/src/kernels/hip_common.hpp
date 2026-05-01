// Common definitions and device helper functions for sparse expm HIP kernels
#ifndef HIP_SPARSE_EXPM_COMMON_HPP
#define HIP_SPARSE_EXPM_COMMON_HPP

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

// Block size for all kernels - should match shared memory constraints
constexpr int BLOCKSIZE = 256;

//==============================================================================
// DEVICE HELPER FUNCTIONS
//==============================================================================

// Binary search: find first position where col_indexes[pos] >= val
__device__ inline long lower_bound_dev(long* col_indexes, long lo, long hi, long val) {
  long mid;
  while (lo < hi) {
    mid = (lo + hi) / 2;
    if (col_indexes[mid] < val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Binary search: find first position where col_indexes[pos] > val
__device__ inline long upper_bound_dev(long* col_indexes, long lo, long hi, long val) {
  long mid;
  while (lo < hi) {
    mid = (lo + hi) / 2;
    if (col_indexes[mid] <= val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

#endif // HIP_SPARSE_EXPM_COMMON_HPP
