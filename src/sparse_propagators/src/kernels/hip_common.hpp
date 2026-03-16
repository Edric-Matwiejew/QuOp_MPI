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

// Hash lookup for remote column position
// Returns 0-based position in recv_buf_sorted, or -1 if not found
__device__ inline long hash_lookup_dev(long col, long* hash_keys, long* hash_vals, long hash_size) {
  // Knuth's golden ratio multiplier
  const long HASH_MULT = 2654435769L;
  const long MASK32 = 0xFFFFFFFFL;

  long folded = (col ^ (col >> 32)) & MASK32;
  long hash_pos = ((folded * HASH_MULT) % hash_size); // 0-based hash position
  if (hash_pos < 0)
    hash_pos += hash_size;

  for (long probe = 0; probe < hash_size; probe++) {
    if (hash_keys[hash_pos] == col) {
      return hash_vals[hash_pos]; // Returns 0-based position
    } else if (hash_keys[hash_pos] < 0) {
      return -1; // Not found
    }
    hash_pos = (hash_pos + 1) % hash_size;
  }
  return -1; // Not found
}

#endif // HIP_SPARSE_EXPM_COMMON_HPP
