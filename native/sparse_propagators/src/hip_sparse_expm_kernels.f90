!------------------------------------------------------------------------------
! HIP Sparse Matrix Exponential Kernel Interfaces
!
! Fortran interfaces for HIP kernels defined in the kernels/ directory:
!   - kernels/hip_common.hpp            - Common includes, BLOCKSIZE, device helpers
!   - kernels/hip_vector_kernels.cpp    - Vector ops: norm, copy, scale, axpy
!   - kernels/hip_spmv_kernels.cpp      - Two-phase distributed SpMV
!   - kernels/hip_chebyshev_kernels.cpp - Chebyshev recurrence + accumulate
!   - kernels/hip_reduction_kernels.cpp - Gershgorin, column norms, reduce_max
!
! These provide bind(c) interfaces for calling GPU kernels from Fortran.
!------------------------------------------------------------------------------

module hip_sparse_expm_kernels

    use, intrinsic :: iso_c_binding
    use hipfort_types

    implicit none

    private

    ! Public interfaces for all kernels
    public :: launch_vector_infinity_norm_kernel
    public :: launch_inplace_vec_sum_kernel
    public :: launch_b_scale_kernel
    public :: launch_complex_axpy_kernel
    public :: launch_complex_scale_kernel
    public :: launch_real_scale_kernel
    public :: launch_vec_copy_kernel

    ! Two-phase distributed SpMV kernels
    public :: launch_spmv_local_weighted_kernel
    public :: launch_spmv_local_unit_kernel
    public :: launch_spmv_remote_weighted_kernel
    public :: launch_spmv_remote_unit_kernel
    public :: launch_pack_send_buf_kernel

    ! Two-phase distributed Chebyshev recurrence kernels
    public :: launch_chebyshev_local_weighted_kernel
    public :: launch_chebyshev_local_unit_kernel
    public :: launch_chebyshev_remote_weighted_kernel
    public :: launch_chebyshev_remote_unit_kernel
    public :: launch_chebyshev_accumulate_kernel

    ! Spectral bound and reduction kernels
    public :: launch_gershgorin_bound_kernel
    public :: launch_gershgorin_bound_unit_kernel
    public :: launch_reduce_max_kernel
    public :: launch_csr_column_one_norms_kernel
    public :: launch_csr_column_one_norms_unit_kernel

    ! Legacy SpMM kernels (from original hip_kernels.cpp)
    public :: launch_unity_spmm
    public :: launch_dense_one_norms_kernel
    public :: launch_infinity_norm_kernel
    public :: launch_pack_send_values_kernel
    public :: launch_unpack_rec_values_kernel

    interface

        !----------------------------------------------------------------------
        ! Vector infinity norm: computes max|v[i]| with block-level reduction
        ! Output: infnorm array of size grid%x, needs final reduction
        !----------------------------------------------------------------------
        subroutine launch_vector_infinity_norm_kernel(grid, block, shmem, stream, &
                                                      infnorm, v, N) bind(c, name='launch_vector_infinity_norm_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: infnorm ! double*
            type(c_ptr), value :: v ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_vector_infinity_norm_kernel

        !----------------------------------------------------------------------
        ! In-place vector sum: X = X + Y
        !----------------------------------------------------------------------
        subroutine launch_inplace_vec_sum_kernel(grid, block, shmem, stream, &
                                                 X, Y, N) bind(c, name='launch_inplace_vec_sum_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: X ! hipDoubleComplex*
            type(c_ptr), value :: Y ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_inplace_vec_sum_kernel

        !----------------------------------------------------------------------
        ! Taylor series scaling: X = X / (s * j)
        !----------------------------------------------------------------------
        subroutine launch_b_scale_kernel(grid, block, shmem, stream, &
                                         X, s, j, N) bind(c, name='launch_b_scale_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: X ! hipDoubleComplex*
            integer(c_int), value :: s, j
            integer(c_size_t), value :: N
        end subroutine launch_b_scale_kernel

        !----------------------------------------------------------------------
        ! Complex AXPY: y = alpha * x + y
        !----------------------------------------------------------------------
        subroutine launch_complex_axpy_kernel(grid, block, shmem, stream, &
                                              alpha, x, y, N) bind(c, name='launch_complex_axpy_kernel')
            import :: c_ptr, c_int, c_size_t, c_double_complex, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            complex(c_double_complex), value :: alpha
            type(c_ptr), value :: x ! hipDoubleComplex*
            type(c_ptr), value :: y ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_complex_axpy_kernel

        !----------------------------------------------------------------------
        ! Complex scale: x = alpha * x
        !----------------------------------------------------------------------
        subroutine launch_complex_scale_kernel(grid, block, shmem, stream, &
                                               alpha, x, N) bind(c, name='launch_complex_scale_kernel')
            import :: c_ptr, c_int, c_size_t, c_double_complex, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            complex(c_double_complex), value :: alpha
            type(c_ptr), value :: x ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_complex_scale_kernel

        !----------------------------------------------------------------------
        ! Real scale: x = alpha * x
        !----------------------------------------------------------------------
        subroutine launch_real_scale_kernel(grid, block, shmem, stream, &
                                            alpha, x, N) bind(c, name='launch_real_scale_kernel')
            import :: c_ptr, c_int, c_size_t, c_double, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            real(c_double), value :: alpha
            type(c_ptr), value :: x ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_real_scale_kernel

        !----------------------------------------------------------------------
        ! Vector copy: y = x
        !----------------------------------------------------------------------
        subroutine launch_vec_copy_kernel(grid, block, shmem, stream, &
                                          x, y, N) bind(c, name='launch_vec_copy_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: x ! hipDoubleComplex*
            type(c_ptr), value :: y ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_vec_copy_kernel

        !======================================================================
        ! TWO-PHASE DISTRIBUTED SpMV KERNELS (halo-based)
        !======================================================================

        !----------------------------------------------------------------------
        ! Phase 1 Local SpMV (weighted): y = A_local * x_local
        ! Reads diagonal entries from x_local using col_halo (0-based local
        ! indices) and the precomputed [diag_lo, diag_hi] range.
        !----------------------------------------------------------------------
        subroutine launch_spmv_local_weighted_kernel(grid, block, shmem, stream, &
                                                     row_starts, col_halo, values, diag_lo, diag_hi, &
                                                     x_local, y, local_rows) &
            bind(c, name='launch_spmv_local_weighted_kernel')
            import :: c_ptr, c_int, c_long, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: values ! hipDoubleComplex*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: x_local ! hipDoubleComplex*
            type(c_ptr), value :: y ! hipDoubleComplex*
            integer(c_long), value :: local_rows
        end subroutine launch_spmv_local_weighted_kernel

        !----------------------------------------------------------------------
        ! Phase 1 Local SpMV (unit weight): y = A_local * x_local
        !----------------------------------------------------------------------
        subroutine launch_spmv_local_unit_kernel(grid, block, shmem, stream, &
                                                 row_starts, col_halo, diag_lo, diag_hi, x_local, y, &
                                                 local_rows) &
            bind(c, name='launch_spmv_local_unit_kernel')
            import :: c_ptr, c_int, c_long, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: x_local ! hipDoubleComplex*
            type(c_ptr), value :: y ! hipDoubleComplex*
            integer(c_long), value :: local_rows
        end subroutine launch_spmv_local_unit_kernel

        !----------------------------------------------------------------------
        ! Phase 2 Remote SpMV (weighted): y = scalar * (y + A_off * recv_buf)
        ! Reads off-diagonal entries from recv_buf using col_halo - n_local.
        !----------------------------------------------------------------------
        subroutine launch_spmv_remote_weighted_kernel(grid, block, shmem, stream, &
                                                      row_starts, col_halo, values, diag_lo, diag_hi, &
                                                      recv_buf, y, scalar, n_local, local_rows) &
            bind(c, name='launch_spmv_remote_weighted_kernel')
            import :: c_ptr, c_int, c_long, c_double_complex, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: values ! hipDoubleComplex*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: recv_buf ! hipDoubleComplex*
            type(c_ptr), value :: y ! hipDoubleComplex*
            complex(c_double_complex), value :: scalar
            integer(c_long), value :: n_local, local_rows
        end subroutine launch_spmv_remote_weighted_kernel

        !----------------------------------------------------------------------
        ! Phase 2 Remote SpMV (unit weight)
        !----------------------------------------------------------------------
        subroutine launch_spmv_remote_unit_kernel(grid, block, shmem, stream, &
                                                  row_starts, col_halo, diag_lo, diag_hi, recv_buf, y, &
                                                  scalar, n_local, local_rows) &
            bind(c, name='launch_spmv_remote_unit_kernel')
            import :: c_ptr, c_int, c_long, c_double_complex, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: recv_buf ! hipDoubleComplex*
            type(c_ptr), value :: y ! hipDoubleComplex*
            complex(c_double_complex), value :: scalar
            integer(c_long), value :: n_local, local_rows
        end subroutine launch_spmv_remote_unit_kernel

        !----------------------------------------------------------------------
        ! Pack send buffer: send_buf[i] = x_local[send_offsets[i]]
        !----------------------------------------------------------------------
        subroutine launch_pack_send_buf_kernel(grid, block, shmem, stream, &
                                               x_local, send_offsets, send_buf, total_send) &
            bind(c, name='launch_pack_send_buf_kernel')
            import :: c_ptr, c_int, c_long, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: x_local ! hipDoubleComplex*
            type(c_ptr), value :: send_offsets ! long*
            type(c_ptr), value :: send_buf ! hipDoubleComplex*
            integer(c_long), value :: total_send
        end subroutine launch_pack_send_buf_kernel

        !======================================================================
        ! TWO-PHASE DISTRIBUTED CHEBYSHEV RECURRENCE KERNELS
        !======================================================================

        !----------------------------------------------------------------------
        ! Chebyshev Phase 1 Local (weighted): Aw_k = A_local * w_k_local
        !----------------------------------------------------------------------
        subroutine launch_chebyshev_local_weighted_kernel(grid, block, shmem, stream, &
                                                          inv_M, row_starts, col_halo, values, diag_lo, &
                                                          diag_hi, w_k_local, Aw_k, local_rows) &
            bind(c, name='launch_chebyshev_local_weighted_kernel')
            import :: c_ptr, c_int, c_long, c_double, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            real(c_double), value :: inv_M
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: values ! hipDoubleComplex*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: w_k_local ! hipDoubleComplex*
            type(c_ptr), value :: Aw_k ! hipDoubleComplex*
            integer(c_long), value :: local_rows
        end subroutine launch_chebyshev_local_weighted_kernel

        !----------------------------------------------------------------------
        ! Chebyshev Phase 1 Local (unit): Aw_k = A_local * w_k_local
        !----------------------------------------------------------------------
        subroutine launch_chebyshev_local_unit_kernel(grid, block, shmem, stream, &
                                                      inv_M, row_starts, col_halo, diag_lo, diag_hi, &
                                                      w_k_local, Aw_k, local_rows) &
            bind(c, name='launch_chebyshev_local_unit_kernel')
            import :: c_ptr, c_int, c_long, c_double, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            real(c_double), value :: inv_M
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: w_k_local ! hipDoubleComplex*
            type(c_ptr), value :: Aw_k ! hipDoubleComplex*
            integer(c_long), value :: local_rows
        end subroutine launch_chebyshev_local_unit_kernel

        !----------------------------------------------------------------------
        ! Chebyshev Phase 2 Remote (weighted):
        !   w_kp1 = 2 * inv_M * (Aw_k + A_off * recv_buf) - w_km1
        !----------------------------------------------------------------------
        subroutine launch_chebyshev_remote_weighted_kernel(grid, block, shmem, stream, &
                                                           inv_M, row_starts, col_halo, values, diag_lo, &
                                                           diag_hi, recv_buf, Aw_k, w_km1, w_kp1, n_local, &
                                                           local_rows) &
            bind(c, name='launch_chebyshev_remote_weighted_kernel')
            import :: c_ptr, c_int, c_long, c_double, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            real(c_double), value :: inv_M
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: values ! hipDoubleComplex*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: recv_buf ! hipDoubleComplex*
            type(c_ptr), value :: Aw_k ! hipDoubleComplex*
            type(c_ptr), value :: w_km1 ! hipDoubleComplex*
            type(c_ptr), value :: w_kp1 ! hipDoubleComplex*
            integer(c_long), value :: n_local, local_rows
        end subroutine launch_chebyshev_remote_weighted_kernel

        !----------------------------------------------------------------------
        ! Chebyshev Phase 2 Remote (unit):
        !   w_kp1 = 2 * inv_M * (Aw_k + A_off * recv_buf) - w_km1
        !----------------------------------------------------------------------
        subroutine launch_chebyshev_remote_unit_kernel(grid, block, shmem, stream, &
                                                       inv_M, row_starts, col_halo, diag_lo, diag_hi, &
                                                       recv_buf, Aw_k, w_km1, w_kp1, n_local, local_rows) &
            bind(c, name='launch_chebyshev_remote_unit_kernel')
            import :: c_ptr, c_int, c_long, c_double, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            real(c_double), value :: inv_M
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_halo ! long*
            type(c_ptr), value :: diag_lo ! long*
            type(c_ptr), value :: diag_hi ! long*
            type(c_ptr), value :: recv_buf ! hipDoubleComplex*
            type(c_ptr), value :: Aw_k ! hipDoubleComplex*
            type(c_ptr), value :: w_km1 ! hipDoubleComplex*
            type(c_ptr), value :: w_kp1 ! hipDoubleComplex*
            integer(c_long), value :: n_local, local_rows
        end subroutine launch_chebyshev_remote_unit_kernel

        !----------------------------------------------------------------------
        ! Chebyshev accumulate: C = C + coeff * w
        !----------------------------------------------------------------------
        subroutine launch_chebyshev_accumulate_kernel(grid, block, shmem, stream, &
                                                      coeff, w, C, N) bind(c, name='launch_chebyshev_accumulate_kernel')
            import :: c_ptr, c_int, c_size_t, c_double_complex, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            complex(c_double_complex), value :: coeff
            type(c_ptr), value :: w ! hipDoubleComplex*
            type(c_ptr), value :: C ! hipDoubleComplex*
            integer(c_size_t), value :: N
        end subroutine launch_chebyshev_accumulate_kernel

        !----------------------------------------------------------------------
        ! Gershgorin bound (weighted): computes |A_ii| + sum|A_ij| for each row
        !----------------------------------------------------------------------
        subroutine launch_gershgorin_bound_kernel(grid, block, shmem, stream, &
                                                  row_starts, col_inds, values, row_bounds, local_rows, offset) &
            bind(c, name='launch_gershgorin_bound_kernel')
            import :: c_ptr, c_int, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_inds ! int*
            type(c_ptr), value :: values ! hipDoubleComplex*
            type(c_ptr), value :: row_bounds ! double*
            integer(c_int), value :: local_rows, offset
        end subroutine launch_gershgorin_bound_kernel

        !----------------------------------------------------------------------
        ! Gershgorin bound (unit weight): computes 1 + nnz_row - 1 for each row
        !----------------------------------------------------------------------
        subroutine launch_gershgorin_bound_unit_kernel(grid, block, shmem, stream, &
                                                       row_starts, col_inds, row_bounds, local_rows, offset) &
            bind(c, name='launch_gershgorin_bound_unit_kernel')
            import :: c_ptr, c_int, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_inds ! int*
            type(c_ptr), value :: row_bounds ! double*
            integer(c_int), value :: local_rows, offset
        end subroutine launch_gershgorin_bound_unit_kernel

        !----------------------------------------------------------------------
        ! Reduce max: finds maximum value in array with block-level reduction
        !----------------------------------------------------------------------
        subroutine launch_reduce_max_kernel(grid, block, shmem, stream, &
                                            data, result, N) bind(c, name='launch_reduce_max_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: data ! double*
            type(c_ptr), value :: result ! double*
            integer(c_size_t), value :: N
        end subroutine launch_reduce_max_kernel

        !----------------------------------------------------------------------
        ! CSR column 1-norms (weighted): sum of |A_ij| for each column
        !----------------------------------------------------------------------
        subroutine launch_csr_column_one_norms_kernel(grid, block, shmem, stream, &
                                                      row_starts, col_inds, values, col_norms, num_rows, num_cols) &
            bind(c, name='launch_csr_column_one_norms_kernel')
            import :: c_ptr, c_int, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_inds ! int*
            type(c_ptr), value :: values ! hipDoubleComplex*
            type(c_ptr), value :: col_norms ! double*
            integer(c_int), value :: num_rows, num_cols
        end subroutine launch_csr_column_one_norms_kernel

        !----------------------------------------------------------------------
        ! CSR column 1-norms (unit weight): count of nonzeros per column
        !----------------------------------------------------------------------
        subroutine launch_csr_column_one_norms_unit_kernel(grid, block, shmem, stream, &
                                                           row_starts, col_inds, col_norms, num_rows, num_cols) &
            bind(c, name='launch_csr_column_one_norms_unit_kernel')
            import :: c_ptr, c_int, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: row_starts ! long*
            type(c_ptr), value :: col_inds ! int*
            type(c_ptr), value :: col_norms ! double*
            integer(c_int), value :: num_rows, num_cols
        end subroutine launch_csr_column_one_norms_unit_kernel

        !----------------------------------------------------------------------
        ! Unity SpMM: vec_R = (i * alpha) * A * vec_L (unit edge weights)
        ! From original hip_kernels.cpp
        !----------------------------------------------------------------------
        subroutine launch_unity_spmm(grid, block, shmem, stream, &
                                     alpha, row_starts, col_inds, vec_L, vec_R, m, n, local_i) &
            bind(c, name='launch_unity_spmm')
            import :: c_ptr, c_int, c_long, c_double, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            real(c_double), value :: alpha
            type(c_ptr), value :: row_starts ! int*
            type(c_ptr), value :: col_inds ! int*
            type(c_ptr), value :: vec_L ! hipDoubleComplex*
            type(c_ptr), value :: vec_R ! hipDoubleComplex*
            integer(c_long), value :: m
            integer(c_int), value :: n
            integer(c_long), value :: local_i
        end subroutine launch_unity_spmm

        !----------------------------------------------------------------------
        ! Dense 1-norms: computes 1-norm for l vectors of length N
        !----------------------------------------------------------------------
        subroutine launch_dense_one_norms_kernel(grid, block, shmem, stream, &
                                                 result, X, N, l) bind(c, name='launch_dense_one_norms_kernel')
            import :: c_ptr, c_int, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: result ! double*
            type(c_ptr), value :: X ! hipDoubleComplex*
            integer(c_int), value :: N, l
        end subroutine launch_dense_one_norms_kernel

        !----------------------------------------------------------------------
        ! Infinity norm (row-wise): computes row sums for matrix stored as columns
        !----------------------------------------------------------------------
        subroutine launch_infinity_norm_kernel(grid, block, shmem, stream, &
                                               result, X, N, l) bind(c, name='launch_infinity_norm_kernel')
            import :: c_ptr, c_int, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: result ! double*
            type(c_ptr), value :: X ! hipDoubleComplex*
            integer(c_int), value :: N, l
        end subroutine launch_infinity_norm_kernel

        !----------------------------------------------------------------------
        ! Pack send values: gather values for MPI communication
        !----------------------------------------------------------------------
        subroutine launch_pack_send_values_kernel(grid, block, shmem, stream, &
                                                  send_values, source, RHS_send_inds, l, N, pad, num_send) &
            bind(c, name='launch_pack_send_values_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: send_values ! hipDoubleComplex*
            type(c_ptr), value :: source ! hipDoubleComplex*
            type(c_ptr), value :: RHS_send_inds ! int*
            integer(c_int), value :: l
            integer(c_size_t), value :: N
            integer(c_int), value :: pad, num_send
        end subroutine launch_pack_send_values_kernel

        !----------------------------------------------------------------------
        ! Unpack recv values: scatter values after MPI communication
        !----------------------------------------------------------------------
        subroutine launch_unpack_rec_values_kernel(grid, block, shmem, stream, &
                                                   target, rec_values, l, N, pad, num_rec) &
            bind(c, name='launch_unpack_rec_values_kernel')
            import :: c_ptr, c_int, c_size_t, dim3
            implicit none
            type(dim3), intent(in) :: grid, block
            integer(c_int), value :: shmem
            type(c_ptr), value :: stream
            type(c_ptr), value :: target ! hipDoubleComplex*
            type(c_ptr), value :: rec_values ! hipDoubleComplex*
            integer(c_int), value :: l
            integer(c_size_t), value :: N
            integer(c_int), value :: pad, num_rec
        end subroutine launch_unpack_rec_values_kernel

    end interface

end module hip_sparse_expm_kernels
