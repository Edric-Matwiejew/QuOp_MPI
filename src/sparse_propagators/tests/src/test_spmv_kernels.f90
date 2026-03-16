! test_spmv_kernels.f90
! Unit tests for HIP SpMV (Sparse Matrix-Vector) operation kernels
!
! Tests:
!   1. spmv_local_weighted - local SpMV with weighted edges
!   2. spmv_local_unit - local SpMV with unit edge weights
!   3. spmv_remote_weighted - remote SpMV with hash table lookup (weighted)
!   4. spmv_remote_unit - remote SpMV with hash table lookup (unit)
!   5. reorder_recv_buf - reorder receive buffer by permutation
!   6. pack_send_buf - pack send buffer from local vector

program test_spmv_kernels
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_types
    use hip_sparse_expm_kernels, only: &
        launch_pack_send_buf_kernel, launch_reorder_recv_buf_kernel, launch_spmv_local_unit_kernel, &
        launch_spmv_local_weighted_kernel, launch_spmv_remote_unit_kernel, launch_spmv_remote_weighted_kernel
    use test_hash_utils, only: build_hash_table
    implicit none

    integer(int32), parameter :: BLOCKSIZE = 256
    real(real64), parameter :: tolerance = 1.0e-10_real64

    integer(int32) :: total_tests, passed_tests

    total_tests = 0
    passed_tests = 0

    write (*, *) "========================================"
    write (*, *) " SpMV Kernels Unit Tests"
    write (*, *) "========================================"

    call hipCheck(hipInit(0))

    call test_spmv_local_weighted()
    call test_spmv_local_unit()
    call test_spmv_remote_weighted()
    call test_spmv_remote_unit()
    call test_reorder_recv_buf()
    call test_pack_send_buf()

    write (*, *) ""
    write (*, *) "========================================"
    write (*, '(A,I0,A,I0,A)') " Results: ", passed_tests, "/", total_tests, " tests passed"
    write (*, *) "========================================"

    if (passed_tests == total_tests) then
        call exit(0)
    else
        call exit(1)
    end if

contains

    !--------------------------------------------------------------------------
    ! Test spmv_local_weighted: y = A_local * x_local (weighted edges)
    ! Interface: launch_spmv_local_weighted_kernel(grid, block, shmem, stream,
    !            row_starts, col_indexes, values, x_local, y, lb, ub, local_rows)
    !--------------------------------------------------------------------------
    subroutine test_spmv_local_weighted()
        integer(c_long), parameter :: N = 4
        integer(c_long), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_long), target :: col_indexes(10)
        complex(real64), target :: values(10), x_host(4), y_host(4)
        complex(real64), allocatable, target :: expected(:)
        type(c_ptr) :: row_starts_dev, col_indexes_dev, values_dev, x_dev, y_dev
        type(dim3) :: grid, block
        integer(int32) :: i, j
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: spmv_local_weighted..."

        ! 4x4 tridiagonal matrix in CSR (0-indexed column indices)
        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0_c_long, 1_c_long, 0_c_long, 1_c_long, 2_c_long, &
                       1_c_long, 2_c_long, 3_c_long, 2_c_long, 3_c_long]
        values = [cmplx(2.0_real64, 0.0_real64, real64), &
                  cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(3.0_real64, 0.0_real64, real64), &
                  cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(4.0_real64, 0.0_real64, real64), &
                  cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(5.0_real64, 0.0_real64, real64)]

        ! Input vector x = [1, 2, 3, 4]
        x_host = [cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(2.0_real64, 0.0_real64, real64), &
                  cmplx(3.0_real64, 0.0_real64, real64), &
                  cmplx(4.0_real64, 0.0_real64, real64)]

        ! Expected: y = A * x
        ! Row 0: 2*1 + 1*2 = 4
        ! Row 1: 1*1 + 3*2 + 1*3 = 10
        ! Row 2: 1*2 + 4*3 + 1*4 = 18
        ! Row 3: 1*3 + 5*4 = 23
        allocate (expected(4))
        expected = [cmplx(4.0_real64, 0.0_real64, real64), &
                    cmplx(10.0_real64, 0.0_real64, real64), &
                    cmplx(18.0_real64, 0.0_real64, real64), &
                    cmplx(23.0_real64, 0.0_real64, real64)]

        y_host = cmplx(0.0_real64, 0.0_real64, real64)

        ! Allocate and copy to device
        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 8, c_size_t)))
        call hipCheck(hipMalloc(values_dev, int(nnz * 16, c_size_t)))
        call hipCheck(hipMalloc(x_dev, int(N * 16, c_size_t)))
        call hipCheck(hipMalloc(y_dev, int(N * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(values_dev, c_loc(values), int(nnz * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        ! lb=0, ub=N-1 (inclusive), local_rows=N
        call launch_spmv_local_weighted_kernel(grid, block, 0, c_null_ptr, &
                                               row_starts_dev, col_indexes_dev, values_dev, x_dev, y_dev, &
                                               0_c_long, N - 1, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(y_host), y_dev, int(N * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(y_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(y_host(i)), aimag(y_host(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(values_dev))
        call hipCheck(hipFree(x_dev))
        call hipCheck(hipFree(y_dev))
        deallocate (expected)
    end subroutine test_spmv_local_weighted

    !--------------------------------------------------------------------------
    ! Test spmv_local_unit: y = A_local * x_local (unit edge weights)
    ! Interface: launch_spmv_local_unit_kernel(grid, block, shmem, stream,
    !            row_starts, col_indexes, x_local, y, lb, ub, local_rows)
    ! Note: no 'values' parameter
    !--------------------------------------------------------------------------
    subroutine test_spmv_local_unit()
        integer(c_long), parameter :: N = 4
        integer(c_long), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_long), target :: col_indexes(10)
        complex(real64), target :: x_host(4), y_host(4)
        complex(real64), allocatable, target :: expected(:)
        type(c_ptr) :: row_starts_dev, col_indexes_dev, x_dev, y_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: spmv_local_unit..."

        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0_c_long, 1_c_long, 0_c_long, 1_c_long, 2_c_long, &
                       1_c_long, 2_c_long, 3_c_long, 2_c_long, 3_c_long]

        x_host = [cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(2.0_real64, 0.0_real64, real64), &
                  cmplx(3.0_real64, 0.0_real64, real64), &
                  cmplx(4.0_real64, 0.0_real64, real64)]

        ! Expected: y = A * x with unit weights
        ! Row 0: 1*1 + 1*2 = 3
        ! Row 1: 1*1 + 1*2 + 1*3 = 6
        ! Row 2: 1*2 + 1*3 + 1*4 = 9
        ! Row 3: 1*3 + 1*4 = 7
        allocate (expected(4))
        expected = [cmplx(3.0_real64, 0.0_real64, real64), &
                    cmplx(6.0_real64, 0.0_real64, real64), &
                    cmplx(9.0_real64, 0.0_real64, real64), &
                    cmplx(7.0_real64, 0.0_real64, real64)]

        y_host = cmplx(0.0_real64, 0.0_real64, real64)

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 8, c_size_t)))
        call hipCheck(hipMalloc(x_dev, int(N * 16, c_size_t)))
        call hipCheck(hipMalloc(y_dev, int(N * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_spmv_local_unit_kernel(grid, block, 0, c_null_ptr, &
                                           row_starts_dev, col_indexes_dev, x_dev, y_dev, &
                                           0_c_long, N - 1, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(y_host), y_dev, int(N * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(y_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(y_host(i)), aimag(y_host(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(x_dev))
        call hipCheck(hipFree(y_dev))
        deallocate (expected)
    end subroutine test_spmv_local_unit

    !--------------------------------------------------------------------------
    ! Test spmv_remote_weighted: y = scalar * (y + A_remote * recv_buf)
    ! Interface: launch_spmv_remote_weighted_kernel(grid, block, shmem, stream,
    !            row_starts, col_indexes, values, recv_buf_sorted,
    !            hash_keys, hash_vals, hash_size, y, scalar, lb, ub, local_rows)
    !--------------------------------------------------------------------------
    subroutine test_spmv_remote_weighted()
        integer(c_long), parameter :: local_rows = 2_c_long
        integer(c_long), parameter :: lb = 0_c_long, ub = 1_c_long
        integer(c_long), parameter :: nnz_remote = 4
        integer(c_long), parameter :: hash_table_size = 16_c_long
        integer(c_long), target :: row_starts(3)
        integer(c_long), target :: col_indexes(4) ! Remote columns only
        complex(real64), target :: values(4), recv_buf(2), y_host(2)
        integer(c_long), target :: hash_keys(16), hash_vals(16)
        integer(c_long) :: remote_cols(2)
        complex(real64) :: scalar
        type(c_ptr) :: row_starts_dev, col_indexes_dev, values_dev
        type(c_ptr) :: recv_buf_dev, hash_keys_dev, hash_vals_dev, y_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64) :: expected(2)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: spmv_remote_weighted..."

        ! Remote matrix has 2 rows, referencing columns [2, 3] (outside local [0,1])
        ! CSR for remote contribution
        row_starts = [0_c_long, 2_c_long, 4_c_long] ! 2 entries per row
        col_indexes = [2_c_long, 3_c_long, 2_c_long, 3_c_long] ! Columns 2,3
        values = [cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(2.0_real64, 0.0_real64, real64), &
                  cmplx(3.0_real64, 0.0_real64, real64), &
                  cmplx(4.0_real64, 0.0_real64, real64)]

        ! recv_buf contains values for columns 2 and 3 after sorting
        ! Position 0 -> col 2, Position 1 -> col 3
        recv_buf = [cmplx(1.0_real64, 1.0_real64, real64), & ! Value for col 2
                    cmplx(2.0_real64, 0.0_real64, real64)] ! Value for col 3

        ! Build hash table: maps column -> 1-based position in recv_buf
        remote_cols = [2_c_long, 3_c_long]
        call build_hash_table(remote_cols, 2, hash_keys, hash_vals, hash_table_size)

        ! Initial y values (from local SpMV phase)
        y_host = [cmplx(10.0_real64, 0.0_real64, real64), &
                  cmplx(20.0_real64, 0.0_real64, real64)]

        scalar = cmplx(1.0_real64, 0.0_real64, real64) ! No scaling

        ! Expected: y = scalar * (y + A_remote * recv_buf)
        ! Row 0: 1*(10 + 1*(1+i) + 2*(2+0i)) = 1*(10 + 1+i + 4) = 15+i
        ! Row 1: 1*(20 + 3*(1+i) + 4*(2+0i)) = 1*(20 + 3+3i + 8) = 31+3i
        expected = [cmplx(15.0_real64, 1.0_real64, real64), &
                    cmplx(31.0_real64, 3.0_real64, real64)]

        call hipCheck(hipMalloc(row_starts_dev, int(3 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz_remote * 8, c_size_t)))
        call hipCheck(hipMalloc(values_dev, int(nnz_remote * 16, c_size_t)))
        call hipCheck(hipMalloc(recv_buf_dev, int(2 * 16, c_size_t)))
        call hipCheck(hipMalloc(hash_keys_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(hash_vals_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(y_dev, int(local_rows * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(3 * 8, c_size_t), hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz_remote * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(values_dev, c_loc(values), int(nnz_remote * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(recv_buf_dev, c_loc(recv_buf), int(2 * 16, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_keys_dev, c_loc(hash_keys), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_vals_dev, c_loc(hash_vals), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_spmv_remote_weighted_kernel(grid, block, 0, c_null_ptr, &
                                                row_starts_dev, col_indexes_dev, values_dev, recv_buf_dev, &
                                                hash_keys_dev, hash_vals_dev, hash_table_size, y_dev, scalar, &
                                                lb, ub, local_rows)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(y_host), y_dev, int(local_rows * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, local_rows
            if (abs(y_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(y_host(i)), aimag(y_host(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(values_dev))
        call hipCheck(hipFree(recv_buf_dev))
        call hipCheck(hipFree(hash_keys_dev))
        call hipCheck(hipFree(hash_vals_dev))
        call hipCheck(hipFree(y_dev))
    end subroutine test_spmv_remote_weighted

    !--------------------------------------------------------------------------
    ! Test spmv_remote_unit: y = scalar * (y + A_remote * recv_buf) with unit weights
    ! Interface: launch_spmv_remote_unit_kernel(grid, block, shmem, stream,
    !            row_starts, col_indexes, recv_buf_sorted,
    !            hash_keys, hash_vals, hash_size, y, scalar, lb, ub, local_rows)
    ! Note: no 'values' parameter
    !--------------------------------------------------------------------------
    subroutine test_spmv_remote_unit()
        integer(c_long), parameter :: local_rows = 2_c_long
        integer(c_long), parameter :: lb = 0_c_long, ub = 1_c_long
        integer(c_long), parameter :: nnz_remote = 4
        integer(c_long), parameter :: hash_table_size = 16_c_long
        integer(c_long), target :: row_starts(3)
        integer(c_long), target :: col_indexes(4)
        complex(real64), target :: recv_buf(2), y_host(2)
        integer(c_long), target :: hash_keys(16), hash_vals(16)
        integer(c_long) :: remote_cols(2)
        complex(real64) :: scalar
        type(c_ptr) :: row_starts_dev, col_indexes_dev
        type(c_ptr) :: recv_buf_dev, hash_keys_dev, hash_vals_dev, y_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64) :: expected(2)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: spmv_remote_unit..."

        row_starts = [0_c_long, 2_c_long, 4_c_long]
        col_indexes = [2_c_long, 3_c_long, 2_c_long, 3_c_long]

        recv_buf = [cmplx(1.0_real64, 1.0_real64, real64), &
                    cmplx(2.0_real64, 0.0_real64, real64)]

        ! Build hash table with correct hash function
        remote_cols = [2_c_long, 3_c_long]
        call build_hash_table(remote_cols, 2, hash_keys, hash_vals, hash_table_size)

        y_host = [cmplx(10.0_real64, 0.0_real64, real64), &
                  cmplx(20.0_real64, 0.0_real64, real64)]

        scalar = cmplx(1.0_real64, 0.0_real64, real64)

        ! Expected with unit weights:
        ! Row 0: 1*(10 + 1*(1+i) + 1*(2+0i)) = 13+i
        ! Row 1: 1*(20 + 1*(1+i) + 1*(2+0i)) = 23+i
        expected = [cmplx(13.0_real64, 1.0_real64, real64), &
                    cmplx(23.0_real64, 1.0_real64, real64)]

        call hipCheck(hipMalloc(row_starts_dev, int(3 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz_remote * 8, c_size_t)))
        call hipCheck(hipMalloc(recv_buf_dev, int(2 * 16, c_size_t)))
        call hipCheck(hipMalloc(hash_keys_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(hash_vals_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(y_dev, int(local_rows * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(3 * 8, c_size_t), hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz_remote * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(recv_buf_dev, c_loc(recv_buf), int(2 * 16, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_keys_dev, c_loc(hash_keys), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_vals_dev, c_loc(hash_vals), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_spmv_remote_unit_kernel(grid, block, 0, c_null_ptr, &
                                            row_starts_dev, col_indexes_dev, recv_buf_dev, &
                                            hash_keys_dev, hash_vals_dev, hash_table_size, y_dev, scalar, &
                                            lb, ub, local_rows)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(y_host), y_dev, int(local_rows * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, local_rows
            if (abs(y_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(y_host(i)), aimag(y_host(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(recv_buf_dev))
        call hipCheck(hipFree(hash_keys_dev))
        call hipCheck(hipFree(hash_vals_dev))
        call hipCheck(hipFree(y_dev))
    end subroutine test_spmv_remote_unit

    !--------------------------------------------------------------------------
    ! Test reorder_recv_buf: recv_buf_sorted[i] = recv_buf[sort_perm[i]]
    ! Interface: launch_reorder_recv_buf_kernel(grid, block, shmem, stream,
    !            recv_buf, sort_perm, recv_buf_sorted, total_recv)
    ! Note: sort_perm uses 0-based indices
    !--------------------------------------------------------------------------
    subroutine test_reorder_recv_buf()
        integer(c_long), parameter :: N = 8_c_long
        complex(real64), target :: recv_buf(8), recv_buf_sorted(8)
        integer(c_long), target :: sort_perm(8)
        type(c_ptr) :: recv_buf_dev, sort_perm_dev, recv_buf_sorted_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64) :: expected(8)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: reorder_recv_buf..."

        ! Input buffer (1-indexed in Fortran)
        do i = 1, N
            recv_buf(i) = cmplx(real(i, real64), real(i, real64) * 0.5_real64, real64)
        end do

        ! Permutation using 1-based indices (kernel does sort_perm[i] - 1)
        ! sorted[0] = recv_buf[8-1], sorted[1] = recv_buf[7-1], etc.
        sort_perm = [8_c_long, 7_c_long, 6_c_long, 5_c_long, 4_c_long, 3_c_long, 2_c_long, 1_c_long]

        ! Expected output (sort_perm is 1-based, matches Fortran indexing directly)
        do i = 1, N
            expected(i) = recv_buf(sort_perm(i)) ! sort_perm is already 1-based
        end do

        recv_buf_sorted = cmplx(0.0_real64, 0.0_real64, real64)

        call hipCheck(hipMalloc(recv_buf_dev, int(N * 16, c_size_t)))
        call hipCheck(hipMalloc(sort_perm_dev, int(N * 8, c_size_t)))
        call hipCheck(hipMalloc(recv_buf_sorted_dev, int(N * 16, c_size_t)))

        call hipCheck(hipMemcpy(recv_buf_dev, c_loc(recv_buf), int(N * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(sort_perm_dev, c_loc(sort_perm), int(N * 8, c_size_t), hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(recv_buf_sorted_dev, c_loc(recv_buf_sorted), int(N * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_reorder_recv_buf_kernel(grid, block, 0, c_null_ptr, &
                                            recv_buf_dev, sort_perm_dev, recv_buf_sorted_dev, N)
        call hipCheck(hipDeviceSynchronize())

     call hipCheck(hipMemcpy(c_loc(recv_buf_sorted), recv_buf_sorted_dev, int(N * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(recv_buf_sorted(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(recv_buf_sorted(i)), aimag(recv_buf_sorted(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(recv_buf_dev))
        call hipCheck(hipFree(sort_perm_dev))
        call hipCheck(hipFree(recv_buf_sorted_dev))
    end subroutine test_reorder_recv_buf

    !--------------------------------------------------------------------------
    ! Test pack_send_buf: send_buf[i] = x_local[send_offsets[i]]
    ! Interface: launch_pack_send_buf_kernel(grid, block, shmem, stream,
    !            x_local, send_offsets, send_buf, total_send)
    !--------------------------------------------------------------------------
    subroutine test_pack_send_buf()
        integer(c_long), parameter :: local_size = 16_c_long
        integer(c_long), parameter :: send_size = 4_c_long
        complex(real64), target :: x_local(16), send_buf(4)
        integer(c_long), target :: send_offsets(4)
        type(c_ptr) :: x_local_dev, send_offsets_dev, send_buf_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64) :: expected(4)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: pack_send_buf..."

        ! Local vector
        do i = 1, local_size
            x_local(i) = cmplx(real(i * 10, real64), real(i, real64), real64)
        end do

        ! Send specific indices (1-based, kernel does offset - 1)
        send_offsets = [1_c_long, 6_c_long, 11_c_long, 16_c_long]

        ! Expected: gather from x_local at specified offsets (1-based matches Fortran)
        do i = 1, send_size
            expected(i) = x_local(send_offsets(i)) ! 1-based matches Fortran indexing
        end do

        call hipCheck(hipMalloc(x_local_dev, int(local_size * 16, c_size_t)))
        call hipCheck(hipMalloc(send_offsets_dev, int(send_size * 8, c_size_t)))
        call hipCheck(hipMalloc(send_buf_dev, int(send_size * 16, c_size_t)))

        call hipCheck(hipMemcpy(x_local_dev, c_loc(x_local), int(local_size * 16, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(send_offsets_dev, c_loc(send_offsets), int(send_size * 8, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_pack_send_buf_kernel(grid, block, 0, c_null_ptr, &
                                         x_local_dev, send_offsets_dev, send_buf_dev, send_size)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(send_buf), send_buf_dev, int(send_size * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, send_size
            if (abs(send_buf(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(send_buf(i)), aimag(send_buf(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(x_local_dev))
        call hipCheck(hipFree(send_offsets_dev))
        call hipCheck(hipFree(send_buf_dev))
    end subroutine test_pack_send_buf

end program test_spmv_kernels
