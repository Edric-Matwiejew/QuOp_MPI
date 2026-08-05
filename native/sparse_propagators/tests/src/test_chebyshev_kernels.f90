! test_chebyshev_kernels.f90
! Unit tests for HIP Chebyshev recurrence kernels
!
! Tests:
!   1. chebyshev_local_weighted - local phase with weighted edges
!   2. chebyshev_local_unit - local phase with unit edge weights
!   3. chebyshev_remote_weighted - remote phase with weighted edges
!   4. chebyshev_remote_unit - remote phase with unit edge weights
!   5. chebyshev_accumulate - accumulate scaled contribution

program test_chebyshev_kernels
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_types
    use hip_sparse_expm_kernels, only: &
        launch_chebyshev_accumulate_kernel, launch_chebyshev_local_unit_kernel, &
        launch_chebyshev_local_weighted_kernel, launch_chebyshev_remote_unit_kernel, &
        launch_chebyshev_remote_weighted_kernel
    use test_hash_utils, only: build_hash_table
    implicit none

    integer(int32), parameter :: BLOCKSIZE = 256
    real(real64), parameter :: tolerance = 1.0e-10_real64

    integer(int32) :: total_tests, passed_tests

    total_tests = 0
    passed_tests = 0

    write (*, *) "========================================"
    write (*, *) " Chebyshev Kernels Unit Tests"
    write (*, *) "========================================"

    call hipCheck(hipInit(0))

    call test_chebyshev_local_weighted()
    call test_chebyshev_local_unit()
    call test_chebyshev_remote_weighted()
    call test_chebyshev_remote_unit()
    call test_chebyshev_accumulate()

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
    ! Test chebyshev_local_weighted: Aw_k = A_local * w_k_local
    ! Interface: launch_chebyshev_local_weighted_kernel(grid, block, shmem, stream,
    !            inv_M, row_starts, col_indexes, values, w_k_local, Aw_k,
    !            lb, ub, local_rows)
    ! Note: inv_M is c_double, used to scale the result
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_local_weighted()
        integer(c_long), parameter :: N = 4
        integer(c_long), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_long), target :: col_indexes(10)
        complex(real64), target :: values(10), w_k(4), Aw_k_host(4)
        real(c_double) :: inv_M
        type(c_ptr) :: row_starts_dev, col_indexes_dev, values_dev, w_k_dev, Aw_k_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64), allocatable, target :: expected(:)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: chebyshev_local_weighted..."

        ! 4x4 tridiagonal matrix in CSR (0-indexed)
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

        ! Input vector w_k = [1, 2, 3, 4]
        w_k = [cmplx(1.0_real64, 0.0_real64, real64), &
               cmplx(2.0_real64, 0.0_real64, real64), &
               cmplx(3.0_real64, 0.0_real64, real64), &
               cmplx(4.0_real64, 0.0_real64, real64)]

        ! inv_M scaling factor (not used in local phase, but still a parameter)
        inv_M = 0.5_c_double

        ! Expected: Aw_k = A * w_k (no inv_M scaling in local phase)
        ! Raw A*w: [4, 10, 18, 23]
        allocate (expected(4))
        expected = [cmplx(4.0_real64, 0.0_real64, real64), &
                    cmplx(10.0_real64, 0.0_real64, real64), &
                    cmplx(18.0_real64, 0.0_real64, real64), &
                    cmplx(23.0_real64, 0.0_real64, real64)]

        Aw_k_host = cmplx(0.0_real64, 0.0_real64, real64)

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 8, c_size_t)))
        call hipCheck(hipMalloc(values_dev, int(nnz * 16, c_size_t)))
        call hipCheck(hipMalloc(w_k_dev, int(N * 16, c_size_t)))
        call hipCheck(hipMalloc(Aw_k_dev, int(N * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(values_dev, c_loc(values), int(nnz * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(w_k_dev, c_loc(w_k), int(N * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(Aw_k_dev, c_loc(Aw_k_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_chebyshev_local_weighted_kernel(grid, block, 0, c_null_ptr, &
                                                inv_M, row_starts_dev, col_indexes_dev, values_dev, w_k_dev, Aw_k_dev, &
                                                    0_c_long, N - 1, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(Aw_k_host), Aw_k_dev, int(N * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(Aw_k_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(Aw_k_host(i)), aimag(Aw_k_host(i)), ")"
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
        call hipCheck(hipFree(w_k_dev))
        call hipCheck(hipFree(Aw_k_dev))
        deallocate (expected)
    end subroutine test_chebyshev_local_weighted

    !--------------------------------------------------------------------------
    ! Test chebyshev_local_unit: Aw_k = A_local * w_k_local with unit weights
    ! Interface: launch_chebyshev_local_unit_kernel(grid, block, shmem, stream,
    !            inv_M, row_starts, col_indexes, w_k_local, Aw_k,
    !            lb, ub, local_rows)
    ! Note: no 'values' parameter
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_local_unit()
        integer(c_long), parameter :: N = 4
        integer(c_long), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_long), target :: col_indexes(10)
        complex(real64), target :: w_k(4), Aw_k_host(4)
        real(c_double) :: inv_M
        type(c_ptr) :: row_starts_dev, col_indexes_dev, w_k_dev, Aw_k_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64), allocatable, target :: expected(:)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: chebyshev_local_unit..."

        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0_c_long, 1_c_long, 0_c_long, 1_c_long, 2_c_long, &
                       1_c_long, 2_c_long, 3_c_long, 2_c_long, 3_c_long]

        w_k = [cmplx(1.0_real64, 0.0_real64, real64), &
               cmplx(2.0_real64, 0.0_real64, real64), &
               cmplx(3.0_real64, 0.0_real64, real64), &
               cmplx(4.0_real64, 0.0_real64, real64)]

        inv_M = 0.5_c_double

        ! Expected with unit weights: A_unit * w_k (no inv_M in local phase)
        ! A_unit * w: [3, 6, 9, 7]
        allocate (expected(4))
        expected = [cmplx(3.0_real64, 0.0_real64, real64), &
                    cmplx(6.0_real64, 0.0_real64, real64), &
                    cmplx(9.0_real64, 0.0_real64, real64), &
                    cmplx(7.0_real64, 0.0_real64, real64)]

        Aw_k_host = cmplx(0.0_real64, 0.0_real64, real64)

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 8, c_size_t)))
        call hipCheck(hipMalloc(w_k_dev, int(N * 16, c_size_t)))
        call hipCheck(hipMalloc(Aw_k_dev, int(N * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(w_k_dev, c_loc(w_k), int(N * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(Aw_k_dev, c_loc(Aw_k_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_chebyshev_local_unit_kernel(grid, block, 0, c_null_ptr, &
                                                inv_M, row_starts_dev, col_indexes_dev, w_k_dev, Aw_k_dev, &
                                                0_c_long, N - 1, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(Aw_k_host), Aw_k_dev, int(N * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(Aw_k_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(Aw_k_host(i)), aimag(Aw_k_host(i)), ")"
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(w_k_dev))
        call hipCheck(hipFree(Aw_k_dev))
        deallocate (expected)
    end subroutine test_chebyshev_local_unit

    !--------------------------------------------------------------------------
    ! Test chebyshev_remote_weighted: w_kp1 = 2*inv_M*(Aw_k + A_remote*recv_buf) - w_km1
    ! Interface: launch_chebyshev_remote_weighted_kernel(grid, block, shmem, stream,
    !            inv_M, row_starts, col_indexes, values, recv_buf_sorted,
    !            hash_keys, hash_vals, hash_size, Aw_k, w_km1, w_kp1,
    !            lb, ub, local_rows)
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_remote_weighted()
        integer(c_long), parameter :: local_rows = 2_c_long
        integer(c_long), parameter :: lb = 0_c_long, ub = 1_c_long
        integer(c_long), parameter :: nnz_remote = 4
        integer(c_long), parameter :: hash_table_size = 16_c_long
        integer(c_long), target :: row_starts(3)
        integer(c_long), target :: col_indexes(4)
        complex(real64), target :: values(4), recv_buf(2), Aw_k(2), w_km1(2), w_kp1_host(2)
        integer(c_long), target :: hash_keys(16), hash_vals(16)
        integer(c_long) :: remote_cols(2)
        real(c_double) :: inv_M
        type(c_ptr) :: row_starts_dev, col_indexes_dev, values_dev
        type(c_ptr) :: recv_buf_dev, hash_keys_dev, hash_vals_dev
        type(c_ptr) :: Aw_k_dev, w_km1_dev, w_kp1_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64) :: expected(2)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: chebyshev_remote_weighted..."

        ! Remote matrix: 2 rows, columns [2,3] (outside [0,1])
        row_starts = [0_c_long, 2_c_long, 4_c_long]
        col_indexes = [2_c_long, 3_c_long, 2_c_long, 3_c_long]
        values = [cmplx(1.0_real64, 0.0_real64, real64), &
                  cmplx(2.0_real64, 0.0_real64, real64), &
                  cmplx(3.0_real64, 0.0_real64, real64), &
                  cmplx(4.0_real64, 0.0_real64, real64)]

        ! recv_buf: values for cols 2,3 at positions 0,1
        recv_buf = [cmplx(1.0_real64, 0.0_real64, real64), &
                    cmplx(2.0_real64, 0.0_real64, real64)]

        ! Build hash table using proper hash function
        remote_cols = [2_c_long, 3_c_long]
        call build_hash_table(remote_cols, 2, hash_keys, hash_vals, hash_table_size)

        ! Aw_k from local phase (simulated - in real use would come from local kernel)
        Aw_k = [cmplx(10.0_real64, 0.0_real64, real64), &
                cmplx(20.0_real64, 0.0_real64, real64)]

        ! Previous iteration w_{k-1}
        w_km1 = [cmplx(1.0_real64, 0.0_real64, real64), &
                 cmplx(2.0_real64, 0.0_real64, real64)]

        inv_M = 0.5_c_double ! M = 2, so inv_M = 1/M = 0.5

        ! A_remote * recv_buf:
        ! Row 0: 1*1 + 2*2 = 5
        ! Row 1: 3*1 + 4*2 = 11
        !
        ! w_kp1 = 2 * inv_M * (Aw_k + A_remote*recv_buf) - w_km1
        ! Row 0: 2*0.5*(10 + 5) - 1 = 1*15 - 1 = 14
        ! Row 1: 2*0.5*(20 + 11) - 2 = 1*31 - 2 = 29
        expected = [cmplx(14.0_real64, 0.0_real64, real64), &
                    cmplx(29.0_real64, 0.0_real64, real64)]

        w_kp1_host = cmplx(0.0_real64, 0.0_real64, real64)

        call hipCheck(hipMalloc(row_starts_dev, int(3 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz_remote * 8, c_size_t)))
        call hipCheck(hipMalloc(values_dev, int(nnz_remote * 16, c_size_t)))
        call hipCheck(hipMalloc(recv_buf_dev, int(2 * 16, c_size_t)))
        call hipCheck(hipMalloc(hash_keys_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(hash_vals_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(Aw_k_dev, int(local_rows * 16, c_size_t)))
        call hipCheck(hipMalloc(w_km1_dev, int(local_rows * 16, c_size_t)))
        call hipCheck(hipMalloc(w_kp1_dev, int(local_rows * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(3 * 8, c_size_t), hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz_remote * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(values_dev, c_loc(values), int(nnz_remote * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(recv_buf_dev, c_loc(recv_buf), int(2 * 16, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_keys_dev, c_loc(hash_keys), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_vals_dev, c_loc(hash_vals), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(Aw_k_dev, c_loc(Aw_k), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(w_km1_dev, c_loc(w_km1), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(w_kp1_dev, c_loc(w_kp1_host), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_chebyshev_remote_weighted_kernel(grid, block, 0, c_null_ptr, &
                                                     inv_M, row_starts_dev, col_indexes_dev, values_dev, recv_buf_dev, &
                                        hash_keys_dev, hash_vals_dev, hash_table_size, Aw_k_dev, w_km1_dev, w_kp1_dev, &
                                                     lb, ub, local_rows)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(w_kp1_host), w_kp1_dev, int(local_rows * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, local_rows
            if (abs(w_kp1_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(w_kp1_host(i)), aimag(w_kp1_host(i)), ")"
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
        call hipCheck(hipFree(Aw_k_dev))
        call hipCheck(hipFree(w_km1_dev))
        call hipCheck(hipFree(w_kp1_dev))
    end subroutine test_chebyshev_remote_weighted

    !--------------------------------------------------------------------------
    ! Test chebyshev_remote_unit: w_kp1 = 2*inv_M*(Aw_k + A_remote*recv_buf) - w_km1
    ! Interface: launch_chebyshev_remote_unit_kernel(grid, block, shmem, stream,
    !            inv_M, row_starts, col_indexes, recv_buf_sorted,
    !            hash_keys, hash_vals, hash_size, Aw_k, w_km1, w_kp1,
    !            lb, ub, local_rows)
    ! Note: no 'values' parameter
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_remote_unit()
        integer(c_long), parameter :: local_rows = 2_c_long
        integer(c_long), parameter :: lb = 0_c_long, ub = 1_c_long
        integer(c_long), parameter :: nnz_remote = 4
        integer(c_long), parameter :: hash_table_size = 16_c_long
        integer(c_long), target :: row_starts(3)
        integer(c_long), target :: col_indexes(4)
        complex(real64), target :: recv_buf(2), Aw_k(2), w_km1(2), w_kp1_host(2)
        integer(c_long), target :: hash_keys(16), hash_vals(16)
        integer(c_long) :: remote_cols(2)
        real(c_double) :: inv_M
        type(c_ptr) :: row_starts_dev, col_indexes_dev
        type(c_ptr) :: recv_buf_dev, hash_keys_dev, hash_vals_dev
        type(c_ptr) :: Aw_k_dev, w_km1_dev, w_kp1_dev
        type(dim3) :: grid, block
        integer(c_long) :: i
        logical :: test_passed
        complex(real64) :: expected(2)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: chebyshev_remote_unit..."

        row_starts = [0_c_long, 2_c_long, 4_c_long]
        col_indexes = [2_c_long, 3_c_long, 2_c_long, 3_c_long]

        ! recv_buf: values for cols 2,3 at positions 0,1
        recv_buf = [cmplx(1.0_real64, 0.0_real64, real64), &
                    cmplx(2.0_real64, 0.0_real64, real64)]

        ! Build hash table using proper hash function
        remote_cols = [2_c_long, 3_c_long]
        call build_hash_table(remote_cols, 2, hash_keys, hash_vals, hash_table_size)

        Aw_k = [cmplx(10.0_real64, 0.0_real64, real64), &
                cmplx(20.0_real64, 0.0_real64, real64)]

        w_km1 = [cmplx(1.0_real64, 0.0_real64, real64), &
                 cmplx(2.0_real64, 0.0_real64, real64)]

        inv_M = 0.5_c_double

        ! With unit weights: A_remote * recv_buf = [1*1 + 1*2, 1*1 + 1*2] = [3, 3]
        ! w_kp1 = 2*0.5*(Aw_k + remote) - w_km1
        ! Row 0: 1*(10 + 3) - 1 = 12
        ! Row 1: 1*(20 + 3) - 2 = 21
        expected = [cmplx(12.0_real64, 0.0_real64, real64), &
                    cmplx(21.0_real64, 0.0_real64, real64)]

        w_kp1_host = cmplx(0.0_real64, 0.0_real64, real64)

        call hipCheck(hipMalloc(row_starts_dev, int(3 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz_remote * 8, c_size_t)))
        call hipCheck(hipMalloc(recv_buf_dev, int(2 * 16, c_size_t)))
        call hipCheck(hipMalloc(hash_keys_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(hash_vals_dev, int(hash_table_size * 8, c_size_t)))
        call hipCheck(hipMalloc(Aw_k_dev, int(local_rows * 16, c_size_t)))
        call hipCheck(hipMalloc(w_km1_dev, int(local_rows * 16, c_size_t)))
        call hipCheck(hipMalloc(w_kp1_dev, int(local_rows * 16, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(3 * 8, c_size_t), hipMemcpyHostToDevice))
     call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz_remote * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(recv_buf_dev, c_loc(recv_buf), int(2 * 16, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_keys_dev, c_loc(hash_keys), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(hash_vals_dev, c_loc(hash_vals), int(hash_table_size * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(Aw_k_dev, c_loc(Aw_k), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(w_km1_dev, c_loc(w_km1), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(w_kp1_dev, c_loc(w_kp1_host), int(local_rows * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_chebyshev_remote_unit_kernel(grid, block, 0, c_null_ptr, &
                                                 inv_M, row_starts_dev, col_indexes_dev, recv_buf_dev, &
                                        hash_keys_dev, hash_vals_dev, hash_table_size, Aw_k_dev, w_km1_dev, w_kp1_dev, &
                                                 lb, ub, local_rows)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(w_kp1_host), w_kp1_dev, int(local_rows * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, local_rows
            if (abs(w_kp1_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(w_kp1_host(i)), aimag(w_kp1_host(i)), ")"
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
        call hipCheck(hipFree(Aw_k_dev))
        call hipCheck(hipFree(w_km1_dev))
        call hipCheck(hipFree(w_kp1_dev))
    end subroutine test_chebyshev_remote_unit

    !--------------------------------------------------------------------------
    ! Test chebyshev_accumulate: C = C + coeff * w
    ! Interface: launch_chebyshev_accumulate_kernel(grid, block, shmem, stream,
    !            coeff, w, C, N)
    ! coeff is complex(c_double_complex)
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_accumulate()
        integer(c_size_t), parameter :: N = 256_c_size_t
        complex(real64), allocatable, target :: w_host(:), C_host(:), expected(:)
        complex(c_double_complex) :: coeff
        type(c_ptr) :: w_dev, C_dev
        type(dim3) :: grid, block
        integer(c_size_t) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: chebyshev_accumulate..."

        allocate (w_host(N), C_host(N), expected(N))

        ! Initialize
        do i = 1, N
            w_host(i) = cmplx(real(i, real64), real(i, real64) * 0.5_real64, real64)
            C_host(i) = cmplx(real(i, real64) * 10.0_real64, 0.0_real64, real64)
        end do

        coeff = cmplx(2.0_real64, 1.0_real64, c_double_complex)

        ! Expected: C_new = C_old + coeff * w
        ! coeff * w = (2+i) * (i + 0.5i*i) = (2+i) * (i*(1+0.5i))
        ! For i=1: w = (1, 0.5), coeff*w = (2+i)*(1+0.5i) = 2+i+i+0.5i^2 = 2-0.5+2i = 1.5+2i
        ! C_new = 10 + 1.5 + 2i = 11.5 + 2i
        do i = 1, N
            expected(i) = C_host(i) + coeff * w_host(i)
        end do

        call hipCheck(hipMalloc(w_dev, int(N * 16, c_size_t)))
        call hipCheck(hipMalloc(C_dev, int(N * 16, c_size_t)))

        call hipCheck(hipMemcpy(w_dev, c_loc(w_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(C_dev, c_loc(C_host), int(N * 16, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_chebyshev_accumulate_kernel(grid, block, 0, c_null_ptr, &
                                                coeff, w_dev, C_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(C_host), C_dev, int(N * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(C_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at ", i - 1, &
                    ": expected (", real(expected(i)), aimag(expected(i)), &
                    "), got (", real(C_host(i)), aimag(C_host(i)), ")"
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(w_dev))
        call hipCheck(hipFree(C_dev))
        deallocate (w_host, C_host, expected)
    end subroutine test_chebyshev_accumulate

end program test_chebyshev_kernels
