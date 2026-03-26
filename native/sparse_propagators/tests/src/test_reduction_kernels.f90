! test_reduction_kernels.f90
! Unit tests for HIP reduction operation kernels
!
! Tests:
!   1. reduce_max - find maximum value in array
!   2. dense_one_norms - compute column 1-norms of dense matrix
!   3. gershgorin_bound - compute Gershgorin disk radius (weighted)
!   4. gershgorin_bound_unit - compute Gershgorin disk radius (unit diagonal)
!   5. csr_column_one_norms - CSR column 1-norms (weighted)
!   6. csr_column_one_norms_unit - CSR column 1-norms (unit diagonal)

program test_reduction_kernels
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_types
    use hip_sparse_expm_kernels, only: &
        launch_csr_column_one_norms_kernel, launch_csr_column_one_norms_unit_kernel, launch_dense_one_norms_kernel, &
        launch_gershgorin_bound_kernel, launch_gershgorin_bound_unit_kernel, launch_reduce_max_kernel
    implicit none

    integer(int32), parameter :: BLOCKSIZE = 256
    real(real64), parameter :: tolerance = 1.0e-10_real64

    integer(int32) :: total_tests, passed_tests

    total_tests = 0
    passed_tests = 0

    write (*, *) "========================================"
    write (*, *) " Reduction Kernels Unit Tests"
    write (*, *) "========================================"

    call hipCheck(hipInit(0))

    call test_reduce_max()
    call test_dense_one_norms()
    call test_gershgorin_bound()
    call test_gershgorin_bound_unit()
    call test_csr_column_one_norms()
    call test_csr_column_one_norms_unit()

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
    ! Test reduce_max: find maximum value in array
    ! Note: The kernel produces per-block results in result[blockIdx.x].
    !       For multi-block runs, a second pass is needed. Here we test with
    !       a single block (N <= BLOCKSIZE) to verify the kernel correctness.
    !--------------------------------------------------------------------------
    subroutine test_reduce_max()
        ! Use size <= BLOCKSIZE to get result in single block
        integer(c_size_t), parameter :: N = 256_c_size_t
        real(real64), allocatable, target :: data_host(:)
        real(real64), target :: result_host
        type(c_ptr) :: data_dev, result_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed
        real(real64) :: expected_max

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: reduce_max..."

        allocate (data_host(N))

        expected_max = 0.0_real64
        do i = 1, N
            data_host(i) = real(i, real64) * 0.1_real64
            if (data_host(i) > expected_max) expected_max = data_host(i)
        end do
        ! Put max value somewhere in the middle
        data_host(128) = 999.0_real64
        expected_max = 999.0_real64

        result_host = 0.0_real64

        call hipCheck(hipMalloc(data_dev, int(N * 8, c_size_t)))
        call hipCheck(hipMalloc(result_dev, 8_c_size_t))
        call hipCheck(hipMemcpy(data_dev, c_loc(data_host), int(N * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(result_dev, c_loc(result_host), 8_c_size_t, hipMemcpyHostToDevice))

        ! Single block for single-pass reduction
        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_reduce_max_kernel(grid, block, 0, c_null_ptr, data_dev, result_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(result_host), result_dev, 8_c_size_t, hipMemcpyDeviceToHost))

        test_passed = abs(result_host - expected_max) < tolerance

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
            write (*, '(A,F12.6,A,F12.6)') "    Expected: ", expected_max, ", Got: ", result_host
        else
            write (*, '(A,F12.6,A,F12.6)') "  FAILED: Expected ", expected_max, ", got ", result_host
        end if

        call hipCheck(hipFree(data_dev))
        call hipCheck(hipFree(result_dev))
        deallocate (data_host)
    end subroutine test_reduce_max

    !--------------------------------------------------------------------------
    ! Test dense_one_norms: column 1-norms of dense matrix
    ! Interface: launch_dense_one_norms_kernel(grid, block, shmem, stream, result, X, N, l)
    ! N = rows, l = columns (l <= 5)
    ! Note: Output layout is result[i * BLOCKSIZE + blockIdx.x] for column i.
    !       With single block, results are at indices 0, 256, 512, ... (i * BLOCKSIZE).
    !       Need result array of size l * BLOCKSIZE for single block.
    !--------------------------------------------------------------------------
    subroutine test_dense_one_norms()
        integer(c_int), parameter :: n_rows = 64
        integer(c_int), parameter :: n_cols = 4 ! Must be <= 5 for this kernel
        integer(c_int), parameter :: result_size = n_cols * BLOCKSIZE
        complex(real64), allocatable, target :: matrix_host(:, :)
        real(real64), allocatable, target :: norms_host(:), expected_norms(:)
        type(c_ptr) :: matrix_dev, norms_dev
        type(dim3) :: grid, block
        integer(int32) :: i, j
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: dense_one_norms..."

        allocate (matrix_host(n_rows, n_cols))
        allocate (norms_host(result_size), expected_norms(n_cols))

        do j = 1, n_cols
            expected_norms(j) = 0.0_real64
            do i = 1, n_rows
                matrix_host(i, j) = cmplx(real(i + j, real64), -real(i, real64) * 0.1_real64, real64)
                expected_norms(j) = expected_norms(j) + abs(matrix_host(i, j))
            end do
        end do

        norms_host = 0.0_real64

        call hipCheck(hipMalloc(matrix_dev, int(n_rows * n_cols * 16, c_size_t)))
        call hipCheck(hipMalloc(norms_dev, int(result_size * 8, c_size_t)))
    call hipCheck(hipMemcpy(matrix_dev, c_loc(matrix_host), int(n_rows * n_cols * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(norms_dev, c_loc(norms_host), int(result_size * 8, c_size_t), hipMemcpyHostToDevice))

        ! Single block to get single-pass reduction
        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(1, 1, 1)

        call launch_dense_one_norms_kernel(grid, block, 0, c_null_ptr, &
                                           norms_dev, matrix_dev, n_rows, n_cols)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(norms_host), norms_dev, int(result_size * 8, c_size_t), hipMemcpyDeviceToHost))

        ! Results are at indices 0, BLOCKSIZE, 2*BLOCKSIZE, ...
        test_passed = .true.
        do j = 1, n_cols
            i = (j - 1) * BLOCKSIZE + 1 ! Fortran 1-indexed: 1, 257, 513, 769
            if (abs(norms_host(i) - expected_norms(j)) > tolerance * expected_norms(j)) then
                write (*, '(A,I0,A,F12.6,A,F12.6)') "  FAILED at column ", j, &
                    ": expected ", expected_norms(j), ", got ", norms_host(i)
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(matrix_dev))
        call hipCheck(hipFree(norms_dev))
        deallocate (matrix_host, norms_host, expected_norms)
    end subroutine test_dense_one_norms

    !--------------------------------------------------------------------------
    ! Test gershgorin_bound: weighted diagonal case
    ! Interface: launch_gershgorin_bound_kernel(grid, block, shmem, stream,
    !            row_starts, col_inds, values, row_bounds, local_rows, offset)
    ! Note: col_inds is int*, not long*
    !--------------------------------------------------------------------------
    subroutine test_gershgorin_bound()
        integer(c_int), parameter :: N = 4
        integer(c_int), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_int), target :: col_indexes(10) ! c_int, not c_long
        complex(real64), target :: values(10)
        real(real64), allocatable, target :: gershgorin_host(:)
        type(c_ptr) :: row_starts_dev, col_indexes_dev, values_dev, gershgorin_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed
        real(real64) :: expected(4)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: gershgorin_bound (weighted)..."

        ! CSR format (0-indexed)
        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
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

        expected = [3.0_real64, 5.0_real64, 6.0_real64, 6.0_real64]

        allocate (gershgorin_host(N))
        gershgorin_host = 0.0_real64

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 4, c_size_t))) ! 4 bytes per int
        call hipCheck(hipMalloc(values_dev, int(nnz * 16, c_size_t)))
        call hipCheck(hipMalloc(gershgorin_dev, int(N * 8, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 4, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(values_dev, c_loc(values), int(nnz * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(gershgorin_dev, c_loc(gershgorin_host), int(N * 8, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_gershgorin_bound_kernel(grid, block, 0, c_null_ptr, &
                                            row_starts_dev, col_indexes_dev, values_dev, gershgorin_dev, N, 0)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(gershgorin_host), gershgorin_dev, int(N * 8, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(gershgorin_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,F12.6,A,F12.6)') "  FAILED at row ", i - 1, &
                    ": expected ", expected(i), ", got ", gershgorin_host(i)
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
        call hipCheck(hipFree(gershgorin_dev))
        deallocate (gershgorin_host)
    end subroutine test_gershgorin_bound

    !--------------------------------------------------------------------------
    ! Test gershgorin_bound_unit: unit diagonal case
    !--------------------------------------------------------------------------
    subroutine test_gershgorin_bound_unit()
        integer(c_int), parameter :: N = 4
        integer(c_int), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_int), target :: col_indexes(10)
        real(real64), allocatable, target :: gershgorin_host(:)
        type(c_ptr) :: row_starts_dev, col_indexes_dev, gershgorin_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed
        real(real64) :: expected(4)

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: gershgorin_bound_unit..."

        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]

        ! Expected for unit: 1 + (nnz_row - 1) = nnz_row
        expected = [2.0_real64, 3.0_real64, 3.0_real64, 2.0_real64]

        allocate (gershgorin_host(N))
        gershgorin_host = 0.0_real64

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 4, c_size_t)))
        call hipCheck(hipMalloc(gershgorin_dev, int(N * 8, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 4, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(gershgorin_dev, c_loc(gershgorin_host), int(N * 8, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((N + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_gershgorin_bound_unit_kernel(grid, block, 0, c_null_ptr, &
                                                 row_starts_dev, col_indexes_dev, gershgorin_dev, N, 0)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(gershgorin_host), gershgorin_dev, int(N * 8, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(gershgorin_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,F12.6,A,F12.6)') "  FAILED at row ", i - 1, &
                    ": expected ", expected(i), ", got ", gershgorin_host(i)
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(gershgorin_dev))
        deallocate (gershgorin_host)
    end subroutine test_gershgorin_bound_unit

    !--------------------------------------------------------------------------
    ! Test csr_column_one_norms: column 1-norms of CSR matrix
    ! Interface uses c_int for col_inds, num_rows, num_cols
    !--------------------------------------------------------------------------
    subroutine test_csr_column_one_norms()
        integer(c_int), parameter :: N = 4
        integer(c_int), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_int), target :: col_indexes(10)
        complex(real64), target :: values(10)
        real(real64), allocatable, target :: norms_host(:), expected(:)
        type(c_ptr) :: row_starts_dev, col_indexes_dev, values_dev, norms_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: csr_column_one_norms..."

        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
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

        allocate (norms_host(N), expected(N))
        expected = [3.0_real64, 5.0_real64, 6.0_real64, 6.0_real64]
        norms_host = 0.0_real64

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 4, c_size_t)))
        call hipCheck(hipMalloc(values_dev, int(nnz * 16, c_size_t)))
        call hipCheck(hipMalloc(norms_dev, int(N * 8, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 4, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(values_dev, c_loc(values), int(nnz * 16, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(norms_dev, c_loc(norms_host), int(N * 8, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((nnz + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_csr_column_one_norms_kernel(grid, block, 0, c_null_ptr, &
                                                row_starts_dev, col_indexes_dev, values_dev, norms_dev, N, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(norms_host), norms_dev, int(N * 8, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(norms_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,F12.6,A,F12.6)') "  FAILED at column ", i - 1, &
                    ": expected ", expected(i), ", got ", norms_host(i)
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
        call hipCheck(hipFree(norms_dev))
        deallocate (norms_host, expected)
    end subroutine test_csr_column_one_norms

    !--------------------------------------------------------------------------
    ! Test csr_column_one_norms_unit: column 1-norms with unit weights
    !--------------------------------------------------------------------------
    subroutine test_csr_column_one_norms_unit()
        integer(c_int), parameter :: N = 4
        integer(c_int), parameter :: nnz = 10
        integer(c_long), target :: row_starts(5)
        integer(c_int), target :: col_indexes(10)
        real(real64), allocatable, target :: norms_host(:), expected(:)
        type(c_ptr) :: row_starts_dev, col_indexes_dev, norms_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: csr_column_one_norms_unit..."

        row_starts = [0_c_long, 2_c_long, 5_c_long, 8_c_long, 10_c_long]
        col_indexes = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]

        ! Count of entries per column
        allocate (norms_host(N), expected(N))
        expected = [2.0_real64, 3.0_real64, 3.0_real64, 2.0_real64]
        norms_host = 0.0_real64

        call hipCheck(hipMalloc(row_starts_dev, int(5 * 8, c_size_t)))
        call hipCheck(hipMalloc(col_indexes_dev, int(nnz * 4, c_size_t)))
        call hipCheck(hipMalloc(norms_dev, int(N * 8, c_size_t)))

        call hipCheck(hipMemcpy(row_starts_dev, c_loc(row_starts), int(5 * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(col_indexes_dev, c_loc(col_indexes), int(nnz * 4, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(norms_dev, c_loc(norms_host), int(N * 8, c_size_t), hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((nnz + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        call launch_csr_column_one_norms_unit_kernel(grid, block, 0, c_null_ptr, &
                                                     row_starts_dev, col_indexes_dev, norms_dev, N, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(norms_host), norms_dev, int(N * 8, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(norms_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A,F12.6,A,F12.6)') "  FAILED at column ", i - 1, &
                    ": expected ", expected(i), ", got ", norms_host(i)
                test_passed = .false.
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(row_starts_dev))
        call hipCheck(hipFree(col_indexes_dev))
        call hipCheck(hipFree(norms_dev))
        deallocate (norms_host, expected)
    end subroutine test_csr_column_one_norms_unit

end program test_reduction_kernels
