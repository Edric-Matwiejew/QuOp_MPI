! test_vector_kernels.f90
! Unit tests for HIP vector operation kernels
!
! Tests:
!   1. vector_infinity_norm - compute max|v[i]|
!   2. inplace_vec_sum - X = X + Y
!   3. b_scale - X = X / (s * j)
!   4. complex_axpy - y = alpha * x + y
!   5. complex_scale - x = alpha * x
!   6. real_scale - x = alpha * x
!   7. vec_copy - y = x

program test_vector_kernels
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_types
    use hip_sparse_expm_kernels, only: &
        launch_b_scale_kernel, launch_complex_axpy_kernel, launch_complex_scale_kernel, &
        launch_inplace_vec_sum_kernel, launch_real_scale_kernel, launch_vec_copy_kernel, &
        launch_vector_infinity_norm_kernel
    implicit none

    integer(int32), parameter :: BLOCKSIZE = 256
    real(real64), parameter :: tolerance = 1.0e-12_real64

    integer(int32) :: total_tests, passed_tests
    integer(int32) :: ierr

    total_tests = 0
    passed_tests = 0

    write (*, *) "========================================"
    write (*, *) " Vector Kernels Unit Tests"
    write (*, *) "========================================"

    ! Initialize HIP
    call hipCheck(hipInit(0))

    call test_vec_copy()
    call test_inplace_vec_sum()
    call test_real_scale()
    call test_complex_scale()
    call test_complex_axpy()
    call test_b_scale()
    call test_vector_infinity_norm()

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
    ! Test vec_copy: y = x
    !--------------------------------------------------------------------------
    subroutine test_vec_copy()
        integer(c_size_t), parameter :: N = 1024
        complex(real64), allocatable, target :: x_host(:), y_host(:)
        type(c_ptr) :: x_dev, y_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: vec_copy (y = x)..."

        allocate (x_host(N), y_host(N))

        ! Initialize x with known values, y with zeros
        do i = 1, N
            x_host(i) = cmplx(real(i, real64), -real(i, real64), real64)
            y_host(i) = cmplx(0.0_real64, 0.0_real64, real64)
        end do

        ! Allocate device memory
        call hipCheck(hipMalloc(x_dev, N * 16)) ! 16 bytes per complex
        call hipCheck(hipMalloc(y_dev, N * 16))

        ! Copy to device
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), N * 16, hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), N * 16, hipMemcpyHostToDevice))

        ! Launch kernel
        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1)

        call launch_vec_copy_kernel(grid, block, 0, c_null_ptr, x_dev, y_dev, N)
        call hipCheck(hipDeviceSynchronize())

        ! Copy result back
        call hipCheck(hipMemcpy(c_loc(y_host), y_dev, N * 16, hipMemcpyDeviceToHost))

        ! Verify
        test_passed = .true.
        do i = 1, N
            if (abs(y_host(i) - x_host(i)) > tolerance) then
                write (*, '(A,I0,A,2F12.6,A,2F12.6)') "  FAILED at index ", i, &
                    ": got (", real(y_host(i)), aimag(y_host(i)), &
                    "), expected (", real(x_host(i)), aimag(x_host(i)), ")"
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        ! Cleanup
        call hipCheck(hipFree(x_dev))
        call hipCheck(hipFree(y_dev))
        deallocate (x_host, y_host)
    end subroutine test_vec_copy

    !--------------------------------------------------------------------------
    ! Test inplace_vec_sum: X = X + Y
    !--------------------------------------------------------------------------
    subroutine test_inplace_vec_sum()
        integer(c_size_t), parameter :: N = 1024
        complex(real64), allocatable, target :: x_host(:), y_host(:), expected(:)
        type(c_ptr) :: x_dev, y_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: inplace_vec_sum (X = X + Y)..."

        allocate (x_host(N), y_host(N), expected(N))

        do i = 1, N
            x_host(i) = cmplx(real(i, real64), real(i, real64) * 0.5_real64, real64)
            y_host(i) = cmplx(real(N - i + 1, real64), -real(i, real64), real64)
            expected(i) = x_host(i) + y_host(i)
        end do

        call hipCheck(hipMalloc(x_dev, N * 16))
        call hipCheck(hipMalloc(y_dev, N * 16))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), N * 16, hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), N * 16, hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1)

        call launch_inplace_vec_sum_kernel(grid, block, 0, c_null_ptr, x_dev, y_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(x_host), x_dev, N * 16, hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(x_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A)') "  FAILED at index ", i
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(x_dev))
        call hipCheck(hipFree(y_dev))
        deallocate (x_host, y_host, expected)
    end subroutine test_inplace_vec_sum

    !--------------------------------------------------------------------------
    ! Test real_scale: x = alpha * x
    !--------------------------------------------------------------------------
    subroutine test_real_scale()
        integer(c_size_t), parameter :: N = 1024
        real(real64), parameter :: alpha = 2.5_real64
        complex(real64), allocatable, target :: x_host(:), expected(:)
        type(c_ptr) :: x_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: real_scale (x = alpha * x)..."

        allocate (x_host(N), expected(N))

        do i = 1, N
            x_host(i) = cmplx(real(i, real64), -real(i, real64) * 0.3_real64, real64)
            expected(i) = alpha * x_host(i)
        end do

        call hipCheck(hipMalloc(x_dev, N * 16))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), N * 16, hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1)

        call launch_real_scale_kernel(grid, block, 0, c_null_ptr, alpha, x_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(x_host), x_dev, N * 16, hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(x_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A)') "  FAILED at index ", i
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(x_dev))
        deallocate (x_host, expected)
    end subroutine test_real_scale

    !--------------------------------------------------------------------------
    ! Test complex_scale: x = alpha * x
    !--------------------------------------------------------------------------
    subroutine test_complex_scale()
        integer(c_size_t), parameter :: N = 1024
        complex(real64), parameter :: alpha = (2.0_real64, -1.5_real64)
        complex(real64), allocatable, target :: x_host(:), expected(:)
        type(c_ptr) :: x_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: complex_scale (x = alpha * x)..."

        allocate (x_host(N), expected(N))

        do i = 1, N
            x_host(i) = cmplx(real(i, real64), real(i, real64) * 0.2_real64, real64)
            expected(i) = alpha * x_host(i)
        end do

        call hipCheck(hipMalloc(x_dev, N * 16))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), N * 16, hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1)

        call launch_complex_scale_kernel(grid, block, 0, c_null_ptr, alpha, x_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(x_host), x_dev, N * 16, hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(x_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A)') "  FAILED at index ", i
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(x_dev))
        deallocate (x_host, expected)
    end subroutine test_complex_scale

    !--------------------------------------------------------------------------
    ! Test complex_axpy: y = alpha * x + y
    !--------------------------------------------------------------------------
    subroutine test_complex_axpy()
        integer(c_size_t), parameter :: N = 1024
        complex(real64), parameter :: alpha = (1.5_real64, 0.5_real64)
        complex(real64), allocatable, target :: x_host(:), y_host(:), expected(:)
        type(c_ptr) :: x_dev, y_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: complex_axpy (y = alpha * x + y)..."

        allocate (x_host(N), y_host(N), expected(N))

        do i = 1, N
            x_host(i) = cmplx(real(i, real64), -real(i, real64), real64)
            y_host(i) = cmplx(real(N - i, real64), real(i, real64) * 0.1_real64, real64)
            expected(i) = alpha * x_host(i) + y_host(i)
        end do

        call hipCheck(hipMalloc(x_dev, N * 16))
        call hipCheck(hipMalloc(y_dev, N * 16))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), N * 16, hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(y_dev, c_loc(y_host), N * 16, hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1)

        call launch_complex_axpy_kernel(grid, block, 0, c_null_ptr, alpha, x_dev, y_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(y_host), y_dev, N * 16, hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(y_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A)') "  FAILED at index ", i
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(x_dev))
        call hipCheck(hipFree(y_dev))
        deallocate (x_host, y_host, expected)
    end subroutine test_complex_axpy

    !--------------------------------------------------------------------------
    ! Test b_scale: X = X / (s * j)
    !--------------------------------------------------------------------------
    subroutine test_b_scale()
        integer(c_size_t), parameter :: N = 1024
        integer(c_int), parameter :: s = 3, j = 4
        complex(real64), allocatable, target :: x_host(:), expected(:)
        type(c_ptr) :: x_dev
        type(dim3) :: grid, block
        integer(int32) :: i
        logical :: test_passed
        real(real64) :: scale_factor

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: b_scale (X = X / (s * j))..."

        allocate (x_host(N), expected(N))

        scale_factor = 1.0_real64 / (real(s, real64) * real(j, real64))
        do i = 1, N
            x_host(i) = cmplx(real(i, real64) * 12.0_real64, -real(i, real64) * 6.0_real64, real64)
            expected(i) = x_host(i) * scale_factor
        end do

        call hipCheck(hipMalloc(x_dev, N * 16))
        call hipCheck(hipMemcpy(x_dev, c_loc(x_host), N * 16, hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1)

        call launch_b_scale_kernel(grid, block, 0, c_null_ptr, x_dev, s, j, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(x_host), x_dev, N * 16, hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, N
            if (abs(x_host(i) - expected(i)) > tolerance) then
                write (*, '(A,I0,A)') "  FAILED at index ", i
                test_passed = .false.
                exit
            end if
        end do

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
        end if

        call hipCheck(hipFree(x_dev))
        deallocate (x_host, expected)
    end subroutine test_b_scale

    !--------------------------------------------------------------------------
    ! Test vector_infinity_norm: compute max|v[i]|
    !--------------------------------------------------------------------------
    subroutine test_vector_infinity_norm()
        integer(c_size_t), parameter :: N = 1024
        integer(int32) :: num_blocks
        complex(real64), allocatable, target :: v_host(:)
        real(real64), allocatable, target :: infnorm_host(:)
        type(c_ptr) :: v_dev, infnorm_dev
        type(dim3) :: grid, block
        integer(int32) :: i, max_idx
        logical :: test_passed
        real(real64) :: expected_norm, computed_norm, abs_val

        total_tests = total_tests + 1
        write (*, *) ""
        write (*, *) "Test: vector_infinity_norm (max|v[i]|)..."

        num_blocks = (N + BLOCKSIZE - 1) / BLOCKSIZE
        allocate (v_host(N), infnorm_host(num_blocks))

        ! Create vector with known maximum
        max_idx = 512
        expected_norm = 0.0_real64
        do i = 1, N
            v_host(i) = cmplx(real(i, real64) * 0.01_real64, -real(i, real64) * 0.005_real64, real64)
            abs_val = abs(v_host(i))
            if (abs_val > expected_norm) expected_norm = abs_val
        end do
        ! Make one element clearly the largest
        v_host(max_idx) = cmplx(100.0_real64, 100.0_real64, real64)
        expected_norm = abs(v_host(max_idx))

        call hipCheck(hipMalloc(v_dev, N * 16))
        call hipCheck(hipMalloc(infnorm_dev, int(num_blocks, c_size_t) * 8)) ! 8 bytes per double
        call hipCheck(hipMemcpy(v_dev, c_loc(v_host), N * 16, hipMemcpyHostToDevice))

        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(num_blocks, 1, 1)

        call launch_vector_infinity_norm_kernel(grid, block, 0, c_null_ptr, &
                                                infnorm_dev, v_dev, N)
        call hipCheck(hipDeviceSynchronize())

        call hipCheck(hipMemcpy(c_loc(infnorm_host), infnorm_dev, &
                                int(num_blocks, c_size_t) * 8, hipMemcpyDeviceToHost))

        ! Final reduction on host
        computed_norm = 0.0_real64
        do i = 1, num_blocks
            if (infnorm_host(i) > computed_norm) computed_norm = infnorm_host(i)
        end do

        test_passed = (abs(computed_norm - expected_norm) < tolerance)

        if (test_passed) then
            passed_tests = passed_tests + 1
            write (*, *) "  PASSED"
            write (*, '(A,F12.6,A,F12.6)') "    Expected: ", expected_norm, ", Got: ", computed_norm
        else
            write (*, '(A,F12.6,A,F12.6)') "  FAILED: Expected ", expected_norm, &
                ", got ", computed_norm
        end if

        call hipCheck(hipFree(v_dev))
        call hipCheck(hipFree(infnorm_dev))
        deallocate (v_host, infnorm_host)
    end subroutine test_vector_infinity_norm

end program test_vector_kernels
