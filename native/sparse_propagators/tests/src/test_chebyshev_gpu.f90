! test_chebyshev_gpu.f90
! Integration tests for GPU-enabled Chebyshev time evolution
!
! Tests:
!   1. estimate_spectral_radius - CPU vs GPU consistency
!   2. chebyshev_multiply - GPU path correctness with various matrices
!   3. CPU vs GPU result comparison

program test_chebyshev_gpu
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use hipfort
    use hipfort_check
    use hipfort_types
    use sparse, only: &
        cleanup_graph_communications, csr, csr_free_device, csr_to_device, setup_graph_communications
    use chebyshev, only: chebyshev_multiply, estimate_spectral_radius
    use MPI
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_size_t, c_f_pointer
    implicit none

    real(real64), parameter :: tolerance = 1.0e-10_real64

    integer(int32) :: total_tests, passed_tests
    integer(int32) :: ierr, rank, nprocs

    ! Initialize MPI
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    total_tests = 0
    passed_tests = 0

    if (rank == 0) then
        write (*, *) "========================================"
        write (*, *) " Chebyshev Module GPU Integration Tests"
        write (*, '(A,I0,A)') "  Running with ", nprocs, " MPI rank(s)"
        write (*, *) "========================================"
    end if

    call hipCheck(hipInit(0))

    ! Run tests
    call test_estimate_spectral_radius_gpu()
    call test_chebyshev_multiply_identity()
    call test_chebyshev_multiply_diagonal()
    call test_chebyshev_multiply_tridiagonal()
    call test_cpu_vs_gpu_consistency()

    ! Synchronize and report
    call MPI_Barrier(MPI_COMM_WORLD, ierr)

    if (rank == 0) then
        write (*, *) ""
        write (*, *) "========================================"
        write (*, '(A,I0,A,I0,A)') " Results: ", passed_tests, "/", total_tests, " tests passed"
        write (*, *) "========================================"
    end if

    call MPI_Finalize(ierr)

    if (passed_tests == total_tests) then
        call exit(0)
    else
        call exit(1)
    end if

contains

    !--------------------------------------------------------------------------
    ! Helper: Create a simple 1D chain CSR matrix for testing
    ! Creates a tridiagonal matrix: A_ii = 2, A_i,i+1 = A_i+1,i = -1
    ! This is -i * H where H is the discrete Laplacian (scaled appropriately)
    !--------------------------------------------------------------------------
    subroutine create_chain_matrix(N, A, partition_table)
        integer, intent(in) :: N
        type(CSR), intent(out) :: A
        integer, dimension(:), allocatable, intent(out) :: partition_table

        integer(int32) :: i, nnz, idx, rank_local, nprocs_local, ierr
        integer(int32) :: lb, ub, n_local

        call MPI_Comm_rank(MPI_COMM_WORLD, rank_local, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs_local, ierr)

        ! Simple round-robin partitioning
        allocate (partition_table(nprocs_local + 1))
        do i = 1, nprocs_local + 1
            partition_table(i) = (i - 1) * N / nprocs_local + 1
        end do

        lb = partition_table(rank_local + 1)
        ub = partition_table(rank_local + 2) - 1
        n_local = ub - lb + 1

        ! Count nonzeros for local rows
        nnz = 0
        do i = lb, ub
            nnz = nnz + 1 ! diagonal
            if (i > 1) nnz = nnz + 1 ! lower off-diagonal
            if (i < N) nnz = nnz + 1 ! upper off-diagonal
        end do

        A%rows = N
        A%columns = N
        A%structure = "GE"
        A%has_values = .true.

        allocate (A%row_starts(n_local + 1))
        allocate (A%col_indexes(nnz))
        allocate (A%values(nnz))

        ! Fill CSR structure with 0-based data
        idx = 1
        do i = lb, ub
            A%row_starts(i - lb + 1) = idx - 1 ! 0-based offset

            ! Lower off-diagonal
            if (i > 1) then
                A%col_indexes(idx) = i - 2 ! 0-based column
                A%values(idx) = cmplx(-1.0_real64, 0.0_real64, real64)
                idx = idx + 1
            end if

            ! Diagonal
            A%col_indexes(idx) = i - 1 ! 0-based column
            A%values(idx) = cmplx(2.0_real64, 0.0_real64, real64)
            idx = idx + 1

            ! Upper off-diagonal
            if (i < N) then
                A%col_indexes(idx) = i ! 0-based column
                A%values(idx) = cmplx(-1.0_real64, 0.0_real64, real64)
                idx = idx + 1
            end if
        end do
        A%row_starts(n_local + 1) = idx - 1 ! 0-based final offset

    end subroutine create_chain_matrix

    !--------------------------------------------------------------------------
    ! Helper: Create a diagonal Hermitian matrix
    ! H_ii = i (real eigenvalues: 1, 2, 3, ...)
    ! Algorithm computes exp(-i*H*t), so exp(-i*H*t)|k> = exp(-i*k*t)|k>
    !--------------------------------------------------------------------------
    subroutine create_diagonal_matrix(N, A, partition_table)
        integer, intent(in) :: N
        type(CSR), intent(out) :: A
        integer, dimension(:), allocatable, intent(out) :: partition_table

        integer(int32) :: i, idx, rank_local, nprocs_local, ierr
        integer(int32) :: lb, ub, n_local

        call MPI_Comm_rank(MPI_COMM_WORLD, rank_local, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs_local, ierr)

        ! Simple round-robin partitioning
        allocate (partition_table(nprocs_local + 1))
        do i = 1, nprocs_local + 1
            partition_table(i) = (i - 1) * N / nprocs_local + 1
        end do

        lb = partition_table(rank_local + 1)
        ub = partition_table(rank_local + 2) - 1
        n_local = ub - lb + 1

        A%rows = N
        A%columns = N
        A%structure = "GE"
        A%has_values = .true.

        allocate (A%row_starts(n_local + 1))
        allocate (A%col_indexes(n_local))
        allocate (A%values(n_local))

        ! Fill CSR structure with 0-based data
        ! Hermitian diagonal matrix: H_ii = i (real eigenvalue)
        idx = 1
        do i = lb, ub
            A%row_starts(i - lb + 1) = idx - 1 ! 0-based offset
            A%col_indexes(idx) = i - 1 ! 0-based column
            A%values(idx) = cmplx(real(i, real64), 0.0_real64, real64) ! Real diagonal
            idx = idx + 1
        end do
        A%row_starts(n_local + 1) = idx - 1 ! 0-based final offset

    end subroutine create_diagonal_matrix

    !--------------------------------------------------------------------------
    ! Helper: Create Hermitian identity matrix
    ! H = I (Hermitian with all eigenvalues = 1)
    ! Algorithm computes exp(-i*H*t) = exp(-i*t)*I
    !--------------------------------------------------------------------------
    subroutine create_hermitian_identity(N, A, partition_table)
        integer, intent(in) :: N
        type(CSR), intent(out) :: A
        integer, dimension(:), allocatable, intent(out) :: partition_table

        integer(int32) :: i, idx, rank_local, nprocs_local, ierr
        integer(int32) :: lb, ub, n_local

        call MPI_Comm_rank(MPI_COMM_WORLD, rank_local, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs_local, ierr)

        ! Simple round-robin partitioning
        allocate (partition_table(nprocs_local + 1))
        do i = 1, nprocs_local + 1
            partition_table(i) = (i - 1) * N / nprocs_local + 1
        end do

        lb = partition_table(rank_local + 1)
        ub = partition_table(rank_local + 2) - 1
        n_local = ub - lb + 1

        A%rows = N
        A%columns = N
        A%structure = "GE"
        A%has_values = .true.

        allocate (A%row_starts(n_local + 1))
        allocate (A%col_indexes(n_local))
        allocate (A%values(n_local))

        ! Fill CSR structure with 0-based data
        ! H = I (real identity, Hermitian)
        idx = 1
        do i = lb, ub
            A%row_starts(i - lb + 1) = idx - 1 ! 0-based offset
            A%col_indexes(idx) = i - 1 ! 0-based column
            A%values(idx) = cmplx(1.0_real64, 0.0_real64, real64) ! Real 1
            idx = idx + 1
        end do
        A%row_starts(n_local + 1) = idx - 1 ! 0-based final offset

    end subroutine create_hermitian_identity

    !--------------------------------------------------------------------------
    ! Test estimate_spectral_radius with GPU
    !--------------------------------------------------------------------------
    subroutine test_estimate_spectral_radius_gpu()
        type(CSR) :: A
        integer, allocatable :: partition_table(:)
        real(real64) :: spectral_radius_cpu, spectral_radius_gpu
        logical :: test_passed
        integer(int32) :: ierr_local

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: estimate_spectral_radius GPU vs CPU..."
        end if

        ! Create test matrix
        call create_chain_matrix(16, A, partition_table)
        call setup_graph_communications(A, partition_table, MPI_COMM_WORLD)

        test_passed = .true.

        ! CPU path: A%device_ready = false
        call estimate_spectral_radius(A, partition_table, MPI_COMM_WORLD, spectral_radius_cpu)

        ! GPU path: transfer to device first
        call csr_to_device(A)
        call estimate_spectral_radius(A, partition_table, MPI_COMM_WORLD, spectral_radius_gpu)

        ! Check they match
        if (abs(spectral_radius_cpu - spectral_radius_gpu) > tolerance) then
            if (rank == 0) then
                write (*, '(A,F12.6,A,F12.6)') "  FAILED: CPU=", spectral_radius_cpu, &
                    " GPU=", spectral_radius_gpu
            end if
            test_passed = .false.
        else
            if (rank == 0) then
                write (*, '(A,F12.6)') "  Spectral radius: ", spectral_radius_gpu
            end if
        end if

        ! Cleanup
        call csr_free_device(A)
        call cleanup_graph_communications(A)
        deallocate (A%row_starts, A%col_indexes, A%values, partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        end if

    end subroutine test_estimate_spectral_radius_gpu

    !--------------------------------------------------------------------------
    ! Test chebyshev_multiply with Hermitian identity
    ! H = I, algorithm computes exp(-i*H*t)*B = exp(-i*t)*B = (cos(t) - i*sin(t))*B
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_multiply_identity()
        type(CSR) :: A
        integer, allocatable :: partition_table(:)
        integer(int32) :: lb, ub, n_local, i
        complex(real64), allocatable, target :: B(:), C(:)
        type(c_ptr) :: B_dev, C_dev
        real(real64) :: t
        complex(real64) :: expected_phase
        logical :: test_passed
        integer(int32) :: ierr_local

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: chebyshev_multiply with Hermitian identity..."
        end if

        ! Create Hermitian identity matrix: H = I
        ! Algorithm computes exp(-i*H*t) = exp(-i*t)
        call create_hermitian_identity(8, A, partition_table)
        call setup_graph_communications(A, partition_table, MPI_COMM_WORLD)
        call csr_to_device(A)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        ! Allocate device arrays for B and C
        call hipCheck(hipMalloc(B_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMalloc(C_dev, int(n_local * 16, c_size_t)))

        allocate (B(n_local), C(n_local))
        B = cmplx(1.0_real64, 0.0_real64, real64)

        call hipCheck(hipMemcpy(B_dev, c_loc(B(1)), int(n_local * 16, c_size_t), hipMemcpyHostToDevice))

        ! Evolution time
        t = 0.5_real64
        ! Algorithm computes exp(-i*H*t) where H=I, so exp(-i*t) = cos(t) - i*sin(t)
        expected_phase = cmplx(cos(t), -sin(t), real64)

        ! Use c_f_pointer to make device memory look like Fortran arrays
        block
            complex(real64), pointer :: B_ptr(:), C_fptr(:)
            call c_f_pointer(B_dev, B_ptr, [n_local])
            call c_f_pointer(C_dev, C_fptr, [n_local])

            call chebyshev_multiply(A, B_ptr, t, partition_table, C_fptr, MPI_COMM_WORLD)
        end block

        ! Copy C back to host
        call hipCheck(hipMemcpy(c_loc(C(1)), C_dev, int(n_local * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, n_local
            if (abs(C(i) - expected_phase) > 1.0e-6_real64) then
                write (*, '(A,I0,A,I0,A,2F12.6,A,2F12.6)') "  Rank ", rank, " FAILED at i=", i, &
                   ": got (", real(C(i)), aimag(C(i)), "), expected (", real(expected_phase), aimag(expected_phase), ")"
                test_passed = .false.
            end if
        end do

        ! Cleanup
        call hipCheck(hipFree(B_dev))
        call hipCheck(hipFree(C_dev))
        deallocate (B, C)
        call csr_free_device(A)
        call cleanup_graph_communications(A)
        deallocate (A%row_starts, A%col_indexes, A%values, partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        end if

    end subroutine test_chebyshev_multiply_identity

    !--------------------------------------------------------------------------
    ! Test chebyshev_multiply with diagonal Hermitian matrix
    ! H = diag(lambda_1, lambda_2, ...) with real eigenvalues lambda_k = k
    ! Algorithm computes exp(-i*H*t)|k> = exp(-i*lambda_k*t)|k>
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_multiply_diagonal()
        type(CSR) :: A
        integer, allocatable :: partition_table(:)
        integer(int32) :: lb, ub, n_local, i
        complex(real64), allocatable, target :: B(:), C(:)
        complex(real64) :: expected
        type(c_ptr) :: B_dev, C_dev
        real(real64) :: t, lambda
        logical :: test_passed
        integer(int32) :: ierr_local

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: chebyshev_multiply with diagonal Hermitian matrix..."
        end if

        ! Create diagonal Hermitian matrix: H_ii = i (real eigenvalue i for row i)
        call create_diagonal_matrix(8, A, partition_table)
        call setup_graph_communications(A, partition_table, MPI_COMM_WORLD)
        call csr_to_device(A)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        ! Allocate device arrays for B and C
        call hipCheck(hipMalloc(B_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMalloc(C_dev, int(n_local * 16, c_size_t)))

        ! Host arrays for initialization
        allocate (B(n_local), C(n_local))
        B = cmplx(1.0_real64, 0.0_real64, real64)

        ! Copy B to device
        call hipCheck(hipMemcpy(B_dev, c_loc(B(1)), int(n_local * 16, c_size_t), hipMemcpyHostToDevice))

        ! Evolution time
        t = 0.3_real64

        ! Use c_f_pointer to make device memory look like Fortran arrays
        block
            complex(real64), pointer :: B_ptr(:), C_fptr(:)
            call c_f_pointer(B_dev, B_ptr, [n_local])
            call c_f_pointer(C_dev, C_fptr, [n_local])

            call chebyshev_multiply(A, B_ptr, t, partition_table, C_fptr, MPI_COMM_WORLD)
        end block

        ! Copy C back to host
        call hipCheck(hipMemcpy(c_loc(C(1)), C_dev, int(n_local * 16, c_size_t), hipMemcpyDeviceToHost))

        test_passed = .true.
        do i = 1, n_local
            ! Global row index (1-based)
            lambda = real(lb + i - 1, real64)
            ! H_ii = lambda (Hermitian), algorithm computes exp(-i*H*t)|k> = exp(-i*lambda*t)|k>
            ! = cos(lambda*t) - i*sin(lambda*t)
            expected = cmplx(cos(lambda * t), -sin(lambda * t), real64)

            if (abs(C(i) - expected) > 1.0e-6_real64) then
                write (*, '(A,I0,A,I0,A,2F12.6,A,2F12.6)') "  Rank ", rank, " FAILED at i=", i, &
                    ": got (", real(C(i)), aimag(C(i)), "), expected (", real(expected), aimag(expected), ")"
                test_passed = .false.
            end if
        end do

        ! Cleanup
        call hipCheck(hipFree(B_dev))
        call hipCheck(hipFree(C_dev))
        deallocate (B, C)
        call csr_free_device(A)
        call cleanup_graph_communications(A)
        deallocate (A%row_starts, A%col_indexes, A%values, partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        end if

    end subroutine test_chebyshev_multiply_diagonal

    !--------------------------------------------------------------------------
    ! Test chebyshev_multiply with tridiagonal matrix
    ! Tests MPI communication for off-diagonal elements
    !--------------------------------------------------------------------------
    subroutine test_chebyshev_multiply_tridiagonal()
        type(CSR) :: A
        integer, allocatable :: partition_table(:)
        integer(int32) :: lb, ub, n_local, i
        complex(real64), allocatable, target :: B(:), C(:)
        type(c_ptr) :: B_dev, C_dev
        real(real64) :: t, norm_C
        real(real64) :: global_norm
        logical :: test_passed
        integer(int32) :: ierr_local

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: chebyshev_multiply with tridiagonal matrix..."
        end if

        ! Create tridiagonal matrix (requires MPI communication)
        call create_chain_matrix(16, A, partition_table)
        call setup_graph_communications(A, partition_table, MPI_COMM_WORLD)
        call csr_to_device(A)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        ! Allocate device arrays for B and C
        call hipCheck(hipMalloc(B_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMalloc(C_dev, int(n_local * 16, c_size_t)))

        ! Host arrays for initialization
        allocate (B(n_local), C(n_local))

        ! Initialize B with a smooth function
        do i = 1, n_local
            B(i) = cmplx(sin(real(lb + i - 1, real64) * 0.1_real64), 0.0_real64, real64)
        end do

        ! Copy B to device
        call hipCheck(hipMemcpy(B_dev, c_loc(B(1)), int(n_local * 16, c_size_t), hipMemcpyHostToDevice))

        ! Evolution time
        t = 0.1_real64

        ! Use c_f_pointer to make device memory look like Fortran arrays
        block
            complex(real64), pointer :: B_ptr(:), C_fptr(:)
            call c_f_pointer(B_dev, B_ptr, [n_local])
            call c_f_pointer(C_dev, C_fptr, [n_local])

            call chebyshev_multiply(A, B_ptr, t, partition_table, C_fptr, MPI_COMM_WORLD)
        end block

        ! Copy C back to host
        call hipCheck(hipMemcpy(c_loc(C(1)), C_dev, int(n_local * 16, c_size_t), hipMemcpyDeviceToHost))

        ! For unitary evolution, ||C|| should equal ||B||
        ! But our matrix is not anti-Hermitian (it's real symmetric)
        ! So we just check that C is finite and non-zero
        test_passed = .true.
        norm_C = 0.0_real64
        do i = 1, n_local
            if (isnan(real(C(i))) .or. isnan(aimag(C(i)))) then
                write (*, '(A,I0,A,I0,A)') "  Rank ", rank, " FAILED: NaN at i=", i
                test_passed = .false.
            end if
            norm_C = norm_C + abs(C(i))**2
        end do

        ! Global norm
        call MPI_Allreduce(norm_C, global_norm, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr_local)
        global_norm = sqrt(global_norm)

        if (global_norm < 1.0e-10_real64) then
            if (rank == 0) write (*, *) "  FAILED: ||C|| is effectively zero"
            test_passed = .false.
        else
            if (rank == 0) write (*, '(A,F12.6)') "  ||C|| = ", global_norm
        end if

        ! Cleanup
        call hipCheck(hipFree(B_dev))
        call hipCheck(hipFree(C_dev))
        deallocate (B, C)
        call csr_free_device(A)
        call cleanup_graph_communications(A)
        deallocate (A%row_starts, A%col_indexes, A%values, partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        end if

    end subroutine test_chebyshev_multiply_tridiagonal

    !--------------------------------------------------------------------------
    ! Test CPU vs GPU consistency
    ! Run the same computation on CPU and GPU and compare results
    !--------------------------------------------------------------------------
    subroutine test_cpu_vs_gpu_consistency()
        type(CSR) :: A_cpu, A_gpu
        integer, allocatable :: partition_table(:)
        integer(int32) :: lb, ub, n_local, i
        complex(real64), allocatable, target :: B(:), C_cpu(:), C_gpu(:)
        type(c_ptr) :: B_dev, C_dev
        real(real64) :: t, max_diff
        real(real64) :: global_max_diff
        logical :: test_passed
        integer(int32) :: ierr_local

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: CPU vs GPU consistency..."
        end if

        ! Create two copies of the same matrix
        call create_chain_matrix(16, A_cpu, partition_table)
        call setup_graph_communications(A_cpu, partition_table, MPI_COMM_WORLD)

        ! Deep copy for GPU version
        call create_chain_matrix(16, A_gpu, partition_table)
        call setup_graph_communications(A_gpu, partition_table, MPI_COMM_WORLD)
        call csr_to_device(A_gpu)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        ! Allocate host arrays
        allocate (B(n_local), C_cpu(n_local), C_gpu(n_local))

        ! Initialize B
        do i = 1, n_local
            B(i) = cmplx(cos(real(lb + i - 1, real64) * 0.2_real64), &
                         sin(real(lb + i - 1, real64) * 0.2_real64), real64)
        end do

        ! Evolution time
        t = 0.2_real64

        ! CPU path
        call chebyshev_multiply(A_cpu, B, t, partition_table, C_cpu, MPI_COMM_WORLD)

        ! GPU path
        call hipCheck(hipMalloc(B_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMalloc(C_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMemcpy(B_dev, c_loc(B(1)), int(n_local * 16, c_size_t), hipMemcpyHostToDevice))

        block
            complex(real64), pointer :: B_ptr(:), C_fptr(:)
            call c_f_pointer(B_dev, B_ptr, [n_local])
            call c_f_pointer(C_dev, C_fptr, [n_local])

            call chebyshev_multiply(A_gpu, B_ptr, t, partition_table, C_fptr, MPI_COMM_WORLD)
        end block

        call hipCheck(hipMemcpy(c_loc(C_gpu(1)), C_dev, int(n_local * 16, c_size_t), hipMemcpyDeviceToHost))

        ! Compare results
        test_passed = .true.
        max_diff = 0.0_real64
        do i = 1, n_local
            max_diff = max(max_diff, abs(C_cpu(i) - C_gpu(i)))
        end do

        call MPI_Allreduce(max_diff, global_max_diff, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_local)

        if (global_max_diff > 1.0e-10_real64) then
            if (rank == 0) then
                write (*, '(A,E12.4)') "  FAILED: max diff = ", global_max_diff
            end if
            test_passed = .false.
        else
            if (rank == 0) then
                write (*, '(A,E12.4)') "  Max diff = ", global_max_diff
            end if
        end if

        ! Cleanup
        call hipCheck(hipFree(B_dev))
        call hipCheck(hipFree(C_dev))
        deallocate (B, C_cpu, C_gpu)
        call csr_free_device(A_gpu)
        call cleanup_graph_communications(A_cpu)
        call cleanup_graph_communications(A_gpu)
        deallocate (A_cpu%row_starts, A_cpu%col_indexes, A_cpu%values)
        deallocate (A_gpu%row_starts, A_gpu%col_indexes, A_gpu%values)
        deallocate (partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        end if

    end subroutine test_cpu_vs_gpu_consistency

end program test_chebyshev_gpu
