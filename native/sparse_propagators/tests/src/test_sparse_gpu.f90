! test_sparse_gpu.f90
! Integration tests for GPU-enabled portions of the Sparse module
!
! Tests:
!   1. csr_to_device / csr_free_device - Device memory allocation and transfer
!   2. spmv_gpu - GPU-accelerated SpMV with MPI communication

program test_sparse_gpu
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_types
    use sparse, only: &
        cleanup_graph_communications, csr, csr_free_device, csr_to_device, setup_graph_communications, spmv_gpu
    use MPI
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
        write (*, *) " Sparse Module GPU Integration Tests"
        write (*, '(A,I0,A)') "  Running with ", nprocs, " MPI rank(s)"
        write (*, *) "========================================"
    end if

    call hipCheck(hipInit(0))

    ! Run tests
    call test_csr_to_device()
    call test_spmv_gpu_simple()

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

        ! Fill CSR structure with 0-based data:
        ! - row_starts contains 0-based offsets (0, nnz_row1, nnz_row1+nnz_row2, ...)
        ! - col_indexes contains 0-based column indices (0 to N-1)
        idx = 1
        do i = lb, ub
            A%row_starts(i - lb + 1) = idx - 1 ! 0-based offset

            ! Lower off-diagonal
            if (i > 1) then
                A%col_indexes(idx) = i - 2 ! 0-based column (i-1 in 1-based becomes i-2)
                A%values(idx) = cmplx(-1.0_real64, 0.0_real64, real64)
                idx = idx + 1
            end if

            ! Diagonal
            A%col_indexes(idx) = i - 1 ! 0-based column (i in 1-based becomes i-1)
            A%values(idx) = cmplx(2.0_real64, 0.0_real64, real64)
            idx = idx + 1

            ! Upper off-diagonal
            if (i < N) then
                A%col_indexes(idx) = i ! 0-based column (i+1 in 1-based becomes i)
                A%values(idx) = cmplx(-1.0_real64, 0.0_real64, real64)
                idx = idx + 1
            end if
        end do
        A%row_starts(n_local + 1) = idx - 1 ! 0-based final offset (nnz)

    end subroutine create_chain_matrix

    !--------------------------------------------------------------------------
    ! Test csr_to_device and csr_free_device
    !--------------------------------------------------------------------------
    subroutine test_csr_to_device()
        type(CSR) :: A
        integer, allocatable :: partition_table(:)
        integer(int32) :: ierr_local
        logical :: test_passed

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: csr_to_device / csr_free_device..."
        end if

        ! Create a small test matrix
        call create_chain_matrix(8, A, partition_table)

        ! Set up graph communications (required before csr_to_device)
        call setup_graph_communications(A, partition_table, MPI_COMM_WORLD)

        test_passed = .true.

        ! Transfer to device
        call csr_to_device(A)

        ! Check that device_ready flag is set
        if (.not. A%device_ready) then
            if (rank == 0) write (*, *) "  FAILED: device_ready not set after csr_to_device"
            test_passed = .false.
        end if

        ! Check that device pointers are associated
        if (.not. c_associated(A%row_starts_dev)) then
            if (rank == 0) write (*, *) "  FAILED: row_starts_dev not allocated"
            test_passed = .false.
        end if

        if (.not. c_associated(A%col_indexes_dev)) then
            if (rank == 0) write (*, *) "  FAILED: col_indexes_dev not allocated"
            test_passed = .false.
        end if

        ! Free device memory
        call csr_free_device(A)

        ! Check that device_ready is false
        if (A%device_ready) then
            if (rank == 0) write (*, *) "  FAILED: device_ready not cleared after csr_free_device"
            test_passed = .false.
        end if

        ! Cleanup
        call cleanup_graph_communications(A)
        deallocate (A%row_starts, A%col_indexes, A%values, partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        else
            write (*, '(A,I0,A)') "  Rank ", rank, " failed test_csr_to_device"
        end if

    end subroutine test_csr_to_device

    !--------------------------------------------------------------------------
    ! Test spmv_gpu with a simple matrix
    !--------------------------------------------------------------------------
    subroutine test_spmv_gpu_simple()
        type(CSR) :: A
        integer, allocatable :: partition_table(:)
        integer(int64) :: n_local, i
        integer(int32) :: lb, ub
        complex(real64), allocatable, target :: y_host(:), y_expected(:)
        type(c_ptr) :: x_dev, y_dev
        complex(real64) :: scalar
        logical :: test_passed
        integer(int32) :: ierr_local

        total_tests = total_tests + 1
        if (rank == 0) then
            write (*, *) ""
            write (*, *) "Test: spmv_gpu (simple)..."
        end if

        ! Only run on single rank for simplicity (tests MPI path when nprocs > 1)
        call create_chain_matrix(8, A, partition_table)
        call setup_graph_communications(A, partition_table, MPI_COMM_WORLD)
        call csr_to_device(A)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        allocate (y_host(n_local), y_expected(n_local))

        ! For tridiagonal [-1, 2, -1], A * [1,1,...,1]:
        ! Interior rows: -1 + 2 - 1 = 0
        ! First row: 2 - 1 = 1
        ! Last row: -1 + 2 = 1
        y_expected = cmplx(0.0_real64, 0.0_real64, real64)
        do i = 1, n_local
            if (lb + i - 1 == 1) then
                y_expected(i) = cmplx(1.0_real64, 0.0_real64, real64)
            else if (lb + i - 1 == 8) then
                y_expected(i) = cmplx(1.0_real64, 0.0_real64, real64)
            else
                y_expected(i) = cmplx(0.0_real64, 0.0_real64, real64)
            end if
        end do

        ! Allocate device vectors
        call hipCheck(hipMalloc(x_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMalloc(y_dev, int(n_local * 16, c_size_t)))

        ! Initialize x = [1, 1, 1, ...] on device
        block
            complex(real64), allocatable, target :: x_init(:)
            allocate (x_init(n_local))
            x_init = cmplx(1.0_real64, 0.0_real64, real64)
            call hipCheck(hipMemcpy(x_dev, c_loc(x_init(1)), int(n_local * 16, c_size_t), hipMemcpyHostToDevice))
            deallocate (x_init)
        end block
        call hipCheck(hipMemset(y_dev, 0, int(n_local * 16, c_size_t)))

        ! Call spmv_gpu with scalar = 1
        scalar = cmplx(1.0_real64, 0.0_real64, real64)

        ! Use c_f_pointer to make device memory look like Fortran arrays
        block
            complex(real64), pointer :: x_ptr(:), y_ptr(:)
            call c_f_pointer(x_dev, x_ptr, [int(n_local)])
            call c_f_pointer(y_dev, y_ptr, [int(n_local)])

            call spmv_gpu(A, x_ptr, y_ptr, scalar)
        end block

        ! Copy result back to host
        call hipCheck(hipMemcpy(c_loc(y_host(1)), y_dev, int(n_local * 16, c_size_t), hipMemcpyDeviceToHost))

        ! Check result
        test_passed = .true.
        do i = 1, n_local
            if (abs(y_host(i) - y_expected(i)) > tolerance) then
                write (*, '(A,I0,A,I0,A,2F12.6,A,2F12.6)') "  Rank ", rank, " FAILED at local index ", i, &
                    ": expected (", real(y_expected(i)), aimag(y_expected(i)), &
                    "), got (", real(y_host(i)), aimag(y_host(i)), ")"
                test_passed = .false.
            end if
        end do

        ! Cleanup
        call hipCheck(hipFree(x_dev))
        call hipCheck(hipFree(y_dev))
        call csr_free_device(A)
        call cleanup_graph_communications(A)
        deallocate (y_host, y_expected, A%row_starts, A%col_indexes, A%values, partition_table)

        call MPI_Barrier(MPI_COMM_WORLD, ierr_local)

        if (test_passed) then
            passed_tests = passed_tests + 1
            if (rank == 0) write (*, *) "  PASSED"
        else
            write (*, '(A,I0,A)') "  Rank ", rank, " failed test_spmv_gpu_simple"
        end if

    end subroutine test_spmv_gpu_simple

end program test_sparse_gpu
