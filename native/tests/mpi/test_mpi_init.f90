! test_mpi_init.f90
! Basic test to verify MPI initialization works correctly

program test_mpi_init
    use, intrinsic :: iso_fortran_env, only: int32
    use mpi_f08
    implicit none

    integer(int32) :: rank, nprocs, ierr
    logical :: initialized, finalized
    integer(int32) :: passed, failed

    passed = 0
    failed = 0

    ! Test 1: MPI_Init should succeed
    call MPI_Init(ierr)
    if (ierr == MPI_SUCCESS) then
        passed = passed + 1
    else
        failed = failed + 1
        write (*, '(A)') "FAIL: MPI_Init returned error"
    end if

    ! Test 2: Check initialization status
    call MPI_Initialized(initialized, ierr)
    if (initialized) then
        passed = passed + 1
    else
        failed = failed + 1
        write (*, '(A)') "FAIL: MPI_Initialized returned false after MPI_Init"
    end if

    ! Test 3: Get rank
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    if (ierr == MPI_SUCCESS .and. rank >= 0) then
        passed = passed + 1
    else
        failed = failed + 1
        write (*, '(A)') "FAIL: MPI_Comm_rank failed"
    end if

    ! Test 4: Get size
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    if (ierr == MPI_SUCCESS .and. nprocs >= 1) then
        passed = passed + 1
    else
        failed = failed + 1
        write (*, '(A)') "FAIL: MPI_Comm_size failed"
    end if

    ! Test 5: Barrier should succeed
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (ierr == MPI_SUCCESS) then
        passed = passed + 1
    else
        failed = failed + 1
        write (*, '(A)') "FAIL: MPI_Barrier failed"
    end if

    ! Print summary from rank 0
    if (rank == 0) then
        write (*, '(A)') "================================================"
        write (*, '(A,I0,A)') "MPI Init Test Results (", nprocs, " processes)"
        write (*, '(A)') "================================================"
        write (*, '(A,I0)') "  Passed: ", passed
        write (*, '(A,I0)') "  Failed: ", failed
        write (*, '(A)') "================================================"
        if (failed == 0) then
            write (*, '(A)') "All tests PASSED"
        else
            write (*, '(A)') "Some tests FAILED"
        end if
    end if

    ! Finalize
    call MPI_Finalize(ierr)

    ! Exit with error code if any test failed
    if (failed > 0) then
        call exit(1)
    end if

end program test_mpi_init
