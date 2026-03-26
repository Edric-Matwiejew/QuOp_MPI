! test_comm_info_abort_validate.f90
!
! T1.10: Verify that validate() on a bad partition (sum != system_size)
! returns a nonzero error_code (without aborting).
!
! Requires 2 MPI processes.

program test_comm_info_abort_validate
    use, intrinsic :: iso_fortran_env, only: int32, int64
    use mpi
    use comm_info_module, only: quop_mpi_layout_t
    implicit none

    type(quop_mpi_layout_t), pointer :: ci
    integer(int32) :: rank, ierr_int, error_code, dup_comm, setter_err
    integer(int64) :: system_size

    call MPI_Init(ierr_int)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr_int)

    system_size = 100

    allocate (ci)
    call ci%set_MPI_COMM(MPI_COMM_WORLD, setter_err)
    call MPI_Comm_dup(MPI_COMM_WORLD, dup_comm, ierr_int)
    call ci%set_SUBCOMM(dup_comm, setter_err)

    call ci%set_system_size(system_size, setter_err)

    ! Set WRONG partitioning: each rank claims 100 elements = 200 total != 100
    call ci%set_alloc_local(system_size, setter_err)
    call ci%set_partitioning(system_size, int(rank, int64) * system_size, error_code=setter_err)
    call ci%build_partition_table(setter_err)

    ! This should return a nonzero error_code due to completeness check
    call ci%validate(system_size, error_code)

    if (error_code == 0) then
        write (*, '(A)') "ERROR: validate on bad partition returned error_code=0"
        call ci%destroy()
        deallocate (ci)
        call MPI_Finalize(ierr_int)
        call exit(1)
    end if

    ! Success: validate detected the bad partition without aborting
    call ci%destroy()
    deallocate (ci)
    call MPI_Finalize(ierr_int)
    call exit(0)

end program test_comm_info_abort_validate
