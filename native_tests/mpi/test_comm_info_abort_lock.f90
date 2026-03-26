! test_comm_info_abort_lock.f90
!
! T1.3: Verify that set_partitioning on a locked quop_mpi_layout_t
! returns a nonzero error_code (recoverable error, no abort).

program test_comm_info_abort_lock
    use, intrinsic :: iso_fortran_env, only: int32, int64
    use mpi
    use comm_info_module, only: quop_mpi_layout_t
    implicit none

    type(quop_mpi_layout_t), pointer :: ci
    integer(int32) :: ierr_int, dup_comm, setter_err

    call MPI_Init(ierr_int)

    allocate (ci)
    call ci%set_MPI_COMM(MPI_COMM_WORLD, setter_err)
    call MPI_Comm_dup(MPI_COMM_WORLD, dup_comm, ierr_int)
    call ci%set_SUBCOMM(dup_comm, setter_err)

    ! Lock the layout
    call ci%lock(setter_err)

    ! This should return nonzero error_code (locked guard)
    call ci%set_partitioning(int(10, int64), int(0, int64), error_code=setter_err)

    if (setter_err /= 0) then
        ! Success: locked layout correctly returned nonzero error_code
        call ci%unlock(setter_err)
        call ci%destroy()
        deallocate (ci)
        call MPI_Finalize(ierr_int)
        call exit(0)
    end if

    ! Should never reach here
    write (*, '(A)') "ERROR: set_partitioning on locked layout returned error_code=0"
    call ci%unlock(setter_err)
    call ci%destroy()
    deallocate (ci)
    call MPI_Finalize(ierr_int)
    call exit(1)

end program test_comm_info_abort_lock
