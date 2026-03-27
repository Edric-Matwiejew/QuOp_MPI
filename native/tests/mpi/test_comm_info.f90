! test_comm_info.f90
!
! CTest for quop_mpi_layout_t -- non-abort tests (T1.1, T1.2, T1.4-T1.9).
! Runs at 1, 2, and 4 MPI processes.

program test_comm_info
    use, intrinsic :: iso_fortran_env, only: int32, int64, error_unit
    use mpi
    use comm_info_module, only: quop_mpi_layout_t, split_info_t
    implicit none

    integer(int32) :: rank, nprocs, ierr_int
    integer(int32) :: passed, failed
    type(quop_mpi_layout_t), pointer :: ci
    integer(int64) :: system_size
    integer(int64) :: base_size, remainder
    integer(int64) :: expected_local_i, expected_offset
    integer(int32) :: subcomm_size, subcomm_rank
    integer(int32) :: mpi_comm_handle
    integer(int32) :: setter_err
    logical :: flag
    integer(int32) :: i

    passed = 0
    failed = 0

    call MPI_Init(ierr_int)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr_int)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr_int)

    system_size = 100

    ! ================================================================
    ! T1.1: create -> check MPI_COMM stored, SUBCOMM valid dup, unlocked
    ! ================================================================
    allocate (ci)
    call ci%set_MPI_COMM(MPI_COMM_WORLD, setter_err)
    call MPI_Comm_dup(MPI_COMM_WORLD, mpi_comm_handle, ierr_int)
    call ci%set_SUBCOMM(mpi_comm_handle, setter_err)
    call MPI_Comm_size(ci%get_SUBCOMM(), subcomm_size, ierr_int)
    call ci%set_n_processes(int(subcomm_size, int64), setter_err)

    ! Check MPI_COMM is stored
    if (ci%get_MPI_COMM() == MPI_COMM_WORLD) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A)') "FAIL T1.1a: MPI_COMM not stored correctly"
    end if

    ! Check SUBCOMM is a valid dup (same size, different handle)
    if (subcomm_size == nprocs .and. ci%get_SUBCOMM() /= MPI_COMM_WORLD) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A)') "FAIL T1.1b: SUBCOMM not a valid dup of MPI_COMM"
    end if

    ! Check unlocked
    flag = ci%is_locked()
    if (.not. flag) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A)') "FAIL T1.1c: layout should start unlocked"
    end if

    ! ================================================================
    ! T1.2: set_partitioning -> read back fields
    ! ================================================================
    call ci%set_system_size(system_size, setter_err)
    base_size = system_size / int(nprocs, int64)
    remainder = mod(system_size, int(nprocs, int64))

    if (rank < int(remainder, int32)) then
        expected_local_i = base_size + 1
        expected_offset = int(rank, int64) * expected_local_i
    else
        expected_local_i = base_size
        expected_offset = int(rank, int64) * expected_local_i + remainder
    end if

    call ci%set_partitioning(expected_local_i, expected_offset, error_code=setter_err)

    if (ci%get_local_i() == expected_local_i) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A,I0,A,I0)') &
            "FAIL T1.2a: local_i=", ci%get_local_i(), " expected=", expected_local_i
    end if

    if (ci%get_local_i_offset() == expected_offset) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A,I0,A,I0)') &
            "FAIL T1.2b: local_i_offset=", ci%get_local_i_offset(), " expected=", expected_offset
    end if

    if (ci%get_system_size() == system_size) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A)') "FAIL T1.2c: system_size not stored correctly"
    end if

    ! ================================================================
    ! T1.4: build_partition_table -> verify boundaries
    ! ================================================================
    call ci%build_partition_table(setter_err)

    block
        integer(int64), pointer :: pt(:)
        pt => ci%get_partition_table()

        if (pt(1) == 1) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T1.4a: partition_table(1)=", pt(1)
        end if

        if (pt(nprocs + 1) == system_size + 1) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0,A,I0)') &
                "FAIL T1.4b: partition_table(n+1)=", pt(nprocs + 1), &
                " expected=", system_size + 1
        end if
    end block

    ! ================================================================
    ! T1.8: shrink (n >= 2 only, skip on 1 proc)
    ! ================================================================
    if (nprocs >= 2) then
        ! Create a fresh layout for shrink test
        block
            type(quop_mpi_layout_t), pointer :: ci_shrink
            integer(int32) :: shrink_rank

            integer(int32) :: dup_comm

            allocate (ci_shrink)
            call ci_shrink%set_MPI_COMM(MPI_COMM_WORLD, setter_err)
            call MPI_Comm_dup(MPI_COMM_WORLD, dup_comm, ierr_int)
            call ci_shrink%set_SUBCOMM(dup_comm, setter_err)
            call MPI_Comm_size(ci_shrink%get_SUBCOMM(), subcomm_size, ierr_int)
            call ci_shrink%set_n_processes(int(subcomm_size, int64), setter_err)

            call ci_shrink%shrink(int(1, int64), setter_err)

            call MPI_Comm_rank(MPI_COMM_WORLD, shrink_rank, ierr_int)

            if (shrink_rank == 0) then
                ! Rank 0 should get a valid SUBCOMM
                if (ci_shrink%get_SUBCOMM() /= MPI_COMM_NULL) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A)') "FAIL T1.8a: rank 0 got MPI_COMM_NULL after shrink"
                end if
            else
                ! Other ranks should get MPI_COMM_NULL
                if (ci_shrink%get_SUBCOMM() == MPI_COMM_NULL) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A,I0,A)') "FAIL T1.8b: rank ", shrink_rank, " did not get MPI_COMM_NULL"
                end if
            end if

            ! Clean up shrink test layout
            call ci_shrink%destroy()
            deallocate (ci_shrink)
        end block
    end if

    ! ================================================================
    ! T1.8b: filter_active_ranks (n >= 2 only, skip on 1 proc)
    ! ================================================================
    if (nprocs >= 2) then
        block
            type(quop_mpi_layout_t), pointer :: ci_filter
            integer(int32) :: filter_rank
            integer(int32) :: dup_comm

            allocate (ci_filter)
            call ci_filter%set_MPI_COMM(MPI_COMM_WORLD, setter_err)
            call MPI_Comm_dup(MPI_COMM_WORLD, dup_comm, ierr_int)
            call ci_filter%set_SUBCOMM(dup_comm, setter_err)
            call MPI_Comm_size(ci_filter%get_SUBCOMM(), subcomm_size, ierr_int)
            call ci_filter%set_n_processes(int(subcomm_size, int64), setter_err)
            call ci_filter%set_system_size(system_size, setter_err)

            if (rank == 0) then
                call ci_filter%set_partitioning(system_size, 0_int64, error_code=setter_err)
            else
                call ci_filter%set_partitioning(0_int64, system_size, error_code=setter_err)
            end if

            call ci_filter%filter_active_ranks(setter_err)
            call MPI_Comm_rank(MPI_COMM_WORLD, filter_rank, ierr_int)

            if (filter_rank == 0) then
                if (ci_filter%get_SUBCOMM() /= MPI_COMM_NULL .and. &
                    ci_filter%get_local_i() == system_size) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A)') "FAIL T1.8c: active rank not preserved correctly after filter_active_ranks"
                end if
            else
                if (ci_filter%get_SUBCOMM() == MPI_COMM_NULL) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A,I0,A)') "FAIL T1.8d: rank ", filter_rank, " not excluded by filter_active_ranks"
                end if
            end if

            call ci_filter%destroy()
            deallocate (ci_filter)
        end block
    end if

    ! ================================================================
    ! T1.9: validate on a good partition -> no abort
    ! ================================================================
    ! Restore correct partitioning (undone by T1.6 round-trip test)
    call ci%set_system_size(system_size, setter_err)
    call ci%set_alloc_local(expected_local_i, setter_err)
    call ci%set_partitioning(expected_local_i, expected_offset, error_code=setter_err)
    call ci%build_partition_table(setter_err)
    block
        integer(int32) :: val_err
        call ci%validate(system_size, val_err)
        if (val_err /= 0) then
            write (error_unit, '(A,I0)') "T1.9 FAIL: validate returned error_code=", val_err
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr_int)
        end if
    end block
    passed = passed + 1 ! If we get here, validate didn't abort

    ! ================================================================
    ! T1.7: destroy -> no crash, fields zeroed
    ! ================================================================
    call ci%destroy()

    if (ci%get_local_i() == 0 .and. &
        ci%get_local_i_offset() == 0 .and. &
        ci%get_system_size() == 0 .and. &
        ci%get_SUBCOMM() == MPI_COMM_NULL .and. &
        ci%get_MPI_COMM() == MPI_COMM_NULL .and. &
        .not. ci%is_locked() .and. &
        .not. associated(ci%get_partition_table())) then
        passed = passed + 1
    else
        failed = failed + 1
        if (rank == 0) write (*, '(A)') "FAIL T1.7: destroy did not zero all fields"
    end if

    deallocate (ci)

    ! ================================================================
    ! T1.extra: split_info_t destroy
    ! ================================================================
    block
        type(split_info_t) :: split
        integer(int32) :: dup_comm

        call MPI_Comm_dup(MPI_COMM_WORLD, dup_comm, ierr_int)
        split%SUBCOMM = dup_comm
        split%MPI_COMM = MPI_COMM_WORLD
        split%worker_id = 42
        split%n_workers = 7

        call split%destroy()

        if (split%SUBCOMM == MPI_COMM_NULL .and. &
            split%JACCOMM == MPI_COMM_NULL .and. &
            split%MPI_COMM == MPI_COMM_NULL .and. &
            split%worker_id == 0 .and. &
            split%n_workers == 1) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T1.extra: split_info_t destroy did not reset"
        end if
    end block

    ! ================================================================
    ! Summary
    ! ================================================================
    call MPI_Barrier(MPI_COMM_WORLD, ierr_int)

    if (rank == 0) then
        write (*, '(A)') "================================================"
        write (*, '(A,I0,A)') "comm_info Test Results (", nprocs, " processes)"
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

    call MPI_Finalize(ierr_int)

    if (failed > 0) then
        call exit(1)
    end if

end program test_comm_info
