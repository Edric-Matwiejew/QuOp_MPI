! test_comm_info_entry_points.f90
!
! CTest for entry-point pipeline:
!   discover_topology -> split_workers -> negotiate -> create_jaccomm
! Also exercises error paths and destroy_topology lifecycle.
!
! Runs at 1, 2, and 4 MPI processes.

program test_comm_info_entry_points
    use, intrinsic :: iso_fortran_env, only: int32, int64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_associated, c_f_pointer
    use mpi
    use comm_info_module, only: quop_mpi_layout_t, split_info_t, &
                                discover_topology, destroy_topology, split_workers, negotiate, &
                                create_jaccomm

    implicit none

    integer(int32) :: rank, nprocs, ierr
    integer(int32) :: passed, failed

    passed = 0
    failed = 0

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    ! ================================================================
    ! T-EP1: discover_topology returns a non-null pointer
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr
        integer(int32) :: topo_status

        call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
        if (topo_status /= 0) then
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP1a: discover_topology status=", topo_status
        end if
        if (c_associated(topo_ptr)) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP1: discover_topology returned null"
        end if

        ! Clean up
        call destroy_topology(topo_ptr)

        if (.not. c_associated(topo_ptr)) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP1b: destroy_topology did not null ptr"
        end if
    end block

    ! ================================================================
    ! T-EP2: split_workers with n_jacobian_workers=1 (degenerate)
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr, split_ptr
        type(split_info_t), pointer :: si
        integer(int32) :: wid, stat, sub_size, topo_status

        call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
        if (topo_status /= 0) then
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP2a: discover_topology status=", topo_status
        end if
        call split_workers(split_ptr, MPI_COMM_WORLD, topo_ptr, &
                           1, 0, wid, stat)

        if (stat == 0) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP2a: split status=", stat
        end if

        if (wid == 0) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP2b: worker_id=", wid
        end if

        ! Verify SUBCOMM has same size as MPI_COMM_WORLD
        call c_f_pointer(split_ptr, si)
        call MPI_Comm_size(si%SUBCOMM, sub_size, ierr)
        if (sub_size == nprocs) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0,A,I0)') &
                "FAIL T-EP2c: SUBCOMM size=", sub_size, " expected=", nprocs
        end if

        call si%destroy()
        deallocate (si)
        call destroy_topology(topo_ptr)
    end block

    ! ================================================================
    ! T-EP3: split_workers with invalid n_jacobian_workers -> status=1
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr, split_ptr
        type(split_info_t), pointer :: si
        integer(int32) :: wid, stat, topo_status

        call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
        if (topo_status /= 0) then
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP3a: discover_topology status=", topo_status
        end if
        call split_workers(split_ptr, MPI_COMM_WORLD, topo_ptr, &
                           nprocs + 1, 0, wid, stat)

        if (stat == 1) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP3: expected status=1, got=", stat
        end if

        ! Clean up: split still allocated, SUBCOMM should be null
        call c_f_pointer(split_ptr, si)
        if (si%SUBCOMM == MPI_COMM_NULL) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP3b: SUBCOMM should be null on error"
        end if

        deallocate (si)
        call destroy_topology(topo_ptr)
    end block

    ! ================================================================
    ! T-EP4: Full pipeline: discover -> split -> negotiate -> jaccomm
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr, split_ptr, layout_ptr
        type(split_info_t), pointer :: si
        type(quop_mpi_layout_t), pointer :: ci
        integer(int32) :: wid, stat, sub_rank, topo_status
        integer(int64) :: system_size
        integer(int64), dimension(0) :: dummy_props
        integer(int64), dimension(0) :: dummy_cbs

        system_size = 100

        call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
        if (topo_status /= 0) then
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP4a: discover_topology status=", topo_status
        end if
        call split_workers(split_ptr, MPI_COMM_WORLD, topo_ptr, &
                           1, 0, wid, stat)

        call negotiate(layout_ptr, split_ptr, topo_ptr, &
                       system_size, 0, 0, dummy_props, dummy_cbs, stat)

        if (stat == 0) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP4a: negotiate status=", stat
        end if

        ! Verify layout is locked
        call c_f_pointer(layout_ptr, ci)
        if (ci%is_locked()) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP4b: layout should be locked"
        end if

        ! Verify MPI_COMM is set (was bug #1)
        if (ci%get_MPI_COMM() /= MPI_COMM_NULL) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP4c: MPI_COMM is null after negotiate"
        end if

        ! Verify alloc_local is set (was bug #2)
        if (ci%get_alloc_local() > 0) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP4d: alloc_local is 0 after negotiate"
        end if

        ! Create JACCOMM
        call create_jaccomm(MPI_COMM_WORLD, split_ptr, layout_ptr)

        call c_f_pointer(split_ptr, si)
        call MPI_Comm_rank(ci%get_SUBCOMM(), sub_rank, ierr)

        if (sub_rank == 0) then
            ! Leader should have a valid JACCOMM
            if (si%JACCOMM /= MPI_COMM_NULL) then
                passed = passed + 1
            else
                failed = failed + 1
                if (rank == 0) write (*, '(A)') "FAIL T-EP4e: JACCOMM null for leader"
            end if
        else
            ! Non-leader should have MPI_COMM_NULL
            if (si%JACCOMM == MPI_COMM_NULL) then
                passed = passed + 1
            else
                failed = failed + 1
                write (*, '(A,I0,A)') "FAIL T-EP4e: rank ", rank, " non-leader got JACCOMM"
            end if
        end if

        ! Clean up
        call ci%unlock(ierr)
        call ci%destroy()
        deallocate (ci)
        call si%destroy()
        deallocate (si)
        call destroy_topology(topo_ptr)
    end block

    ! ================================================================
    ! T-EP5: negotiate with system_size=0 -> status=1, null layout
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr, split_ptr, layout_ptr
        type(split_info_t), pointer :: si
        integer(int32) :: wid, stat, topo_status
        integer(int64), dimension(0) :: dummy_props
        integer(int64), dimension(0) :: dummy_cbs

        call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
        if (topo_status /= 0) then
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP5a: discover_topology status=", topo_status
        end if
        call split_workers(split_ptr, MPI_COMM_WORLD, topo_ptr, &
                           1, 0, wid, stat)

        call negotiate(layout_ptr, split_ptr, topo_ptr, &
                       int(0, int64), 0, 0, dummy_props, dummy_cbs, stat)

        if (stat == 1) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP5a: expected status=1, got=", stat
        end if

        if (.not. c_associated(layout_ptr)) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP5b: layout_ptr should be null"
        end if

        ! Clean up: split still owns SUBCOMM (negotiate didn't take it)
        call c_f_pointer(split_ptr, si)
        call si%destroy()
        deallocate (si)
        call destroy_topology(topo_ptr)
    end block

    ! ================================================================
    ! T-EP6: create_jaccomm with null layout_ptr (error path from #2)
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr, split_ptr, layout_ptr
        type(split_info_t), pointer :: si
        integer(int32) :: wid, stat, topo_status
        integer(int64), dimension(0) :: dummy_props
        integer(int64), dimension(0) :: dummy_cbs

        call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
        if (topo_status /= 0) then
            failed = failed + 1
            if (rank == 0) write (*, '(A,I0)') "FAIL T-EP6a: discover_topology status=", topo_status
        end if
        call split_workers(split_ptr, MPI_COMM_WORLD, topo_ptr, &
                           1, 0, wid, stat)

        ! negotiate with bad system_size -> null layout
        call negotiate(layout_ptr, split_ptr, topo_ptr, &
                       int(-1, int64), 0, 0, dummy_props, dummy_cbs, stat)

        ! This should NOT crash (was bug #2)
        call create_jaccomm(MPI_COMM_WORLD, split_ptr, layout_ptr)

        ! JACCOMM should be MPI_COMM_NULL (all ranks inactive)
        call c_f_pointer(split_ptr, si)
        if (si%JACCOMM == MPI_COMM_NULL) then
            passed = passed + 1
        else
            failed = failed + 1
            if (rank == 0) write (*, '(A)') "FAIL T-EP6: JACCOMM should be null with null layout"
        end if

        call si%destroy()
        deallocate (si)
        call destroy_topology(topo_ptr)
    end block

    ! ================================================================
    ! T-EP7: split_workers with n_jacobian_workers > 1 (needs >= 2 procs)
    ! ================================================================
    if (nprocs >= 2) then
        block
            type(c_ptr) :: topo_ptr, split_ptr, layout_ptr
            type(split_info_t), pointer :: si
            type(quop_mpi_layout_t), pointer :: ci
            integer(int32) :: wid, stat, jac_size, topo_status
            integer(int64) :: system_size
            integer(int64), dimension(0) :: dummy_props
            integer(int64), dimension(0) :: dummy_cbs

            system_size = 100

            call discover_topology(topo_ptr, MPI_COMM_WORLD, 0, topo_status)
            if (topo_status /= 0) then
                failed = failed + 1
                if (rank == 0) write (*, '(A,I0)') "FAIL T-EP7a: discover_topology status=", topo_status
            end if
            call split_workers(split_ptr, MPI_COMM_WORLD, topo_ptr, &
                               2, 0, wid, stat)

            if (stat == 0) then
                passed = passed + 1
            else
                failed = failed + 1
                if (rank == 0) write (*, '(A,I0)') "FAIL T-EP7a: split status=", stat
            end if

            ! Each rank should have worker_id 0 or 1
            if (wid >= 0 .and. wid <= 1) then
                passed = passed + 1
            else
                failed = failed + 1
                if (rank == 0) write (*, '(A,I0)') "FAIL T-EP7b: worker_id=", wid
            end if

            call negotiate(layout_ptr, split_ptr, topo_ptr, &
                           system_size, 0, 0, dummy_props, dummy_cbs, stat)

            if (stat == 0) then
                passed = passed + 1
            else
                failed = failed + 1
                if (rank == 0) write (*, '(A,I0)') "FAIL T-EP7c: negotiate status=", stat
            end if

            call create_jaccomm(MPI_COMM_WORLD, split_ptr, layout_ptr)

            call c_f_pointer(split_ptr, si)
            call c_f_pointer(layout_ptr, ci)

            ! Check JACCOMM membership (new semantics):
            !   worker_id > 0  -> ALL ranks get JACCOMM
            !   worker_id == 0 -> only subcomm rank 0 (optimizer) gets JACCOMM
            call MPI_Comm_rank(ci%get_SUBCOMM(), stat, ierr) ! reuse stat for sub_rank
            if (si%worker_id > 0) then
                ! Worker subcomm: every rank should have JACCOMM
                if (si%JACCOMM /= MPI_COMM_NULL) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A,I0,A)') "FAIL T-EP7d: rank ", rank, &
                        " worker got null JACCOMM"
                end if
            else if (stat == 0) then
                ! Optimizer subcomm leader: should have JACCOMM
                if (si%JACCOMM /= MPI_COMM_NULL) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A,I0,A)') "FAIL T-EP7d: rank ", rank, &
                        " optimizer leader got null JACCOMM"
                end if
            else
                ! Optimizer subcomm non-leader: should NOT have JACCOMM
                if (si%JACCOMM == MPI_COMM_NULL) then
                    passed = passed + 1
                else
                    failed = failed + 1
                    write (*, '(A,I0,A)') "FAIL T-EP7d: rank ", rank, &
                        " optimizer non-leader got JACCOMM"
                end if
            end if

            ! Clean up
            call ci%unlock(ierr)
            call ci%destroy()
            deallocate (ci)
            call si%destroy()
            deallocate (si)
            call destroy_topology(topo_ptr)
        end block
    end if

    ! ================================================================
    ! T-EP8: destroy_topology on null ptr is no-op
    ! ================================================================
    block
        type(c_ptr) :: topo_ptr
        topo_ptr = c_null_ptr
        call destroy_topology(topo_ptr)
        passed = passed + 1 ! If we get here, it didn't crash
    end block

    ! ================================================================
    ! Summary
    ! ================================================================
    call MPI_Barrier(MPI_COMM_WORLD, ierr)

    if (rank == 0) then
        write (*, '(A)') "================================================"
        write (*, '(A,I0,A)') "Entry Point Test Results (", nprocs, " processes)"
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

    call MPI_Finalize(ierr)

    if (failed > 0) then
        call exit(1)
    end if

end program test_comm_info_entry_points
