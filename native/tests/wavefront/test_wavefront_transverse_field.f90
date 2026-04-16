! test_wavefront_transverse_field.f90
! Focused regression tests for the wavefront transverse-field propagator.
!
! The test layout intentionally keeps SUBCOMM larger than DEVCOMM by assigning
! one device rank per node. That exercises the host/device partition mismatch
! that previously caused gen_operator to reject valid wavefront layouts.

program test_wavefront_transverse_field
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding, only: c_loc
    use mpi
    use wavefront, only: wavefront_context
    use wavefront_transverse_field, only: transverse_field_propagator
    use comm_info_module, only: quop_mpi_layout_t
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use communicators, only: create_NODECOMM

    implicit none

    integer(int32) :: COMM, ierr
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: total_tests, passed_tests
    integer(int32) :: test_passed, global_passed
    integer(int32) :: first_fail_rank
    integer(int32) :: setup_error, prop_error, ctx_error
    integer(int32) :: i
    integer(int64) :: system_size, local_size, local_offset
    real(real64) :: tolerance, state_norm
    real(real64), dimension(1) :: theta
    character(len=256) :: error_msg
    logical :: setup_ok

    type(quop_mpi_layout_t), target :: ci
    type(wavefront_context) :: context
    type(transverse_field_propagator) :: propagator

    complex(real64), allocatable, target :: state_in(:), state_out(:)
    integer(int32), target :: operator_placeholder(1)
    integer(int64) :: operator_ptrs(1), operator_sizes(1)

    call MPI_Init(ierr)
    COMM = MPI_COMM_WORLD
    call MPI_Comm_rank(COMM, comm_rank, ierr)
    call MPI_Comm_size(COMM, comm_size, ierr)

    total_tests = 0
    passed_tests = 0
    tolerance = 1.0e-10_real64
    system_size = 512_int64
    setup_ok = .false.

    if (comm_rank == 0) then
        write (*, *) '========================================'
        write (*, *) ' Wavefront Transverse-Field Tests'
        write (*, *) ' Running with', comm_size, 'MPI processes'
        write (*, *) '========================================'
    end if

    call block_partition(system_size, comm_rank, comm_size, local_size, local_offset)

    if (local_size > 0) then
        allocate (state_in(local_size), state_out(local_size))
    else
        allocate (state_in(1), state_out(1))
    end if

    operator_placeholder(1) = 0_int32
    operator_ptrs(1) = transfer(c_loc(operator_placeholder(1)), operator_ptrs(1))
    operator_sizes(1) = 1_int64

    ! ====================================================================
    ! Test 1: Setup + operator generation on a partial DEVCOMM layout
    ! ====================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ''
    if (comm_rank == 0) write (*, *) 'Test 1: wavefront transverse-field setup...'
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ''

    call setup_test_layout(ci, COMM, system_size, setup_error)
    if (setup_error /= 0) then
        test_passed = 0
        write (error_msg, '(A,I0)') 'layout setup returned status ', setup_error
    end if

    if (test_passed == 1) then
        call context%setup(ci, ctx_error)
        if (ctx_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'context setup returned status ', ctx_error
        end if
    end if

    if (test_passed == 1) then
        call propagator%plan(context, prop_error)
        if (prop_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'propagator plan returned status ', prop_error
        end if
    end if

    if (test_passed == 1) then
        call propagator%gen_operator(operator_ptrs, operator_sizes, prop_error)
        if (prop_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'gen_operator returned status ', prop_error
        end if
    end if

    call synchronize_test_result(COMM, comm_rank, comm_size, test_passed, error_msg, global_passed, first_fail_rank)
    if (global_passed == 1) then
        passed_tests = passed_tests + 1
        setup_ok = .true.
    end if
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) '  PASSED: setup succeeded with SUBCOMM > DEVCOMM'
        else
            write (*, *) '  FAILED:', trim(error_msg)
        end if
    end if

    ! ====================================================================
    ! Test 2: Theta=0 acts as identity
    ! ====================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ''
    if (comm_rank == 0) write (*, *) 'Test 2: identity propagation...'
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ''

    if (.not. setup_ok) then
        test_passed = 0
        error_msg = 'wavefront transverse-field setup failed in the previous test'
    end if

    if (test_passed == 1 .and. local_size > 0) then
        do i = 1, int(local_size)
            state_in(i) = cmplx(real(local_offset + i, real64), -real(local_offset + i, real64), real64)
        end do
        state_out = cmplx(0.0_real64, 0.0_real64, real64)

        call context%set_state(state_in, ctx_error)
        if (ctx_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'set_state returned status ', ctx_error
        end if
    end if

    if (test_passed == 1) then
        theta(1) = 0.0_real64
        call propagator%propagate(theta, prop_error)
        if (prop_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'propagate(theta=0) returned status ', prop_error
        end if
    end if

    if (test_passed == 1 .and. local_size > 0) then
        call context%get_state(state_out, ctx_error)
        if (ctx_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'get_state returned status ', ctx_error
        end if
    end if

    if (test_passed == 1 .and. local_size > 0) then
        do i = 1, int(local_size)
            if (abs(state_out(i) - state_in(i)) > tolerance) then
                test_passed = 0
                write (error_msg, '(A,I0,A,I0)') &
                    'identity mismatch on rank ', comm_rank, ' at local index ', i
                exit
            end if
        end do
    end if

    call synchronize_test_result(COMM, comm_rank, comm_size, test_passed, error_msg, global_passed, first_fail_rank)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) '  PASSED: theta=0 preserves the host-visible state'
        else
            write (*, *) '  FAILED:', trim(error_msg)
        end if
    end if

    ! ====================================================================
    ! Test 3: Non-trivial propagation preserves norm
    ! ====================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ''
    if (comm_rank == 0) write (*, *) 'Test 3: norm preservation...'
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ''

    if (.not. setup_ok) then
        test_passed = 0
        error_msg = 'wavefront transverse-field setup failed in the previous test'
    end if

    if (test_passed == 1 .and. local_size > 0) then
        do i = 1, int(local_size)
            state_in(i) = cmplx(1.0_real64 / sqrt(real(system_size, real64)), 0.0_real64, real64)
        end do

        call context%set_state(state_in, ctx_error)
        if (ctx_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'set_state returned status ', ctx_error
        end if
    end if

    if (test_passed == 1) then
        theta(1) = 0.37_real64
        call propagator%propagate(theta, prop_error)
        if (prop_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'propagate(theta) returned status ', prop_error
        end if
    end if

    if (test_passed == 1) then
        state_norm = context%get_state_norm(ctx_error)
        if (ctx_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') 'get_state_norm returned status ', ctx_error
        end if
    end if

    if (test_passed == 1 .and. comm_rank == 0) then
        if (abs(state_norm - 1.0_real64) > 1.0e-6_real64) then
            test_passed = 0
            write (error_msg, '(A,F16.10)') 'state norm deviates from 1: ', state_norm
        end if
    end if

    call MPI_Bcast(test_passed, 1, MPI_INTEGER, 0, COMM, ierr)
    call synchronize_test_result(COMM, comm_rank, comm_size, test_passed, error_msg, global_passed, first_fail_rank)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) '  PASSED: non-trivial propagation preserves norm'
        else
            write (*, *) '  FAILED:', trim(error_msg)
        end if
    end if

    call propagator%destroy()
    call context%destroy()
    call ci%destroy()

    if (allocated(state_in)) deallocate (state_in)
    if (allocated(state_out)) deallocate (state_out)

    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) then
        write (*, *) ''
        write (*, *) '========================================'
        write (*, *) ' Passed', passed_tests, 'of', total_tests, 'tests'
        write (*, *) '========================================'
    end if

    call MPI_Finalize(ierr)

contains

    subroutine block_partition(total_size, rank, size_comm, local_i, local_offset)
        integer(int64), intent(in) :: total_size
        integer(int32), intent(in) :: rank, size_comm
        integer(int64), intent(out) :: local_i, local_offset

        integer(int64) :: base_size, remainder

        base_size = total_size / int(size_comm, int64)
        remainder = mod(total_size, int(size_comm, int64))

        if (int(rank, int64) < remainder) then
            local_i = base_size + 1_int64
            local_offset = int(rank, int64) * local_i
        else
            local_i = base_size
            local_offset = int(rank, int64) * local_i + remainder
        end if
    end subroutine block_partition

    subroutine setup_test_layout(ci, comm, total_size, error_code)
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(in) :: comm
        integer(int64), intent(in) :: total_size
        integer(int32), intent(out) :: error_code

        integer(int32) :: dup_comm, node_comm, dev_comm, dev_node_comm
        integer(int32) :: comm_rank_local, comm_size_local, node_rank
        integer(int32) :: devcomm_rank, devcomm_size, dev_node_size
        integer(int32) :: active_color, setter_err
        integer(int64) :: host_local_i, host_local_offset
        integer(int64) :: device_local_i, device_local_offset
        type(gpu_topology_t) :: topo

        error_code = 0
        setter_err = 0

        call MPI_Comm_rank(comm, comm_rank_local, ierr)
        call MPI_Comm_size(comm, comm_size_local, ierr)

        call ci%set_MPI_COMM(comm, setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call MPI_Comm_dup(comm, dup_comm, ierr)
        call ci%set_SUBCOMM(dup_comm, setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call ci%set_system_size(total_size, setter_err)
        call ci%set_n_processes(int(comm_size_local, int64), setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call block_partition(total_size, comm_rank_local, comm_size_local, host_local_i, host_local_offset)
        call ci%set_partitioning(host_local_i, host_local_offset, error_code=setter_err)
        call ci%set_alloc_local(host_local_i, setter_err)
        call ci%build_partition_table(setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call init_gpu_topology(ci%get_SUBCOMM(), topo)
        call ci%set_topology(topo, setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call create_NODECOMM(ci%get_SUBCOMM(), node_comm)
        call ci%set_NODECOMM(node_comm, setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call MPI_Comm_rank(ci%get_NODECOMM(), node_rank, ierr)
        if (node_rank == 0) then
            active_color = 1
        else
            active_color = MPI_UNDEFINED
        end if

        call MPI_Comm_split(ci%get_SUBCOMM(), active_color, comm_rank_local, dev_comm, ierr)
        call ci%set_DEVCOMM(dev_comm, setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        call MPI_Comm_split(ci%get_NODECOMM(), active_color, node_rank, dev_node_comm, ierr)
        call ci%set_DEVCOMM_NODE(dev_node_comm, setter_err)
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        if (ci%get_DEVCOMM() /= MPI_COMM_NULL) then
            call MPI_Comm_rank(ci%get_DEVCOMM(), devcomm_rank, ierr)
            call MPI_Comm_size(ci%get_DEVCOMM(), devcomm_size, ierr)
            call block_partition(total_size, devcomm_rank, devcomm_size, device_local_i, device_local_offset)
            call ci%set_partitioning(host_local_i, host_local_offset, device_local_i, device_local_offset, setter_err)
            call ci%set_device_alloc_local(device_local_i, setter_err)
        else
            call ci%set_partitioning(host_local_i, host_local_offset, 0_int64, 0_int64, setter_err)
            call ci%set_device_alloc_local(0_int64, setter_err)
        end if
        if (setter_err /= 0) then
            error_code = setter_err
            return
        end if

        if (ci%get_DEVCOMM_NODE() /= MPI_COMM_NULL) then
            call MPI_Comm_size(ci%get_DEVCOMM_NODE(), dev_node_size, ierr)
            call ci%set_device_n_processes(int(dev_node_size, int64), setter_err)
        else
            call ci%set_device_n_processes(0_int64, setter_err)
        end if

        error_code = setter_err
    end subroutine setup_test_layout

    subroutine synchronize_test_result(comm, rank, size_comm, local_passed, error_message, global_passed, first_fail_rank)
        integer(int32), intent(in) :: comm, rank, size_comm, local_passed
        character(len=*), intent(inout) :: error_message
        integer(int32), intent(out) :: global_passed, first_fail_rank

        integer(int32) :: fail_rank_local

        call MPI_Allreduce(local_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, comm, ierr)

        if (local_passed == 0) then
            fail_rank_local = rank
        else
            fail_rank_local = size_comm
        end if
        call MPI_Allreduce(fail_rank_local, first_fail_rank, 1, MPI_INTEGER, MPI_MIN, comm, ierr)

        if (global_passed == 0) then
            call MPI_Bcast(error_message, len(error_message), MPI_CHARACTER, first_fail_rank, comm, ierr)
        end if
    end subroutine synchronize_test_result

end program test_wavefront_transverse_field