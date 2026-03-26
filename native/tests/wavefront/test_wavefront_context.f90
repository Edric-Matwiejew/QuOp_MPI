! test_wavefront_context.f90
! Integration tests for wavefront_context via context_wrapper interface
!
! Tests validate the complete wavefront context functionality:
!   1. Context setup/destroy lifecycle
!   2. State vector round-trip (set_state -> get_state)
!   3. Observables round-trip (set_observables -> get_observables)
!   4. State norm calculation (normalized state should return 1.0)
!   5. Expectation value calculation (known state/observable pairs)
!
! These tests run with both single-GPU (all ranks share GPU 0) and
! multi-GPU configurations to validate GPU transfer code paths.
!
! NOTE: Non-GPU ranks may have NULL context or context without device.
! Tests skip device-specific operations on such ranks and reduce results.

program test_wavefront_context
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use iso_c_binding, only: c_ptr, c_null_ptr, c_associated, c_loc
    use mpi
    use context_wrapper, only: &
        destroy, get_expectation_value, get_observables, get_state, get_state_norm, set_observables, set_state, setup
    use comm_info_module, only: quop_mpi_layout_t
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use communicators, only: create_NODECOMM, create_devcomm_with_data
    implicit none

    integer(int32) :: COMM, ierr
    integer(int32) :: comm_rank, comm_size
    type(c_ptr) :: context_ptr
    type(c_ptr) :: ci_ptr
    type(quop_mpi_layout_t), pointer, save :: ci
    type(gpu_topology_t) :: topo
    logical :: context_valid
    integer(int32) :: dn_size
    integer(int32) :: devcomm_rank, devcomm_size
    integer(int64) :: dev_base, dev_remainder

    integer(int64) :: system_size, elements_per_rank

    ! Test data
    complex(real64), allocatable :: state_in(:), state_out(:)
    real(real64), allocatable :: obs_in(:), obs_out(:)
    real(real64) :: state_norm, expectation_value
    real(real64) :: expected_norm, expected_expval
    real(real64) :: tolerance

    integer(int32) :: total_tests, passed_tests
    integer(int32) :: test_passed, global_passed
    integer(int32) :: fail_rank, first_fail_rank
    integer(int32) :: context_error
    integer(int32) :: i
    integer(int32) :: setter_err
    integer(int64) :: local_size, local_offset
    character(len=256) :: error_msg

    ! Initialize MPI
    call MPI_Init(ierr)
    COMM = MPI_COMM_WORLD
    call MPI_Comm_rank(COMM, comm_rank, ierr)
    call MPI_Comm_size(COMM, comm_size, ierr)

    total_tests = 0
    passed_tests = 0
    tolerance = 1.0e-10_real64

    if (comm_rank == 0) then
        write (*, *) "========================================"
        write (*, *) " Wavefront Context Integration Tests"
        write (*, *) " Running with", comm_size, "MPI processes"
        write (*, *) "========================================"
    end if

    ! Derive system_size from comm_size to ensure clean division
    ! Use at least 256 elements per rank for meaningful tests
    elements_per_rank = 256_int64
    system_size = elements_per_rank * int(comm_size, int64)

    if (comm_rank == 0) then
        write (*, *) " System size:", system_size, "(", elements_per_rank, "elements/rank)"
    end if

    ! ========================================================================
    ! Test 1: Context Setup
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 1: Context setup..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    ! Create a quop_mpi_layout_t, configure it, and build partition table
    block
        integer(int32) :: dup_comm, node_comm, dev_comm, dev_node_comm
        integer(int64) :: my_local_i, my_offset
        integer(int64) :: my_dev_local_i, my_dev_offset, my_dev_alloc

        allocate (ci)
        call ci%set_MPI_COMM(COMM, setter_err)
        call MPI_Comm_dup(COMM, dup_comm, ierr)
        call ci%set_SUBCOMM(dup_comm, setter_err)
        call ci%set_system_size(system_size, setter_err)
        call ci%set_n_processes(int(comm_size, int64), setter_err)
        ! Default even partitioning
        my_local_i = system_size / int(comm_size, int64)
        if (int(comm_rank, int64) < mod(system_size, int(comm_size, int64))) then
            my_local_i = my_local_i + 1_int64
        end if
        my_offset = 0_int64
        do i = 0, comm_rank - 1
            my_offset = my_offset + system_size / int(comm_size, int64)
            if (int(i, int64) < mod(system_size, int(comm_size, int64))) then
                my_offset = my_offset + 1_int64
            end if
        end do
        call ci%set_partitioning(my_local_i, my_offset, error_code=setter_err)
        call ci%set_alloc_local(my_local_i, setter_err)
        call ci%build_partition_table(setter_err)

        ! Detect GPU topology and create wavefront communicator hierarchy
        call init_gpu_topology(ci%get_SUBCOMM(), topo)
        call ci%set_topology(topo, setter_err)
        call create_NODECOMM(ci%get_SUBCOMM(), node_comm)
        call ci%set_NODECOMM(node_comm, setter_err)
        call create_devcomm_with_data(ci%get_SUBCOMM(), ci%get_NODECOMM(), ci%get_topology(), &
                                      (ci%get_local_i() > 0), dev_comm, dev_node_comm)
        call ci%set_DEVCOMM(dev_comm, setter_err)
        call ci%set_DEVCOMM_NODE(dev_node_comm, setter_err)

        ! Populate device partitioning fields from communicators
        if (ci%get_DEVCOMM_NODE() /= MPI_COMM_NULL) then
            call MPI_Comm_size(ci%get_DEVCOMM_NODE(), dn_size, ierr)
            call ci%set_device_n_processes(int(dn_size, int64), setter_err)
        else
            call ci%set_device_n_processes(0_int64, setter_err)
        end if

        ! Block-distribute the ENTIRE system_size across DEVCOMM ranks.
        ! This mirrors what device_block_distribute() does in production:
        ! each GPU rank is responsible for a contiguous slice of the full
        ! system, not just its own host partition.
        if (ci%get_DEVCOMM() /= MPI_COMM_NULL) then
            call MPI_Comm_rank(ci%get_DEVCOMM(), devcomm_rank, ierr)
            call MPI_Comm_size(ci%get_DEVCOMM(), devcomm_size, ierr)
            dev_base = system_size / int(devcomm_size, int64)
            dev_remainder = mod(system_size, int(devcomm_size, int64))
            if (int(devcomm_rank, int64) < dev_remainder) then
                my_dev_local_i = dev_base + 1_int64
                my_dev_offset = int(devcomm_rank, int64) * my_dev_local_i
            else
                my_dev_local_i = dev_base
                my_dev_offset = int(devcomm_rank, int64) * my_dev_local_i + dev_remainder
            end if
            my_dev_alloc = my_dev_local_i
        else
            my_dev_local_i = 0
            my_dev_offset = 0
            my_dev_alloc = 0
        end if
        call ci%set_partitioning(ci%get_local_i(), ci%get_local_i_offset(), &
                                 my_dev_local_i, my_dev_offset, setter_err)
        call ci%set_device_alloc_local(my_dev_alloc, setter_err)
    end block

    ci_ptr = c_loc(ci)

    ! Extract local partition info for this rank
    local_size = ci%get_local_i()
    local_offset = ci%get_local_i_offset()

    ! Setup context
    context_ptr = c_null_ptr
    call setup(context_ptr, ci_ptr, context_error)
    if (context_error /= 0) then
        test_passed = 0
        write (error_msg, '(A,I0)') "context setup returned status ", context_error
    end if

    ! Check if context is valid (non-null) for this rank
    context_valid = c_associated(context_ptr)

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Context setup completed"
            write (*, *) "          System size:", system_size
            write (*, *) "          Local size:", local_size, "offset:", local_offset
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! Allocate test arrays based on local partition
    ! Guard against zero-size allocations
    if (local_size > 0) then
        allocate (state_in(local_size))
        allocate (state_out(local_size))
        allocate (obs_in(local_size))
        allocate (obs_out(local_size))
    else
        allocate (state_in(1))
        allocate (state_out(1))
        allocate (obs_in(1))
        allocate (obs_out(1))
    end if

    ! ========================================================================
    ! Test 2: State Vector Round-Trip
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 2: State vector round-trip..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    ! Only perform device operations if context is valid and local_size > 0
    if (context_valid .and. local_size > 0) then
        ! Initialize state with known pattern: state[i] = (i+1, -(i+1))
        do i = 1, int(local_size)
            state_in(i) = cmplx(real(local_offset + i, real64), -real(local_offset + i, real64), real64)
        end do
        state_out = cmplx(0.0_real64, 0.0_real64, real64)

        write (*, *) "DEBUG: Rank", comm_rank, "calling set_state, local_size=", local_size
        flush (6)

        ! Set state on device
        call set_state(context_ptr, local_size, state_in, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "set_state returned status ", context_error
        end if

        if (test_passed == 1) then
            write (*, *) "DEBUG: Rank", comm_rank, "set_state complete, calling get_state"
            flush (6)

            ! Get state back from device
            call get_state(context_ptr, local_size, state_out, context_error)
            if (context_error /= 0) then
                test_passed = 0
                write (error_msg, '(A,I0)') "get_state returned status ", context_error
            end if
        end if

        if (test_passed == 1) then
            write (*, *) "DEBUG: Rank", comm_rank, "get_state complete"
            flush (6)

            ! Verify round-trip
            do i = 1, int(local_size)
                if (abs(state_out(i) - state_in(i)) > tolerance) then
                    test_passed = 0
                    write (error_msg, '(A,I0,A,I0,A,I0,A,2F12.6,A,2F12.6)') &
                        "Rank ", comm_rank, " mismatch at local index ", i, &
                        " (global ", local_offset + i - 1, "): got (", &
                        real(state_out(i)), aimag(state_out(i)), &
                        "), expected (", real(state_in(i)), aimag(state_in(i))
                    exit
                end if
            end do
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (test_passed == 0) then
        fail_rank = comm_rank
    else
        fail_rank = comm_size
    end if
    call MPI_Allreduce(fail_rank, first_fail_rank, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 0) then
        call MPI_Bcast(error_msg, len(error_msg), MPI_CHARACTER, first_fail_rank, COMM, ierr)
    end if
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: State vector round-trip verified"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 3: Observables Round-Trip
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 3: Observables round-trip..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    if (context_valid .and. local_size > 0) then
        ! Initialize observables with known pattern: obs[i] = i * 0.5
        do i = 1, int(local_size)
            obs_in(i) = real(local_offset + i, real64) * 0.5_real64
        end do
        obs_out = 0.0_real64

        ! Set observables on device
        call set_observables(context_ptr, local_size, obs_in, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "set_observables returned status ", context_error
        end if

        if (test_passed == 1) then
            ! Get observables back from device
            call get_observables(context_ptr, local_size, obs_out, context_error)
            if (context_error /= 0) then
                test_passed = 0
                write (error_msg, '(A,I0)') "get_observables returned status ", context_error
            end if
        end if

        if (test_passed == 1) then
            ! Verify round-trip
            do i = 1, int(local_size)
                if (abs(obs_out(i) - obs_in(i)) > tolerance) then
                    test_passed = 0
                    write (error_msg, '(A,I0,A,F12.6,A,F12.6)') &
                        "Mismatch at index ", i, ": got ", obs_out(i), ", expected ", obs_in(i)
                    exit
                end if
            end do
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Observables round-trip verified"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 4: State Norm Calculation
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 4: State norm calculation..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    if (context_valid .and. local_size > 0) then
        ! Create a normalized state: |psi> = (1/sqrt(N)) * (1, 1, 1, ..., 1)
        ! Norm should be exactly 1.0
        do i = 1, int(local_size)
            state_in(i) = cmplx(1.0_real64 / sqrt(real(system_size, real64)), 0.0_real64, real64)
        end do

        call set_state(context_ptr, local_size, state_in, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "set_state returned status ", context_error
        end if
    end if

    ! get_state_norm is collective over SUBCOMM - all active ranks must call it
    if (context_valid) then
        call get_state_norm(context_ptr, state_norm, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "get_state_norm returned status ", context_error
        end if
    end if

    expected_norm = 1.0_real64

    ! All active ranks receive the same result; rank 0 validates it
    if (comm_rank == 0) then
        if (abs(state_norm - expected_norm) > 1.0e-6_real64) then
            test_passed = 0
            write (error_msg, '(A,F16.10,A,F16.10)') &
                "State norm = ", state_norm, ", expected ", expected_norm
        end if
    end if

    call MPI_Bcast(test_passed, 1, MPI_INTEGER, 0, COMM, ierr)
    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: State norm =", state_norm, "(expected 1.0)"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 5: Expectation Value Calculation
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 5: Expectation value calculation..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    ! Set up a simple test case:
    ! State: uniform superposition |psi> = (1/sqrt(N)) * (1, 1, ..., 1)
    ! Observable: O_i = i / N (sequence of integers divided by system size)
    ! Expected <psi|O|psi> = (1/N) * sum_{i=1}^{N} i/N = (N+1)/(2N)

    if (context_valid .and. local_size > 0) then
        do i = 1, int(local_size)
            state_in(i) = cmplx(1.0_real64 / sqrt(real(system_size, real64)), 0.0_real64, real64)
            obs_in(i) = real(local_offset + i, real64) / real(system_size, real64)
        end do

        call set_state(context_ptr, local_size, state_in, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "set_state returned status ", context_error
        end if
        if (test_passed == 1) then
            call set_observables(context_ptr, local_size, obs_in, context_error)
            if (context_error /= 0) then
                test_passed = 0
                write (error_msg, '(A,I0)') "set_observables returned status ", context_error
            end if
        end if
    end if

    if (context_valid) then
        call get_expectation_value(context_ptr, expectation_value, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "get_expectation_value returned status ", context_error
        end if
    end if

    expected_expval = real(system_size + 1, real64) / (2.0_real64 * real(system_size, real64))

    ! All active ranks receive the same result; rank 0 validates it
    if (comm_rank == 0) then
        if (abs(expectation_value - expected_expval) > 1.0e-6_real64) then
            test_passed = 0
            write (error_msg, '(A,F16.10,A,F16.10)') &
                "Expectation value = ", expectation_value, ", expected ", expected_expval
        end if
    end if

    call MPI_Bcast(test_passed, 1, MPI_INTEGER, 0, COMM, ierr)
    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, '(A,F16.10,A,F16.10,A)') "   PASSED: Expectation value =", expectation_value, &
                " (expected ", expected_expval, ")"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 6: Non-trivial Expectation Value
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 6: Non-trivial expectation value..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    ! State: non-uniform |psi_i> = sqrt(2*i / (N*(N+1)))
    ! Observable: O_i = i / N (sequence of integers divided by system size)
    ! Expected <psi|O|psi> = sum_{i=1}^{N} (2*i/(N*(N+1))) * (i/N)
    !                      = 2/(N^2*(N+1)) * sum(i^2)
    !                      = 2/(N^2*(N+1)) * N*(N+1)*(2N+1)/6
    !                      = (2N+1)/(3N)

    if (context_valid .and. local_size > 0) then
        do i = 1, int(local_size)
            state_in(i) = cmplx(sqrt(2.0_real64 * real(local_offset + i, real64) &
                                     / (real(system_size, real64) * real(system_size + 1, real64))), 0.0_real64, real64)
            obs_in(i) = real(local_offset + i, real64) / real(system_size, real64)
        end do

        call set_state(context_ptr, local_size, state_in, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "set_state returned status ", context_error
        end if
        if (test_passed == 1) then
            call set_observables(context_ptr, local_size, obs_in, context_error)
            if (context_error /= 0) then
                test_passed = 0
                write (error_msg, '(A,I0)') "set_observables returned status ", context_error
            end if
        end if
    end if

    if (context_valid) then
        call get_expectation_value(context_ptr, expectation_value, context_error)
        if (context_error /= 0) then
            test_passed = 0
            write (error_msg, '(A,I0)') "get_expectation_value returned status ", context_error
        end if
    end if

    expected_expval = (2.0_real64 * real(system_size, real64) + 1.0_real64) &
                      / (3.0_real64 * real(system_size, real64))

    ! All active ranks receive the same result; rank 0 validates it
    if (comm_rank == 0) then
        if (abs(expectation_value - expected_expval) > 1.0e-4_real64) then
            test_passed = 0
            write (error_msg, '(A,F16.10,A,F16.10)') &
                "Expectation value = ", expectation_value, ", expected ", expected_expval
        end if
    end if

    call MPI_Bcast(test_passed, 1, MPI_INTEGER, 0, COMM, ierr)
    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, '(A,F16.10,A,F16.10,A)') "   PASSED: Expectation value =", expectation_value, &
                " (expected ", expected_expval, ")"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 7: Context Destroy
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 7: Context destroy..."
    total_tests = total_tests + 1
    test_passed = 1
    error_msg = ""

    if (context_valid) then
        call destroy(context_ptr)
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Context destroyed without errors"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Summary
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) then
        write (*, *) ""
        write (*, *) "========================================"
        write (*, '(A,I0,A,I0,A)') "  Results: ", passed_tests, "/", total_tests, " tests passed"
        write (*, *) "========================================"
    end if

    ! Cleanup
    deallocate (state_in, state_out, obs_in, obs_out)
    call ci%destroy()
    deallocate (ci)

    call MPI_Finalize(ierr)

    ! Exit with error code if any tests failed
    if (passed_tests /= total_tests) then
        call exit(1)
    end if

end program test_wavefront_context
