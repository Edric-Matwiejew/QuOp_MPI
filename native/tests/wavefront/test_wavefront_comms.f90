! test_wavefront_comms.f90
! Unit tests for wavefront communicator creation and partitioning
!
! These are UNIT TESTS that validate individual module functions:
!   1. create_NODECOMM - shared memory communicator split
!   2. init_gpu_topology - GPU topology detection
!   3. create_devcomm_with_topology - GPU-assigned communicators
!   4. DEVCOMM_NODE_layout_from_DEVCOMM - partition calculation
!   5. NODECOMM_layout_from_DEVCOMM_NODE - distribution layout

program test_wavefront_comms
    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use mpi
    use communicators, only: create_devcomm_with_topology, create_nodecomm, free_communicators
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use partitions, only: devcomm_node_layout_from_devcomm, nodecomm_layout_from_devcomm_node
    implicit none

    integer(int32) :: COMM, NODECOMM, DEVCOMM, DEVCOMM_NODE, ierr
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: nodecomm_rank, nodecomm_size
    integer(int32) :: devcomm_rank, devcomm_size
    integer(int32) :: devcomm_node_rank, devcomm_node_size
    type(gpu_topology_t) :: topology

    integer(int64) :: devcomm_local_i, devcomm_local_i_offset
    integer(int64) :: devcomm_node_local_i, devcomm_node_rank_0_offset
    integer(int64) :: nodecomm_local_i, nodecomm_local_i_offset

    integer(int32) :: test_passed, global_passed
    integer(int32) :: total_tests, passed_tests
    character(len=256) :: error_msg

    ! Initialize MPI
    call MPI_Init(ierr)
    COMM = MPI_COMM_WORLD
    call MPI_Comm_rank(COMM, comm_rank, ierr)
    call MPI_Comm_size(COMM, comm_size, ierr)

    total_tests = 0
    passed_tests = 0
    error_msg = ""

    if (comm_rank == 0) then
        write (*, *) "========================================"
        write (*, *) " Wavefront Communicator Unit Tests"
        write (*, *) " Running with", comm_size, "MPI processes"
        write (*, *) "========================================"
    end if

    ! ========================================================================
    ! Test 1: NODECOMM creation
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 1: NODECOMM creation..."
    total_tests = total_tests + 1
    test_passed = 1

    call create_NODECOMM(COMM, NODECOMM)

    if (NODECOMM == MPI_COMM_NULL) then
        test_passed = 0
        error_msg = "NODECOMM is MPI_COMM_NULL"
    else
        call MPI_Comm_rank(NODECOMM, nodecomm_rank, ierr)
        call MPI_Comm_size(NODECOMM, nodecomm_size, ierr)

        ! All processes should be in same NODECOMM on single node
        if (nodecomm_size < 1) then
            test_passed = 0
            error_msg = "NODECOMM size is invalid"
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: NODECOMM created, size =", nodecomm_size
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 2: GPU topology detection
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 2: GPU topology detection..."
    total_tests = total_tests + 1
    test_passed = 1

    call init_gpu_topology(NODECOMM, topology, suppress_warnings=.true.)

    if (topology%visible_device_count < 1) then
        test_passed = 0
        error_msg = "No GPUs detected"
    else if (topology%n_physical_gpus < 1) then
        test_passed = 0
        error_msg = "No physical GPUs detected"
    else if (topology%devcomm_node_size < 1) then
        test_passed = 0
        error_msg = "No GPU ranks on node"
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Detected", topology%n_physical_gpus, "GPU(s),", &
                topology%devcomm_node_size, "GPU rank(s)"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 3: DEVCOMM and DEVCOMM_NODE creation with topology
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 3: DEVCOMM/DEVCOMM_NODE creation..."
    total_tests = total_tests + 1
    test_passed = 1

    call create_devcomm_with_topology(COMM, NODECOMM, topology, DEVCOMM, DEVCOMM_NODE)

    if (topology%is_gpu_rank) then
        ! This process should be in DEVCOMM
        if (DEVCOMM == MPI_COMM_NULL) then
            test_passed = 0
            error_msg = "GPU rank should be in DEVCOMM but is not"
        else
            call MPI_Comm_rank(DEVCOMM, devcomm_rank, ierr)
            call MPI_Comm_size(DEVCOMM, devcomm_size, ierr)
        end if
        if (DEVCOMM_NODE == MPI_COMM_NULL) then
            test_passed = 0
            error_msg = "GPU rank should be in DEVCOMM_NODE but is not"
        else
            call MPI_Comm_rank(DEVCOMM_NODE, devcomm_node_rank, ierr)
            call MPI_Comm_size(DEVCOMM_NODE, devcomm_node_size, ierr)
        end if
    else
        ! This process should NOT be in DEVCOMM
        if (DEVCOMM /= MPI_COMM_NULL) then
            test_passed = 0
            error_msg = "Non-GPU rank should not be in DEVCOMM but is"
        end if
        if (DEVCOMM_NODE /= MPI_COMM_NULL) then
            test_passed = 0
            error_msg = "Non-GPU rank should not be in DEVCOMM_NODE but is"
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: DEVCOMM/DEVCOMM_NODE created correctly"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 4: Partition calculations
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 4: Partition calculations..."
    total_tests = total_tests + 1
    test_passed = 1

    ! Initialize partition variables
    devcomm_node_local_i = 0
    devcomm_node_rank_0_offset = 0

    if (DEVCOMM /= MPI_COMM_NULL) then
        ! Set up test partitioning: each GPU process owns 10 elements
        devcomm_local_i = 10_int64
        devcomm_local_i_offset = int(devcomm_rank, int64) * devcomm_local_i

        call DEVCOMM_NODE_layout_from_DEVCOMM(devcomm_local_i, devcomm_local_i_offset, &
                                              DEVCOMM_NODE, DEVCOMM, devcomm_node_local_i, devcomm_node_rank_0_offset)

        ! DEVCOMM_NODE_local_i should be sum of all local_i in DEVCOMM_NODE
        if (devcomm_node_local_i /= int(devcomm_node_size, int64) * 10_int64) then
            test_passed = 0
            error_msg = "DEVCOMM_NODE_local_i calculation incorrect"
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Partition calculations correct"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 5: NODECOMM layout distribution
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 5: NODECOMM layout distribution..."
    total_tests = total_tests + 1
    test_passed = 1

    call NODECOMM_layout_from_DEVCOMM_NODE(devcomm_node_local_i, devcomm_node_rank_0_offset, &
                                           DEVCOMM_NODE, NODECOMM, nodecomm_local_i, nodecomm_local_i_offset)

    ! Each NODECOMM process should get some elements (or zero if no GPUs)
    if (nodecomm_local_i < 0) then
        test_passed = 0
        error_msg = "NODECOMM_local_i is negative"
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: NODECOMM layout distribution correct"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Cleanup
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Cleaning up communicators..."

    call free_communicators(DEVCOMM, NODECOMM, DEVCOMM_NODE)

    ! Final summary
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) then
        write (*, *) ""
        write (*, *) "========================================"
        write (*, '(A,I0,A,I0,A)') "  Results: ", passed_tests, "/", total_tests, " tests passed"
        write (*, *) "========================================"
    end if

    ! Return non-zero exit code if any tests failed
    if (passed_tests /= total_tests) then
        call MPI_Abort(COMM, 1, ierr)
    end if

    call MPI_Finalize(ierr)

end program test_wavefront_comms
