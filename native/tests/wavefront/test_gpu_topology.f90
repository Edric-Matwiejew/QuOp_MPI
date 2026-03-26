! test_gpu_topology.f90
! Unit tests for GPU topology detection and configuration
!
! These are UNIT TESTS that validate the gpu_topology module:
!   1. init_gpu_topology - topology detection and configuration
!   2. Environment variable handling (QUOP_RANKS_PER_GPU, QUOP_GPU_BINDING_MODE)
!   3. PCI bus ID gathering and GPU index assignment
!   4. DEVCOMM membership determination
!
! To run with different configurations:
!   Standard:  mpirun -n 4 ./test_gpu_topology
!   Testing multi-rank-per-GPU: QUOP_RANKS_PER_GPU=2 mpirun -n 4 ./test_gpu_topology
!   Force sequential mode: QUOP_GPU_BINDING_MODE=sequential mpirun -n 4 ./test_gpu_topology
!   Debug output: QUOP_DEBUG_BACKEND=1 mpirun -n 4 ./test_gpu_topology

program test_gpu_topology
    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use mpi
    use hipfort
    use hipfort_check
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use communicators, only: create_nodecomm
    implicit none

    integer(int32) :: COMM, NODECOMM, ierr
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: nodecomm_rank, nodecomm_size
    type(gpu_topology_t) :: topology

    integer(int32) :: test_passed, global_passed
    integer(int32) :: total_tests, passed_tests
    character(len=256) :: error_msg
    character(len=64) :: env_val
    integer(int32) :: expected_ranks_per_gpu
    integer(int32) :: actual_device_count
    integer(int32) :: gpu_rank_count, total_gpu_ranks
    integer(int32) :: i

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
        write (*, *) " GPU Topology Module Unit Tests"
        write (*, *) " Running with", comm_size, "MPI processes"
        write (*, *) "========================================"
    end if

    ! ========================================================================
    ! Test 1: NODECOMM creation (prerequisite for topology)
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
    ! Test 2: init_gpu_topology basic functionality
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 2: init_gpu_topology basic functionality..."
    total_tests = total_tests + 1
    test_passed = 1

    call init_gpu_topology(NODECOMM, topology)

    ! Verify topology was populated
    if (topology%node_size /= nodecomm_size) then
        test_passed = 0
        write (error_msg, '(A,I4,A,I4)') "node_size mismatch: got ", topology%node_size, &
            " expected ", nodecomm_size
    else if (topology%node_rank /= nodecomm_rank) then
        test_passed = 0
        write (error_msg, '(A,I4,A,I4)') "node_rank mismatch: got ", topology%node_rank, &
            " expected ", nodecomm_rank
    else if (topology%visible_device_count < 1) then
        test_passed = 0
        error_msg = "visible_device_count < 1"
    else if (topology%n_physical_gpus < 1) then
        test_passed = 0
        error_msg = "n_physical_gpus < 1"
    else if (.not. allocated(topology%visible_gpus)) then
        test_passed = 0
        error_msg = "visible_gpus not allocated"
    else if (size(topology%visible_gpus) /= topology%visible_device_count) then
        test_passed = 0
        write (error_msg, '(A,I4,A,I4)') "visible_gpus size mismatch: got ", &
            size(topology%visible_gpus), " expected ", &
            topology%visible_device_count
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: topology initialized"
            write (*, *) "    visible_device_count =", topology%visible_device_count
            write (*, *) "    n_physical_gpus =", topology%n_physical_gpus
            write (*, *) "    binding_mode =", trim(topology%binding_mode)
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 3: Environment variable handling (QUOP_RANKS_PER_GPU)
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 3: QUOP_RANKS_PER_GPU handling..."
    total_tests = total_tests + 1
    test_passed = 1

    call get_environment_variable('QUOP_RANKS_PER_GPU', env_val)
    if (len_trim(env_val) > 0) then
        read (env_val, *) expected_ranks_per_gpu
    else
        expected_ranks_per_gpu = 1
    end if

    if (topology%ranks_per_gpu /= expected_ranks_per_gpu) then
        test_passed = 0
        write (error_msg, '(A,I4,A,I4)') "ranks_per_gpu mismatch: got ", topology%ranks_per_gpu, &
            " expected ", expected_ranks_per_gpu
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: ranks_per_gpu =", topology%ranks_per_gpu
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 4: GPU index assignment consistency
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 4: GPU index assignment consistency..."
    total_tests = total_tests + 1
    test_passed = 1

    ! my_gpu_index should be in valid range
    if (topology%my_gpu_index < 0 .or. topology%my_gpu_index >= topology%n_physical_gpus) then
        test_passed = 0
        write (error_msg, '(A,I4,A,I4)') "my_gpu_index out of range: ", topology%my_gpu_index, &
            " n_physical_gpus: ", topology%n_physical_gpus
    end if

    ! rank_within_gpu should be non-negative
    if (topology%rank_within_gpu < 0) then
        test_passed = 0
        write (error_msg, '(A,I4)') "rank_within_gpu negative: ", topology%rank_within_gpu
    end if

    if (test_passed == 1) then
        do i = 1, topology%visible_device_count
            if (topology%visible_gpus(i)%device_id /= i - 1) then
                test_passed = 0
                write (error_msg, '(A,I4,A,I4)') "visible_gpus device_id mismatch at slot ", i, &
                    ": got ", topology%visible_gpus(i)%device_id
                exit
            end if
            if (len_trim(topology%visible_gpus(i)%pci_bus_id) == 0) then
                test_passed = 0
                write (error_msg, '(A,I4)') "visible_gpus empty pci_bus_id at slot ", i
                exit
            end if
        end do
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: GPU indices valid"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 5: DEVCOMM membership consistency
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 5: DEVCOMM membership consistency..."
    total_tests = total_tests + 1
    test_passed = 1

    ! Count how many GPU ranks we have
    if (topology%is_gpu_rank) then
        gpu_rank_count = 1
    else
        gpu_rank_count = 0
    end if
    call MPI_Allreduce(gpu_rank_count, total_gpu_ranks, 1, MPI_INTEGER, MPI_SUM, NODECOMM, ierr)

    ! devcomm_node_size should match total GPU ranks on node
    if (topology%devcomm_node_size /= total_gpu_ranks) then
        test_passed = 0
        write (error_msg, '(A,I4,A,I4)') "devcomm_node_size mismatch: ", topology%devcomm_node_size, &
            " vs counted: ", total_gpu_ranks
    end if

    ! GPU ranks should have valid assigned_device_id
    if (topology%is_gpu_rank) then
        if (topology%assigned_device_id < 0 .or. &
            topology%assigned_device_id >= topology%visible_device_count) then
            test_passed = 0
            write (error_msg, '(A,I4,A,I4)') "assigned_device_id out of range: ", &
                topology%assigned_device_id, &
                " visible: ", topology%visible_device_count
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: DEVCOMM membership consistent"
            write (*, *) "    devcomm_node_size =", topology%devcomm_node_size
            write (*, *) "    GPU ranks on node =", total_gpu_ranks
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 6: hipSetDevice works with assigned_device_id
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 6: hipSetDevice with assigned_device_id..."
    total_tests = total_tests + 1
    test_passed = 1

    if (topology%is_gpu_rank) then
        ! Try setting the device - this should succeed
        call hipCheck(hipSetDevice(topology%assigned_device_id))

        ! Verify we're on the correct device
        call hipCheck(hipGetDevice(actual_device_count))
        if (actual_device_count /= topology%assigned_device_id) then
            test_passed = 0
            write (error_msg, '(A,I4,A,I4)') "Device mismatch after hipSetDevice: got ", &
                actual_device_count, &
                " expected ", topology%assigned_device_id
        end if
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: hipSetDevice works with assigned_device_id"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 7: Detailed topology report (informational)
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 7: Detailed topology report..."
    total_tests = total_tests + 1
    test_passed = 1

    ! Print per-rank topology info
    do i = 0, comm_size - 1
        call MPI_Barrier(COMM, ierr)
        if (comm_rank == i) then
            write (*, '(A,I3,A,I2,A,I2,A,I2,A,I2,A,I2,A,L1)') &
                "  Rank ", comm_rank, &
                ": node_rank=", topology%node_rank, &
                ", gpu_idx=", topology%my_gpu_index, &
                ", rank_in_gpu=", topology%rank_within_gpu, &
                ", dev_id=", topology%assigned_device_id, &
                ", visible=", topology%visible_device_count, &
                ", is_gpu=", topology%is_gpu_rank
        end if
    end do
    call MPI_Barrier(COMM, ierr)

    ! This test always passes - it's informational
    passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        write (*, *) "  PASSED: Topology report complete"
    end if

    ! ========================================================================
    ! Summary
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) then
        write (*, *) ""
        write (*, *) "========================================"
        write (*, *) " Test Summary"
        write (*, *) "========================================"
        write (*, '(A,I2,A,I2,A)') " Passed: ", passed_tests, " / ", total_tests, " tests"
        if (passed_tests == total_tests) then
            write (*, *) " STATUS: ALL TESTS PASSED"
        else
            write (*, *) " STATUS: SOME TESTS FAILED"
        end if
        write (*, *) "========================================"
    end if

    ! Cleanup
    if (NODECOMM /= MPI_COMM_NULL) then
        call MPI_Comm_free(NODECOMM, ierr)
    end if

    call MPI_Finalize(ierr)

    ! Exit with error code if tests failed
    if (passed_tests /= total_tests) then
        call exit(1)
    end if

end program test_gpu_topology
