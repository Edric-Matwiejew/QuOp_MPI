! test_array_transfer.f90
! Tests for array transfer operations using the wavefront communicators
!
! This is an INTEGRATION TEST that validates the complete data transfer
! workflow used in the wavefront backend:
!   1. Communicator setup (NODECOMM, DEVCOMM, DEVCOMM_NODE) using topology
!   2. Partition layout via quop_mpi_layout_t
!   3. Distribution arrays via MPI_Allgather (production pattern)
!   4. Overlap-based Alltoallv scatter/gather operations
!
! The test verifies that data can be distributed from host processes to
! GPU-owning processes and gathered back with correct values.

program test_array_transfer
    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc
    use mpi
    use communicators, only: create_devcomm_with_topology, create_nodecomm, free_communicators
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use partitions, only: devcomm_node_layout_from_devcomm, nodecomm_layout_from_devcomm_node
    use comm_info_module, only: quop_mpi_layout_t
    implicit none

    integer(int32) :: COMM, NODECOMM, DEVCOMM, DEVCOMM_NODE, ierr
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: nodecomm_rank, nodecomm_size
    integer(int32) :: devcomm_rank, devcomm_size
    type(gpu_topology_t) :: topology
    type(quop_mpi_layout_t) :: layout

    integer(int64) :: devcomm_node_local_i, devcomm_node_rank_0_offset

    ! Distribution arrays (production pattern: per-NODECOMM-rank descriptors)
    integer(int64), allocatable :: host_counts_arr(:), host_displs_arr(:)
    integer(int64), allocatable :: dev_counts_arr(:), dev_displs_arr(:)

    ! Test arrays
    real(real64), allocatable, target :: host_array(:), device_array(:), result_array(:)
    ! 64-bit overlap-based counts for Alltoallv
    integer(int64), allocatable :: scounts_int64(:), rcounts_int64(:), sdispls_int64(:), rdispls_int64(:)
    ! 32-bit counts for MPI_Alltoallv
    integer(int32), allocatable :: scounts(:), rcounts(:), sdispls(:), rdispls(:)

    ! Overlap computation temporaries
    integer(int64) :: my_start, my_end, their_start, their_end, overlap

    ! Local partition variables (replacing direct layout% field access)
    integer(int64) :: my_local_i, my_local_i_offset
    integer(int64) :: my_device_local_i, my_device_local_i_offset

    integer(int32) :: test_passed, global_passed, i, j
    integer(int32) :: total_tests, passed_tests
    integer(int32) :: setter_err
    character(len=256) :: error_msg

    ! Initialize MPI
    call MPI_Init(ierr)
    COMM = MPI_COMM_WORLD
    call MPI_Comm_rank(COMM, comm_rank, ierr)
    call MPI_Comm_size(COMM, comm_size, ierr)

    total_tests = 0
    passed_tests = 0

    if (comm_rank == 0) then
        write (*, *) "========================================"
        write (*, *) " Array Transfer Integration Tests"
        write (*, *) " Running with", comm_size, "MPI processes"
        write (*, *) "========================================"
    end if

    ! Setup communicators using topology
    call create_NODECOMM(COMM, NODECOMM)
    call MPI_Comm_rank(NODECOMM, nodecomm_rank, ierr)
    call MPI_Comm_size(NODECOMM, nodecomm_size, ierr)

    call init_gpu_topology(NODECOMM, topology, suppress_warnings=.true.)
    call create_devcomm_with_topology(COMM, NODECOMM, topology, DEVCOMM, DEVCOMM_NODE)

    if (DEVCOMM /= MPI_COMM_NULL) then
        call MPI_Comm_rank(DEVCOMM, devcomm_rank, ierr)
        call MPI_Comm_size(DEVCOMM, devcomm_size, ierr)
    end if

    ! Setup partitioning using quop_mpi_layout_t (production pattern)
    call layout%set_NODECOMM(NODECOMM, setter_err)
    call layout%set_topology(topology, setter_err)
    call layout%set_DEVCOMM(DEVCOMM, setter_err)
    call layout%set_DEVCOMM_NODE(DEVCOMM_NODE, setter_err)

    ! Initialize partition fields via local variables
    my_device_local_i = 0_int64
    my_device_local_i_offset = 0_int64
    devcomm_node_local_i = 0_int64
    devcomm_node_rank_0_offset = 0_int64

    if (DEVCOMM /= MPI_COMM_NULL) then
        my_device_local_i = 10_int64
        my_device_local_i_offset = int(devcomm_rank, int64) * my_device_local_i

        call DEVCOMM_NODE_layout_from_DEVCOMM(my_device_local_i, my_device_local_i_offset, &
                                              DEVCOMM_NODE, DEVCOMM, devcomm_node_local_i, devcomm_node_rank_0_offset)
    end if

    call NODECOMM_layout_from_DEVCOMM_NODE(devcomm_node_local_i, devcomm_node_rank_0_offset, &
                                           DEVCOMM_NODE, NODECOMM, my_local_i, my_local_i_offset)

    call layout%set_partitioning(my_local_i, my_local_i_offset, &
                                 my_device_local_i, my_device_local_i_offset, setter_err)

    ! ========================================================================
    ! Test 1: Build distribution arrays and compute transfer counts
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 1: Build distribution arrays and compute transfer counts..."
    total_tests = total_tests + 1

    test_passed = 1
    ! Allocate distribution arrays (per-NODECOMM-rank descriptors)
    allocate (host_counts_arr(nodecomm_size), host_displs_arr(nodecomm_size))
    allocate (dev_counts_arr(nodecomm_size), dev_displs_arr(nodecomm_size))

    ! Build distribution arrays via MPI_Allgather (production pattern)
    call MPI_Allgather(my_local_i, 1, MPI_INTEGER8, &
                       host_counts_arr, 1, MPI_INTEGER8, NODECOMM, ierr)
    call MPI_Allgather(my_local_i_offset, 1, MPI_INTEGER8, &
                       host_displs_arr, 1, MPI_INTEGER8, NODECOMM, ierr)
    call MPI_Allgather(my_device_local_i, 1, MPI_INTEGER8, &
                       dev_counts_arr, 1, MPI_INTEGER8, NODECOMM, ierr)
    call MPI_Allgather(my_device_local_i_offset, 1, MPI_INTEGER8, &
                       dev_displs_arr, 1, MPI_INTEGER8, NODECOMM, ierr)

    ! Allocate Alltoallv count arrays
    allocate (scounts_int64(nodecomm_size), rcounts_int64(nodecomm_size))
    allocate (sdispls_int64(nodecomm_size), rdispls_int64(nodecomm_size))
    allocate (scounts(nodecomm_size), rcounts(nodecomm_size))
    allocate (sdispls(nodecomm_size), rdispls(nodecomm_size))

    ! Compute overlap-based Alltoallv counts from distribution arrays
    ! scounts(j) = overlap of my host range with rank j's device range
    scounts_int64 = 0
    sdispls_int64 = 0
    my_start = my_local_i_offset
    my_end = my_local_i_offset + my_local_i
    do j = 1, nodecomm_size
        their_start = dev_displs_arr(j)
        their_end = dev_displs_arr(j) + dev_counts_arr(j)
        overlap = max(0_int64, min(my_end, their_end) - max(my_start, their_start))
        scounts_int64(j) = overlap
        sdispls_int64(j) = max(0_int64, max(my_start, their_start) - my_start)
    end do

    ! rcounts(j) = overlap of rank j's host range with my device range
    rcounts_int64 = 0
    rdispls_int64 = 0
    my_start = my_device_local_i_offset
    my_end = my_device_local_i_offset + my_device_local_i
    do j = 1, nodecomm_size
        their_start = host_displs_arr(j)
        their_end = host_displs_arr(j) + host_counts_arr(j)
        overlap = max(0_int64, min(my_end, their_end) - max(my_start, their_start))
        rcounts_int64(j) = overlap
        rdispls_int64(j) = max(0_int64, max(my_start, their_start) - my_start)
    end do

    ! Convert to 32-bit for MPI_Alltoallv
    scounts = int(scounts_int64, int32)
    rcounts = int(rcounts_int64, int32)
    sdispls = int(sdispls_int64, int32)
    rdispls = int(rdispls_int64, int32)

    ! Verify counts are non-negative
    do i = 1, nodecomm_size
        if (scounts(i) < 0 .or. rcounts(i) < 0) then
            test_passed = 0
            error_msg = "Negative count detected"
            exit
        end if
    end do

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Distribution arrays and transfer counts computed"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 2: Host-to-device scatter
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 2: Host-to-device scatter..."
    total_tests = total_tests + 1
    test_passed = 1

    ! Allocate host array with known pattern
    allocate (host_array(my_local_i))
    do i = 1, int(my_local_i)
        host_array(i) = real(my_local_i_offset + i, real64)
    end do

    ! Allocate device array for receiving
    allocate (device_array(my_device_local_i))
    device_array = 0.0_real64

    ! Perform scatter: host -> device using MPI_Alltoallv
    call MPI_Alltoallv(host_array, scounts, sdispls, MPI_DOUBLE_PRECISION, &
                       device_array, rcounts, rdispls, MPI_DOUBLE_PRECISION, &
                       NODECOMM, ierr)

    ! Verify device data
    if (DEVCOMM /= MPI_COMM_NULL) then
        do i = 1, int(my_device_local_i)
            if (abs(device_array(i) - real(my_device_local_i_offset + i, real64)) > 1.0e-10_real64) then
                test_passed = 0
                error_msg = "Device data mismatch after scatter"
                exit
            end if
        end do
    end if

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Scatter verified"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Test 3: Device-to-host gather
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Test 3: Device-to-host gather..."
    total_tests = total_tests + 1
    test_passed = 1

    ! Allocate result array
    allocate (result_array(my_local_i))
    result_array = 0.0_real64

    ! Perform gather: device -> host using MPI_Alltoallv (reverse direction)
    call MPI_Alltoallv(device_array, rcounts, rdispls, MPI_DOUBLE_PRECISION, &
                       result_array, scounts, sdispls, MPI_DOUBLE_PRECISION, &
                       NODECOMM, ierr)

    ! Verify result matches original host data
    do i = 1, int(my_local_i)
        if (abs(result_array(i) - host_array(i)) > 1.0e-10_real64) then
            test_passed = 0
            error_msg = "Round-trip data mismatch"
            exit
        end if
    end do

    call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
    if (global_passed == 1) passed_tests = passed_tests + 1
    if (comm_rank == 0) then
        if (global_passed == 1) then
            write (*, *) "  PASSED: Gather and round-trip verified"
        else
            write (*, *) "  FAILED:", trim(error_msg)
        end if
    end if

    ! ========================================================================
    ! Cleanup
    ! ========================================================================
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) write (*, *) ""
    if (comm_rank == 0) write (*, *) "Cleaning up..."

    if (allocated(host_array)) deallocate (host_array)
    if (allocated(device_array)) deallocate (device_array)
    if (allocated(result_array)) deallocate (result_array)
    if (allocated(host_counts_arr)) deallocate (host_counts_arr)
    if (allocated(host_displs_arr)) deallocate (host_displs_arr)
    if (allocated(dev_counts_arr)) deallocate (dev_counts_arr)
    if (allocated(dev_displs_arr)) deallocate (dev_displs_arr)
    if (allocated(scounts_int64)) deallocate (scounts_int64)
    if (allocated(rcounts_int64)) deallocate (rcounts_int64)
    if (allocated(sdispls_int64)) deallocate (sdispls_int64)
    if (allocated(rdispls_int64)) deallocate (rdispls_int64)
    if (allocated(scounts)) deallocate (scounts)
    if (allocated(rcounts)) deallocate (rcounts)
    if (allocated(sdispls)) deallocate (sdispls)
    if (allocated(rdispls)) deallocate (rdispls)

    call free_communicators(DEVCOMM, NODECOMM, DEVCOMM_NODE)

    ! Final summary
    call MPI_Barrier(COMM, ierr)
    if (comm_rank == 0) then
        write (*, *) ""
        write (*, *) "========================================"
        write (*, '(A,I0,A,I0,A)') "  Results: ", passed_tests, "/", total_tests, " tests passed"
        write (*, *) "========================================"
    end if

    if (passed_tests /= total_tests) then
        call MPI_Abort(COMM, 1, ierr)
    end if

    call MPI_Finalize(ierr)

end program test_array_transfer
