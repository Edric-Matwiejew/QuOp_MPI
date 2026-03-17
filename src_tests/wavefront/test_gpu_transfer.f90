! test_gpu_transfer.f90
! Unit tests for GPU transfer operations using quop_mpi_layout_t distribution
!
! These tests validate the gpu_transfer module using the same partitioning
! logic as the production wavefront_context code:
!   1. gpu_allgatherv_dtoh - device to host gather operation
!   2. gpu_allscatterv_htod - host to device scatter operation
!
! The tests use MPI_Allgather to build per-rank distribution arrays,
! matching the production pattern in wavefront_context. This ensures the
! test exercises the exact same code paths as production.
!
! To run with different configurations:
!   Default:     mpirun -n 4 ./test_gpu_transfer
!   Staged path: QUOP_RANKS_PER_GPU=2 mpirun -n 4 ./test_gpu_transfer

program test_gpu_transfer
    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_size_t, c_null_ptr
    use mpi
    use hipfort
    use hipfort_check
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use communicators, only: create_nodecomm
    use gpu_transfer, only: gpu_allgatherv_dtoh, gpu_allscatterv_htod
    use comm_info_module, only: quop_mpi_layout_t
    use partitions, only: devcomm_node_layout_from_devcomm, nodecomm_layout_from_devcomm_node
    implicit none

    integer(int32) :: COMM, NODECOMM, DEVCOMM, DEVCOMM_NODE, ierr
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: nodecomm_rank, nodecomm_size
    integer(int32) :: devcomm_node_rank
    type(gpu_topology_t) :: topology

    integer(int32) :: test_passed, global_passed
    integer(int32) :: total_tests, passed_tests
    character(len=256) :: error_msg

    ! GPU memory pointers
    type(c_ptr) :: d_data

    ! Test data
    complex(real64), allocatable, target :: host_send(:), host_recv(:)
    complex(real64), allocatable, target :: expected(:)

    ! Partition info - these match wavefront_context naming
    integer(int64), allocatable :: NODECOMM_counts(:), NODECOMM_displs(:)
    integer(int64), allocatable :: DEVCOMM_NODE_counts(:), DEVCOMM_NODE_displs(:)

    ! Layout object for partition management (matches production pattern)
    type(quop_mpi_layout_t) :: layout

    ! Local sizes and offsets (matching wavefront_context)
    integer(int64) :: NODECOMM_local_i, NODECOMM_local_i_offset
    integer(int64) :: DEVCOMM_local_i, DEVCOMM_local_i_offset
    integer(int64) :: total_size

    integer(c_size_t) :: element_size
    integer(int32) :: i
    integer(int32) :: n_devs
    integer(int32) :: setter_err
    real(real64) :: tolerance

    ! Initialize MPI
    call MPI_Init(ierr)
    COMM = MPI_COMM_WORLD
    call MPI_Comm_rank(COMM, comm_rank, ierr)
    call MPI_Comm_size(COMM, comm_size, ierr)

    total_tests = 0
    passed_tests = 0
    tolerance = 1.0e-10_real64
    error_msg = ""

    if (comm_rank == 0) then
        write (*, *) "========================================"
        write (*, *) " GPU Transfer Module Unit Tests"
        write (*, *) " Using quop_mpi_layout_t distribution pattern"
        write (*, *) " Running with", comm_size, "MPI processes"
        write (*, *) "========================================"
    end if

    ! ========================================================================
    ! Setup: Create communicators and initialize GPU topology
    ! ========================================================================
    call create_NODECOMM(COMM, NODECOMM)
    call MPI_Comm_rank(NODECOMM, nodecomm_rank, ierr)
    call MPI_Comm_size(NODECOMM, nodecomm_size, ierr)

    call init_gpu_topology(NODECOMM, topology, suppress_warnings=.true.)

    ! Set the GPU device
    call hipCheck(hipSetDevice(topology%assigned_device_id))

    ! Create DEVCOMM (ranks that own GPU data)
    if (topology%is_gpu_rank) then
        call MPI_Comm_split(COMM, 1, comm_rank, DEVCOMM, ierr)
    else
        call MPI_Comm_split(COMM, MPI_UNDEFINED, comm_rank, DEVCOMM, ierr)
    end if

    ! Create DEVCOMM_NODE (GPU ranks within this node)
    if (topology%is_gpu_rank) then
        call MPI_Comm_split(NODECOMM, 1, nodecomm_rank, DEVCOMM_NODE, ierr)
        call MPI_Comm_rank(DEVCOMM_NODE, devcomm_node_rank, ierr)
    else
        call MPI_Comm_split(NODECOMM, MPI_UNDEFINED, nodecomm_rank, DEVCOMM_NODE, ierr)
        devcomm_node_rank = -1
    end if

    ! Get number of GPU ranks on this node
    n_devs = topology%devcomm_node_size

    ! Populate layout with communicator and topology info
    call layout%set_NODECOMM(NODECOMM, setter_err)
    call layout%set_topology(topology, setter_err)
    call layout%set_DEVCOMM(DEVCOMM, setter_err)
    call layout%set_DEVCOMM_NODE(DEVCOMM_NODE, setter_err)

    ! Report test configuration
    if (comm_rank == 0) then
        write (*, *) ""
        write (*, *) "Test Configuration:"
        write (*, *) "  ranks_per_gpu =", topology%ranks_per_gpu
        write (*, *) "  n_physical_gpus =", topology%n_physical_gpus
        write (*, *) "  n_devs (GPU ranks on node) =", n_devs
        write (*, *) "  nodecomm_size =", nodecomm_size
        write (*, *) ""
    end if

    ! Element size for complex double
    element_size = int(2 * real64, c_size_t)

    ! ========================================================================
    ! Test 1: Balanced partitioning - Device to Host (gpu_allgatherv_dtoh)
    ! ========================================================================
    call run_dtoh_test("Test 1: Balanced partitioning DtoH", balanced=.true.)

    ! ========================================================================
    ! Test 2: Unbalanced partitioning - Device to Host (gpu_allgatherv_dtoh)
    ! ========================================================================
    call run_dtoh_test("Test 2: Unbalanced partitioning DtoH", balanced=.false.)

    ! ========================================================================
    ! Test 3: Balanced partitioning - Host to Device (gpu_allscatterv_htod)
    ! ========================================================================
    call run_htod_test("Test 3: Balanced partitioning HtoD", balanced=.true.)

    ! ========================================================================
    ! Test 4: Unbalanced partitioning - Host to Device (gpu_allscatterv_htod)
    ! ========================================================================
    call run_htod_test("Test 4: Unbalanced partitioning HtoD", balanced=.false.)

    ! ========================================================================
    ! Test 5: Round-trip test (HtoD then DtoH)
    ! ========================================================================
    call run_roundtrip_test("Test 5: Round-trip (HtoD->DtoH) balanced", balanced=.true.)

    ! ========================================================================
    ! Test 6: Round-trip test unbalanced
    ! ========================================================================
    call run_roundtrip_test("Test 6: Round-trip (HtoD->DtoH) unbalanced", balanced=.false.)

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

    ! Cleanup communicators
    if (DEVCOMM_NODE /= MPI_COMM_NULL) call MPI_Comm_free(DEVCOMM_NODE, ierr)
    if (DEVCOMM /= MPI_COMM_NULL) call MPI_Comm_free(DEVCOMM, ierr)
    call MPI_Comm_free(NODECOMM, ierr)

    call MPI_Finalize(ierr)

    if (passed_tests /= total_tests) then
        call exit(1)
    end if

contains

    !> Setup partition arrays using production partitioning functions
    !> This mimics exactly how wavefront_context computes its partition arrays
    subroutine setup_partitions(balanced)
        logical, intent(in) :: balanced
        integer(int64) :: base_size
        integer(int64) :: DEVCOMM_NODE_total_i, DEVCOMM_NODE_rank_0_offset

        ! Allocate arrays sized by nodecomm_size (same as production code)
        allocate (NODECOMM_counts(nodecomm_size))
        allocate (NODECOMM_displs(nodecomm_size))
        allocate (DEVCOMM_NODE_counts(nodecomm_size))
        allocate (DEVCOMM_NODE_displs(nodecomm_size))

        ! Step 1: Determine how much data each GPU rank owns (simulating FFTW planning)
        ! This is the "DEVCOMM" level partitioning
        if (balanced) then
            base_size = 100_int64
            if (topology%is_gpu_rank) then
                DEVCOMM_local_i = base_size
                ! Global offset for this GPU rank (sum of previous GPU ranks' sizes)
                DEVCOMM_local_i_offset = int(devcomm_node_rank, int64) * base_size
            else
                DEVCOMM_local_i = 0
                DEVCOMM_local_i_offset = 0
            end if
            total_size = base_size * n_devs
        else
            ! Unbalanced: GPU rank i gets (i+1)*50 elements
            if (topology%is_gpu_rank) then
                DEVCOMM_local_i = int((devcomm_node_rank + 1) * 50, int64)
                ! Compute offset as sum of previous GPU ranks
                DEVCOMM_local_i_offset = 0
                do i = 0, devcomm_node_rank - 1
                    DEVCOMM_local_i_offset = DEVCOMM_local_i_offset + int((i + 1) * 50, int64)
                end do
            else
                DEVCOMM_local_i = 0
                DEVCOMM_local_i_offset = 0
            end if
            ! Compute total
            total_size = 0
            do i = 1, n_devs
                total_size = total_size + int(i * 50, int64)
            end do
        end if

        ! Step 2: Use production function to compute node-level layout
        ! This computes how the node's GPU data is distributed
        call DEVCOMM_NODE_layout_from_DEVCOMM(DEVCOMM_local_i, DEVCOMM_local_i_offset, &
                                              DEVCOMM_NODE, DEVCOMM, DEVCOMM_NODE_total_i, DEVCOMM_NODE_rank_0_offset)

        ! Step 3: Use production function to compute NODECOMM layout
        ! This distributes the node's data across all NODECOMM ranks for I/O
        call NODECOMM_layout_from_DEVCOMM_NODE(DEVCOMM_NODE_total_i, DEVCOMM_NODE_rank_0_offset, &
                                               DEVCOMM_NODE, NODECOMM, NODECOMM_local_i, NODECOMM_local_i_offset)

        ! Step 4: Build distribution arrays via MPI_Allgather (production pattern)
        ! This replaces the old counts_and_displs() call. Each rank contributes
        ! its host and device partition info, gathered over NODECOMM so every
        ! rank on the node has the complete picture.
        ! Store partition info in layout for consistency with production code.
        call layout%set_partitioning(NODECOMM_local_i, NODECOMM_local_i_offset, &
                                     DEVCOMM_local_i, DEVCOMM_local_i_offset, setter_err)

        call MPI_Allgather(NODECOMM_local_i, 1, MPI_INTEGER8, &
                           NODECOMM_counts, 1, MPI_INTEGER8, NODECOMM, ierr)
        call MPI_Allgather(NODECOMM_local_i_offset, 1, MPI_INTEGER8, &
                           NODECOMM_displs, 1, MPI_INTEGER8, NODECOMM, ierr)
        call MPI_Allgather(DEVCOMM_local_i, 1, MPI_INTEGER8, &
                           DEVCOMM_NODE_counts, 1, MPI_INTEGER8, NODECOMM, ierr)
        call MPI_Allgather(DEVCOMM_local_i_offset, 1, MPI_INTEGER8, &
                           DEVCOMM_NODE_displs, 1, MPI_INTEGER8, NODECOMM, ierr)

    end subroutine setup_partitions

    !> Run a device-to-host gather test
    subroutine run_dtoh_test(test_name, balanced)
        character(len=*), intent(in) :: test_name
        logical, intent(in) :: balanced

        call MPI_Barrier(COMM, ierr)
        if (comm_rank == 0) write (*, *) test_name, "..."
        total_tests = total_tests + 1
        test_passed = 1
        error_msg = ""

        ! Setup partitioning using production code
        call setup_partitions(balanced)

        ! Allocate device memory (only GPU ranks have data)
        if (topology%is_gpu_rank) then
            call hipCheck(hipMalloc(d_data, DEVCOMM_local_i * element_size))

            ! Initialize device data with known pattern
            ! Value at global position g = (g, -g)
            allocate (host_send(DEVCOMM_local_i))
            do i = 1, int(DEVCOMM_local_i)
                host_send(i) = cmplx(real(DEVCOMM_local_i_offset + i, real64), &
                                     -real(DEVCOMM_local_i_offset + i, real64), real64)
            end do
            call hipCheck(hipMemcpy(d_data, c_loc(host_send), &
                                    DEVCOMM_local_i * element_size, hipMemcpyHostToDevice))
        else
            d_data = c_null_ptr
            allocate (host_send(1))
        end if

        ! Allocate host receive buffer (all NODECOMM ranks get their portion)
        allocate (host_recv(NODECOMM_local_i))
        host_recv = cmplx(0.0_real64, 0.0_real64, real64)

        ! Perform the gather using production-style call
        ! Note: gpu_allgatherv_dtoh expects:
        !   (dev_counts, host_counts, dev_displs, host_displs, ...)
        call gpu_allgatherv_dtoh(DEVCOMM_NODE_counts, &
                                 NODECOMM_counts, &
                                 DEVCOMM_NODE_displs, &
                                 NODECOMM_displs, &
                                 d_data, &
                                 c_loc(host_recv), &
                                 MPI_DOUBLE_COMPLEX, &
                                 NODECOMM)

        ! Verify results - expected values based on global position
        allocate (expected(NODECOMM_local_i))
        do i = 1, int(NODECOMM_local_i)
            expected(i) = cmplx(real(NODECOMM_local_i_offset + i, real64), &
                                -real(NODECOMM_local_i_offset + i, real64), real64)
        end do

        do i = 1, int(NODECOMM_local_i)
            if (abs(host_recv(i) - expected(i)) > tolerance) then
                test_passed = 0
                write (error_msg, '(A,I0,A,2F12.6,A,2F12.6)') &
                    "Mismatch at index ", i, ": got (", &
                    real(host_recv(i)), aimag(host_recv(i)), &
                    "), expected (", real(expected(i)), aimag(expected(i))
                exit
            end if
        end do

        call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
        if (global_passed == 1) passed_tests = passed_tests + 1
        if (comm_rank == 0) then
            if (global_passed == 1) then
                write (*, *) "  PASSED"
            else
                write (*, *) "  FAILED:", trim(error_msg)
            end if
        end if

        ! Cleanup
        if (topology%is_gpu_rank) then
            call hipCheck(hipFree(d_data))
        end if
        deallocate (host_send, host_recv, expected)
        deallocate (NODECOMM_counts, NODECOMM_displs)
        deallocate (DEVCOMM_NODE_counts, DEVCOMM_NODE_displs)

    end subroutine run_dtoh_test

    !> Run a host-to-device scatter test
    subroutine run_htod_test(test_name, balanced)
        character(len=*), intent(in) :: test_name
        logical, intent(in) :: balanced

        real(real64) :: expected_val_re, expected_val_im

        call MPI_Barrier(COMM, ierr)
        if (comm_rank == 0) write (*, *) test_name, "..."
        total_tests = total_tests + 1
        test_passed = 1
        error_msg = ""

        ! Setup partitioning using production code
        call setup_partitions(balanced)

        ! Allocate and initialize host data (all NODECOMM ranks have data)
        ! Value at global position g = (g, -g)
        allocate (host_send(NODECOMM_local_i))
        do i = 1, int(NODECOMM_local_i)
            host_send(i) = cmplx(real(NODECOMM_local_i_offset + i, real64), &
                                 -real(NODECOMM_local_i_offset + i, real64), real64)
        end do

        ! Allocate device memory (only GPU ranks receive)
        if (topology%is_gpu_rank) then
            call hipCheck(hipMalloc(d_data, DEVCOMM_local_i * element_size))
            call hipCheck(hipMemset(d_data, 0, DEVCOMM_local_i * element_size))
        else
            d_data = c_null_ptr
        end if

        ! Perform the scatter using production-style call
        ! Note: gpu_allscatterv_htod expects:
        !   (host_counts, dev_counts, host_displs, dev_displs, ...)
        call gpu_allscatterv_htod(NODECOMM_counts, &
                                  DEVCOMM_NODE_counts, &
                                  NODECOMM_displs, &
                                  DEVCOMM_NODE_displs, &
                                  c_loc(host_send), &
                                  d_data, &
                                  MPI_DOUBLE_COMPLEX, &
                                  NODECOMM)

        ! Synchronize GPU before reading back for verification
        if (topology%is_gpu_rank) then
            call hipCheck(hipDeviceSynchronize())
        end if
        call MPI_Barrier(NODECOMM, ierr)

        ! Verify results on GPU ranks
        if (topology%is_gpu_rank) then
            allocate (host_recv(DEVCOMM_local_i))
            call hipCheck(hipMemcpy(c_loc(host_recv), d_data, &
                                    DEVCOMM_local_i * element_size, hipMemcpyDeviceToHost))

            ! Expected values based on global position
            do i = 1, int(DEVCOMM_local_i)
                expected_val_re = real(DEVCOMM_local_i_offset + i, real64)
                expected_val_im = -real(DEVCOMM_local_i_offset + i, real64)
                if (abs(real(host_recv(i)) - expected_val_re) > tolerance .or. &
                    abs(aimag(host_recv(i)) - expected_val_im) > tolerance) then
                    test_passed = 0
                    write (error_msg, '(A,I0,A,2F12.6,A,2F12.6)') &
                        "Mismatch at index ", i, ": got (", &
                        real(host_recv(i)), aimag(host_recv(i)), &
                        "), expected (", expected_val_re, expected_val_im
                    exit
                end if
            end do
            deallocate (host_recv)
        end if

        call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
        if (global_passed == 1) passed_tests = passed_tests + 1
        if (comm_rank == 0) then
            if (global_passed == 1) then
                write (*, *) "  PASSED"
            else
                write (*, *) "  FAILED:", trim(error_msg)
            end if
        end if

        ! Cleanup
        if (topology%is_gpu_rank) then
            call hipCheck(hipFree(d_data))
        end if
        deallocate (host_send)
        deallocate (NODECOMM_counts, NODECOMM_displs)
        deallocate (DEVCOMM_NODE_counts, DEVCOMM_NODE_displs)

    end subroutine run_htod_test

    !> Run a round-trip test: host->device->host
    subroutine run_roundtrip_test(test_name, balanced)
        character(len=*), intent(in) :: test_name
        logical, intent(in) :: balanced

        complex(real64), allocatable, target :: host_original(:), host_final(:)

        ! Ensure all prior GPU operations complete before starting this test
        if (topology%is_gpu_rank) then
            call hipCheck(hipDeviceSynchronize())
        end if
        call MPI_Barrier(COMM, ierr)
        if (comm_rank == 0) write (*, *) test_name, "..."
        total_tests = total_tests + 1
        test_passed = 1
        error_msg = ""

        ! Setup partitioning using production code
        call setup_partitions(balanced)

        ! Allocate and initialize host data
        allocate (host_original(NODECOMM_local_i))
        do i = 1, int(NODECOMM_local_i)
            host_original(i) = cmplx(real(NODECOMM_local_i_offset + i, real64), &
                                     -real(NODECOMM_local_i_offset + i, real64), real64)
        end do

        ! Allocate device memory
        if (topology%is_gpu_rank) then
            call hipCheck(hipMalloc(d_data, DEVCOMM_local_i * element_size))
            call hipCheck(hipMemset(d_data, 0, DEVCOMM_local_i * element_size))
        else
            d_data = c_null_ptr
        end if

        ! Host to Device
        call gpu_allscatterv_htod(NODECOMM_counts, &
                                  DEVCOMM_NODE_counts, &
                                  NODECOMM_displs, &
                                  DEVCOMM_NODE_displs, &
                                  c_loc(host_original), &
                                  d_data, &
                                  MPI_DOUBLE_COMPLEX, &
                                  NODECOMM)

        ! Synchronize GPU before reading back
        if (topology%is_gpu_rank) then
            call hipCheck(hipDeviceSynchronize())
        end if
        call MPI_Barrier(NODECOMM, ierr)

        ! Allocate final buffer and zero it
        allocate (host_final(NODECOMM_local_i))
        host_final = cmplx(0.0_real64, 0.0_real64, real64)

        ! Device to Host
        call gpu_allgatherv_dtoh(DEVCOMM_NODE_counts, &
                                 NODECOMM_counts, &
                                 DEVCOMM_NODE_displs, &
                                 NODECOMM_displs, &
                                 d_data, &
                                 c_loc(host_final), &
                                 MPI_DOUBLE_COMPLEX, &
                                 NODECOMM)

        ! Verify round-trip: host_final should equal host_original
        do i = 1, int(NODECOMM_local_i)
            if (abs(host_final(i) - host_original(i)) > tolerance) then
                test_passed = 0
                write (*, '(A,I0,A,I0,A,2F12.6,A,2F12.6)') &
                    "Rank ", comm_rank, " mismatch at index ", i, ": got (", &
                    real(host_final(i)), aimag(host_final(i)), &
                    "), expected (", real(host_original(i)), aimag(host_original(i)), ")"
                flush (6)
                exit
            end if
        end do

        call MPI_Allreduce(test_passed, global_passed, 1, MPI_INTEGER, MPI_MIN, COMM, ierr)
        if (global_passed == 1) passed_tests = passed_tests + 1
        if (comm_rank == 0) then
            if (global_passed == 1) then
                write (*, *) "  PASSED"
            else
                write (*, *) "  FAILED:", trim(error_msg)
            end if
        end if

        ! Sync before cleanup to ensure all transfers complete
        if (topology%is_gpu_rank) then
            call hipCheck(hipDeviceSynchronize())
        end if
        call MPI_Barrier(NODECOMM, ierr)

        ! Cleanup
        if (topology%is_gpu_rank) then
            call hipCheck(hipFree(d_data))
            d_data = c_null_ptr
        end if
        deallocate (host_original, host_final)
        deallocate (NODECOMM_counts, NODECOMM_displs)
        deallocate (DEVCOMM_NODE_counts, DEVCOMM_NODE_displs)

    end subroutine run_roundtrip_test

end program test_gpu_transfer
