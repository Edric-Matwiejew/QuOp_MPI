module partitions
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64
    use mpi
    implicit none

    private

    public :: DEVCOMM_NODE_layout_from_DEVCOMM, NODECOMM_layout_from_DEVCOMM_NODE

contains

    ! Calculate DEVCOMM_NODE_local_i and DEVCOMM_NODE_rank_0_offset
    subroutine DEVCOMM_NODE_layout_from_DEVCOMM(devcomm_local_i, devcomm_local_i_offset, &
                                                DEVCOMM_NODE, DEVCOMM, DEVCOMM_NODE_local_i, DEVCOMM_NODE_rank_0_offset)

        integer(int64), intent(in) :: devcomm_local_i, devcomm_local_i_offset
        integer(int32), intent(in) :: DEVCOMM_NODE, DEVCOMM
        integer(int64), intent(out) :: DEVCOMM_NODE_local_i, DEVCOMM_NODE_rank_0_offset
        integer(int64), allocatable :: all_devcomm_local_i(:), all_devcomm_local_i_offset(:)
        integer(int32) :: ierr, rank_DEVCOMM_NODE, size_DEVCOMM_NODE, rank_DEVCOMM, size_DEVCOMM
        integer(int32) :: i

        ! Initialize outputs to 0 for non-GPU ranks (will be overwritten for GPU ranks)
        DEVCOMM_NODE_local_i = 0
        DEVCOMM_NODE_rank_0_offset = 0

        if (DEVCOMM_NODE /= MPI_COMM_NULL) then

            ! Get the rank and size in the DEVCOMM and DEVCOMM_NODE communicators
            call MPI_Comm_rank(DEVCOMM_NODE, rank_DEVCOMM_NODE, ierr)
            call MPI_Comm_size(DEVCOMM_NODE, size_DEVCOMM_NODE, ierr)
            call MPI_Comm_rank(DEVCOMM, rank_DEVCOMM, ierr)
            call MPI_Comm_size(DEVCOMM, size_DEVCOMM, ierr)

            ! Allocate arrays to gather all devcomm_local_i and devcomm_local_i_offset in DEVCOMM_NODE
            allocate (all_devcomm_local_i(size_DEVCOMM_NODE))
            allocate (all_devcomm_local_i_offset(size_DEVCOMM_NODE))

            ! Gather all devcomm_local_i and devcomm_local_i_offset in DEVCOMM_NODE
            call MPI_Allgather(devcomm_local_i, 1, MPI_INTEGER8, all_devcomm_local_i, 1, &
                               MPI_INTEGER8, DEVCOMM_NODE, ierr)
            call MPI_Allgather(devcomm_local_i_offset, 1, MPI_INTEGER8, &
                               all_devcomm_local_i_offset, 1, MPI_INTEGER8, DEVCOMM_NODE, ierr)

            ! Calculate DEVCOMM_NODE_local_i as the sum of all devcomm_local_i in DEVCOMM_NODE
            DEVCOMM_NODE_local_i = 0
            do i = 1, size_DEVCOMM_NODE
                DEVCOMM_NODE_local_i = DEVCOMM_NODE_local_i + all_devcomm_local_i(i)
            end do

            ! Calculate DEVCOMM_NODE_rank_0_offset as the minimum of all devcomm_local_i_offset in DEVCOMM_NODE
            DEVCOMM_NODE_rank_0_offset = minval(all_devcomm_local_i_offset)

        end if

    end subroutine DEVCOMM_NODE_layout_from_DEVCOMM

    ! Compute a partitioning of DEVCOMM_NODE_local_i elements over all processes in
    ! NODECOMM. When negotiate-time device ownership metadata is provided, ranks
    ! with device-resident data are seeded first and helper ranks on the same CPU
    ! NUMA domain are preferred before falling back to plain NODECOMM-rank order.
    ! Otherwise, the host partition falls back to a simple block distribution.
    subroutine NODECOMM_layout_from_DEVCOMM_NODE(DEVCOMM_NODE_local_i, DEVCOMM_NODE_rank_0_offset, &
                                                 DEVCOMM_NODE, NODECOMM, NODECOMM_local_i, NODECOMM_local_i_offset, &
                                                 SUBCOMM, node_id, n_nodes, active_device_local_i, cpu_numa_node)
        integer(int64), intent(inout) :: DEVCOMM_NODE_local_i, DEVCOMM_NODE_rank_0_offset
        integer(int32), intent(in) :: DEVCOMM_NODE, NODECOMM
        integer(int64), intent(out) :: NODECOMM_local_i, NODECOMM_local_i_offset
        integer(int32), intent(in), optional :: SUBCOMM, node_id, n_nodes
        integer(int64), intent(in), optional :: active_device_local_i
        integer(int32), intent(in), optional :: cpu_numa_node
        integer(int64), allocatable :: recvcounts(:), displs(:)
        integer(int32) :: ierr, rank_NODECOMM, size_NODECOMM, rank_DEVCOMM_NODE
        integer(int64) :: rem, base, i
        integer(int32) :: gpu_root_in_nodecomm, active_gpu_flag_local, my_cpu_numa_node
        integer(int32), allocatable :: active_gpu_flags(:), cpu_numa_nodes(:), assignment_order(:)
        integer(int64) :: broadcast_data(2)
        integer(int64), allocatable :: node_totals(:), node_totals_local(:)
        logical :: have_global_node_offsets, use_gpu_biased_layout

        have_global_node_offsets = present(SUBCOMM) .and. present(node_id) .and. present(n_nodes)
        use_gpu_biased_layout = present(active_device_local_i)

        ! Get the rank and size in the NODECOMM communicator
        call MPI_Comm_rank(NODECOMM, rank_NODECOMM, ierr)
        call MPI_Comm_size(NODECOMM, size_NODECOMM, ierr)

        ! Determine which NODECOMM rank is the GPU root (DEVCOMM_NODE rank 0)
        ! GPU ranks set their DEVCOMM_NODE rank, non-GPU ranks set a large number
        if (DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Comm_rank(DEVCOMM_NODE, rank_DEVCOMM_NODE, ierr)
            if (rank_DEVCOMM_NODE == 0) then
                gpu_root_in_nodecomm = rank_NODECOMM
            else
                gpu_root_in_nodecomm = size_NODECOMM ! Larger than any valid rank
            end if
        else
            gpu_root_in_nodecomm = size_NODECOMM ! Non-GPU ranks
        end if

        ! Find the minimum (the NODECOMM rank that is DEVCOMM_NODE rank 0)
        call MPI_Allreduce(MPI_IN_PLACE, gpu_root_in_nodecomm, 1, MPI_INTEGER, MPI_MIN, NODECOMM, ierr)

        ! Broadcast from the GPU root (DEVCOMM_NODE rank 0) to all NODECOMM ranks.
        ! Some small-system layouts can leave an entire node with no active GPU
        ! ranks after DEVCOMM is rebuilt. In that case there is no valid
        ! DEVCOMM_NODE root to broadcast from, so the node-local totals stay at 0.
        if (gpu_root_in_nodecomm < size_NODECOMM) then
            if (have_global_node_offsets) then
                call MPI_Bcast(DEVCOMM_NODE_local_i, 1, MPI_INTEGER8, gpu_root_in_nodecomm, NODECOMM, ierr)
            else
                if (rank_NODECOMM == gpu_root_in_nodecomm) then
                    broadcast_data(1) = DEVCOMM_NODE_local_i
                    broadcast_data(2) = DEVCOMM_NODE_rank_0_offset
                end if
                call MPI_Bcast(broadcast_data, 2, MPI_INTEGER8, gpu_root_in_nodecomm, NODECOMM, ierr)
                DEVCOMM_NODE_local_i = broadcast_data(1)
                DEVCOMM_NODE_rank_0_offset = broadcast_data(2)
            end if
        else
            DEVCOMM_NODE_local_i = 0_int64
            DEVCOMM_NODE_rank_0_offset = 0_int64
        end if

        ! If the caller can provide SUBCOMM + topology node indexing, derive a
        ! globally consistent node offset even when this node has no active GPU
        ! ranks. This keeps zero-data nodes valid in the host partition.
        if (have_global_node_offsets) then
            allocate (node_totals_local(n_nodes), node_totals(n_nodes))
            node_totals_local = 0_int64
            if (rank_NODECOMM == 0) then
                node_totals_local(node_id + 1) = DEVCOMM_NODE_local_i
            end if
            call MPI_Allreduce(node_totals_local, node_totals, n_nodes, MPI_INTEGER8, MPI_SUM, SUBCOMM, ierr)

            DEVCOMM_NODE_rank_0_offset = 0_int64
            do i = 1, node_id
                DEVCOMM_NODE_rank_0_offset = DEVCOMM_NODE_rank_0_offset + node_totals(i)
            end do

            deallocate (node_totals_local, node_totals)
        end if

        ! Allocate arrays for the receive counts and displacements
        allocate (recvcounts(size_NODECOMM))
        allocate (displs(size_NODECOMM))

        ! Calculate the base number of elements and the remainder
        base = DEVCOMM_NODE_local_i / int(size_NODECOMM, int64)
        rem = DEVCOMM_NODE_local_i - base * int(size_NODECOMM, int64)

        recvcounts = 0_int64
        displs = 0_int64

        if (use_gpu_biased_layout) then
            active_gpu_flag_local = merge(1_int32, 0_int32, active_device_local_i > 0_int64)
            my_cpu_numa_node = -1
            if (present(cpu_numa_node)) my_cpu_numa_node = cpu_numa_node

            allocate (active_gpu_flags(size_NODECOMM), cpu_numa_nodes(size_NODECOMM), assignment_order(size_NODECOMM))
            call MPI_Allgather(active_gpu_flag_local, 1, MPI_INTEGER, &
                               active_gpu_flags, 1, MPI_INTEGER, NODECOMM, ierr)
            call MPI_Allgather(my_cpu_numa_node, 1, MPI_INTEGER, &
                               cpu_numa_nodes, 1, MPI_INTEGER, NODECOMM, ierr)

            if (sum(active_gpu_flags) > 0 .and. DEVCOMM_NODE_local_i > 0_int64) then
                call build_gpu_biased_assignment(active_gpu_flags, cpu_numa_nodes, assignment_order)
                do i = 1, size_NODECOMM
                    if (i <= rem) then
                        recvcounts(assignment_order(i) + 1) = base + 1
                    else
                        recvcounts(assignment_order(i) + 1) = base
                    end if
                end do
            else
                use_gpu_biased_layout = .false.
            end if
        end if

        if (.not. use_gpu_biased_layout) then
            do i = 0, size_NODECOMM - 1
                if (i < rem) then
                    recvcounts(i + 1) = base + 1
                else
                    recvcounts(i + 1) = base
                end if
            end do
        end if

        do i = 2, size_NODECOMM
            displs(i) = displs(i - 1) + recvcounts(i - 1)
        end do

        ! Get the local number of elements and the local offset
        NODECOMM_local_i = recvcounts(rank_NODECOMM + 1)
        NODECOMM_local_i_offset = displs(rank_NODECOMM + 1)

    end subroutine NODECOMM_layout_from_DEVCOMM_NODE

    subroutine build_gpu_biased_assignment(active_gpu_flags, cpu_numa_nodes, assignment_order)
        integer(int32), intent(in) :: active_gpu_flags(:), cpu_numa_nodes(:)
        integer(int32), intent(out) :: assignment_order(:)

        logical, allocatable :: assigned(:)
        logical :: progress
        integer(int32) :: n_ranks, i, order_len, candidate_rank

        n_ranks = size(active_gpu_flags)
        assignment_order = -1
        allocate (assigned(n_ranks))
        assigned = .false.
        order_len = 0

        ! Active device-owning ranks must receive host data first.
        do i = 1, n_ranks
            if (active_gpu_flags(i) /= 0) then
                order_len = order_len + 1
                assignment_order(order_len) = i - 1
                assigned(i) = .true.
            end if
        end do

        ! Prefer helper ranks on the same CPU NUMA domain, searching forward
        ! from each active GPU rank and wrapping around NODECOMM if needed.
        do
            progress = .false.
            do i = 1, n_ranks
                if (active_gpu_flags(i) == 0) cycle
                candidate_rank = next_unassigned_rank(i - 1, assigned, cpu_numa_nodes, &
                                                      cpu_numa_nodes(i), same_numa_only=.true.)
                if (candidate_rank >= 0) then
                    order_len = order_len + 1
                    assignment_order(order_len) = candidate_rank
                    assigned(candidate_rank + 1) = .true.
                    progress = .true.
                end if
            end do
            if (.not. progress) exit
        end do

        ! Fall back to the next highest unassigned NODECOMM rank if a NUMA-local
        ! helper is not available.
        do
            progress = .false.
            do i = 1, n_ranks
                if (active_gpu_flags(i) == 0) cycle
                candidate_rank = next_unassigned_rank(i - 1, assigned, cpu_numa_nodes, &
                                                      cpu_numa_nodes(i), same_numa_only=.false.)
                if (candidate_rank >= 0) then
                    order_len = order_len + 1
                    assignment_order(order_len) = candidate_rank
                    assigned(candidate_rank + 1) = .true.
                    progress = .true.
                end if
            end do
            if (.not. progress) exit
        end do

        do i = 1, n_ranks
            if (.not. assigned(i)) then
                order_len = order_len + 1
                assignment_order(order_len) = i - 1
            end if
        end do
    end subroutine build_gpu_biased_assignment

    integer(int32) function next_unassigned_rank(seed_rank, assigned, cpu_numa_nodes, &
                                                 seed_cpu_numa, same_numa_only)
        integer(int32), intent(in) :: seed_rank, cpu_numa_nodes(:), seed_cpu_numa
        logical, intent(in) :: assigned(:), same_numa_only

        integer(int32) :: n_ranks, step, candidate_idx

        n_ranks = size(assigned)
        next_unassigned_rank = -1

        do step = 1, n_ranks - 1
            candidate_idx = mod(seed_rank + step, n_ranks) + 1
            if (assigned(candidate_idx)) cycle
            if (same_numa_only) then
                if (seed_cpu_numa < 0) cycle
                if (cpu_numa_nodes(candidate_idx) /= seed_cpu_numa) cycle
            end if
            next_unassigned_rank = candidate_idx - 1
            return
        end do
    end function next_unassigned_rank

end module partitions
