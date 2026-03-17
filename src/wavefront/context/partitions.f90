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

    ! Compute a partitioning of DEVCOMM_NODE_local_i elements over all processes in NODECOMM
    ! Note: DEVCOMM_NODE_local_i and DEVCOMM_NODE_rank_0_offset must be valid on the GPU rank
    ! that is rank 0 in NODECOMM, OR they should be broadcast from a GPU rank first.
    ! The caller should ensure this.
    subroutine NODECOMM_layout_from_DEVCOMM_NODE(DEVCOMM_NODE_local_i, DEVCOMM_NODE_rank_0_offset, &
                                                 DEVCOMM_NODE, NODECOMM, NODECOMM_local_i, NODECOMM_local_i_offset)
        integer(int64), intent(inout) :: DEVCOMM_NODE_local_i, DEVCOMM_NODE_rank_0_offset
        integer(int32), intent(in) :: DEVCOMM_NODE, NODECOMM
        integer(int64), intent(out) :: NODECOMM_local_i, NODECOMM_local_i_offset
        integer(int64), allocatable :: recvcounts(:), displs(:)
        integer(int32) :: ierr, rank_NODECOMM, size_NODECOMM, rank_DEVCOMM_NODE
        integer(int64) :: rem, base, i
        integer(int32) :: gpu_root_in_nodecomm
        integer(int64) :: broadcast_data(2)

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

        ! Broadcast from the GPU root (DEVCOMM_NODE rank 0) to all NODECOMM ranks
        if (rank_NODECOMM == gpu_root_in_nodecomm) then
            broadcast_data(1) = DEVCOMM_NODE_local_i
            broadcast_data(2) = DEVCOMM_NODE_rank_0_offset
        end if
        call MPI_Bcast(broadcast_data, 2, MPI_INTEGER8, gpu_root_in_nodecomm, NODECOMM, ierr)
        DEVCOMM_NODE_local_i = broadcast_data(1)
        DEVCOMM_NODE_rank_0_offset = broadcast_data(2)

        ! Allocate arrays for the receive counts and displacements
        allocate (recvcounts(size_NODECOMM))
        allocate (displs(size_NODECOMM))

        ! Calculate the base number of elements and the remainder
        base = DEVCOMM_NODE_local_i / int(size_NODECOMM, int64)
        rem = DEVCOMM_NODE_local_i - base * int(size_NODECOMM, int64)

        ! Set the receive counts and displacements
        do i = 0, size_NODECOMM - 1
            if (i < rem) then
                recvcounts(i + 1) = base + 1
            else
                recvcounts(i + 1) = base
            end if
            if (i == 0) then
                displs(i + 1) = 0 ! relative offset; callers add the absolute base
            else
                displs(i + 1) = displs(i) + recvcounts(i)
            end if
        end do

        ! Get the local number of elements and the local offset
        NODECOMM_local_i = recvcounts(rank_NODECOMM + 1)
        NODECOMM_local_i_offset = displs(rank_NODECOMM + 1)

    end subroutine NODECOMM_layout_from_DEVCOMM_NODE

end module partitions
