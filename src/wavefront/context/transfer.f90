module array_transfer
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64
    use mpi
    use hipfort
    use hipfort_check

    implicit none

    private

    public :: counts_and_displs

contains

    ! Function to calculate the cumulative sum of an array
    function cumsum(array) result(cumulative_sum)
        integer(int64), intent(in) :: array(:)
        integer(int64) :: cumulative_sum(size(array))
        integer(int32) :: i

        if (size(array) == 0) return

        cumulative_sum(1) = array(1)
        do i = 2, size(array)
            cumulative_sum(i) = cumulative_sum(i - 1) + array(i)
        end do
    end function cumsum

    subroutine counts_and_displs(NODECOMM_local_i, &
                                 NODECOMM_local_i_offset, &
                                 DEVCOMM_local_i, &
                                 DEVCOMM_local_i_offset, &
                                 NODECOMM, &
                                 DEVCOMM_NODE, &
                                 scounts, &
                                 sdispls, &
                                 rcounts, &
                                 rdispls)

        integer(int64), intent(in) :: NODECOMM_local_i
        integer(int64), intent(in) :: NODECOMM_local_i_offset
        integer(int64), intent(in) :: DEVCOMM_local_i
        integer(int64), intent(in) :: DEVCOMM_local_i_offset
        integer(int32), intent(in) :: NODECOMM, DEVCOMM_NODE
        integer(int64), intent(out) :: scounts(:)
        integer(int64), intent(out) :: sdispls(:)
        integer(int64), intent(out) :: rcounts(:)
        integer(int64), intent(out) :: rdispls(:)

        integer(int64), allocatable :: rdispls_(:)

        integer(int32) :: ierr, rank_NODECOMM, size_NODECOMM, rank_DEVCOMM_NODE, size_DEVCOMM_NODE
        integer(int64), allocatable :: local_is_NODECOMM(:), offsets_NODECOMM(:)
        integer(int64), allocatable :: local_is_DEVCOMM_NODE(:), offsets_DEVCOMM_NODE(:)
        integer(int64) :: remaining_elements
        integer(int64) :: current_offset
        integer(int64) :: index
        integer(int64) :: start_index, end_index, num_elements_to_send

        ! Get the rank and size of the NODECOMM communicator
        call MPI_Comm_rank(NODECOMM, rank_NODECOMM, ierr)
        call MPI_Comm_size(NODECOMM, size_NODECOMM, ierr)

        ! Get the size of the DEVCOMM_NODE communicator
        if (rank_NODECOMM == 0) then
            call MPI_Comm_size(DEVCOMM_NODE, size_DEVCOMM_NODE, ierr)
        end if

        call MPI_Bcast(size_DEVCOMM_NODE, 1, MPI_INTEGER, 0, NODECOMM, ierr)

        ! Allocate arrays
        allocate (local_is_NODECOMM(size_NODECOMM))
        allocate (offsets_NODECOMM(size_NODECOMM))
        allocate (local_is_DEVCOMM_NODE(size_DEVCOMM_NODE))
        allocate (offsets_DEVCOMM_NODE(size_DEVCOMM_NODE))

        ! Gather the DEVCOMM_local_i values for DEVCOMM_NODE if it is not MPI_COMM_NULL
        if (DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Gather(DEVCOMM_local_i, 1, MPI_INTEGER8, local_is_DEVCOMM_NODE, 1, &
                            MPI_INTEGER8, 0, DEVCOMM_NODE, ierr)
        end if

        call MPI_Bcast(local_is_DEVCOMM_NODE, size_DEVCOMM_NODE, MPI_INTEGER8, 0, NODECOMM, ierr)

        offsets_DEVCOMM_NODE = cumsum(local_is_DEVCOMM_NODE) - local_is_DEVCOMM_NODE

        ! Initialize remaining elements and current offset
        remaining_elements = NODECOMM_local_i
        current_offset = NODECOMM_local_i_offset

        ! Initialize scounts and rdispls arrays
        scounts = 0
        rcounts = 0
        sdispls = 0
        rdispls = 0

        do index = 1, size_DEVCOMM_NODE
      if (current_offset < offsets_DEVCOMM_NODE(index) + local_is_DEVCOMM_NODE(index) .and. remaining_elements > 0) then
                ! Calculate how many elements can be sent to the current DEVCOMM_NODE rank
                start_index = max(current_offset, offsets_DEVCOMM_NODE(index))
        end_index = min(current_offset + remaining_elements, offsets_DEVCOMM_NODE(index) + local_is_DEVCOMM_NODE(index))
                num_elements_to_send = max(0_int64, end_index - start_index)

                if (index > 1) then
                    sdispls(index) = scounts(index - 1)
                end if

                scounts(index) = num_elements_to_send
                rdispls(index) = start_index - offsets_DEVCOMM_NODE(index)

                ! Update the remaining elements and current offset
                remaining_elements = remaining_elements - num_elements_to_send
                current_offset = current_offset + num_elements_to_send
            end if

            if (remaining_elements <= 0) then
                exit
            end if
        end do

        sdispls = cumsum(sdispls)
        allocate (rdispls_(size(rdispls)))
        call MPI_Alltoall(scounts, 1, MPI_INTEGER8, rcounts, 1, MPI_INTEGER8, NODECOMM, ierr)
        call MPI_Alltoall(rdispls, 1, MPI_INTEGER8, rdispls_, 1, MPI_INTEGER8, NODECOMM, ierr)

        rdispls = rdispls_

    end subroutine counts_and_displs

end module array_transfer
