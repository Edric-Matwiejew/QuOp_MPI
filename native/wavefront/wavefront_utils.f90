module wavefront_utils
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64
    use mpi
    use hipfort
    use hipfort_check
    implicit none

    private

    public :: count_available_GPUS

contains

    integer(int32) function count_available_GPUS(COMM)

        integer(int32), intent(in) :: COMM
        integer(int32) ::num_local_devices
        integer(int32) :: temp_comm_node, temp_comm_node_roots
        integer(int32) :: ierr, color, rank, key

        call MPI_Comm_split_type(COMM, &
                                 MPI_COMM_TYPE_SHARED, &
                                 0, &
                                 MPI_INFO_NULL, &
                                 temp_comm_node, &
                                 ierr)

        if (temp_comm_node /= MPI_COMM_NULL) then
            call MPI_Comm_rank(temp_comm_node, rank, ierr)
        end if

        if (rank == 0) then
            call hipCheck(hipGetDeviceCount(num_local_devices))
            color = 1
        else
            color = MPI_UNDEFINED
        end if

        key = rank

        call MPI_Comm_split(COMM, color, key, temp_comm_node_roots, ierr)

        if (rank == 0) then
            call MPI_Reduce(num_local_devices, count_available_GPUS, 1, MPI_INTEGER, MPI_SUM, 0, &
                            temp_comm_node_roots, ierr)
        end if

        call MPI_Bcast(count_available_GPUS, 1, MPI_INTEGER, 0, COMM, ierr)

        if (temp_comm_node /= MPI_COMM_NULL) then
            call MPI_Comm_Free(temp_comm_node, ierr)
        end if
        if (temp_comm_node_roots /= MPI_COMM_NULL) then
            call MPI_Comm_Free(temp_comm_node_roots, ierr)
        end if

    end function count_available_GPUS

end module wavefront_utils
