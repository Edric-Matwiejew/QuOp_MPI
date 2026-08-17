module sparse_vector
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, int32, int64
    use mpi

    implicit none

    private
    public :: to_sparse_vector

contains

    subroutine to_sparse_vector(dense_array, nnz, indexes, values, local_i, local_i_offset, MPI_communicator)
        real(real64), dimension(:), intent(in) :: dense_array
        integer(int32), intent(out) :: nnz
        integer(int64), dimension(:), allocatable, intent(out) :: indexes
        real(real64), dimension(:), allocatable, intent(out) :: values
        integer(int64), intent(in) :: local_i, local_i_offset
        integer(int32), intent(in) :: MPI_communicator

        integer(int64), dimension(:), allocatable :: local_indexes
        real(real64), dimension(:), allocatable :: local_values
        integer(int32) :: local_nnz

        integer(int32) :: rank, flock
        integer(int32), dimension(:), allocatable :: counts, disps
        integer(int64) :: global_i
        integer(int32) :: i, ierr

        local_nnz = 0

        do global_i = local_i_offset + 1_int64, local_i_offset + local_i
            if (abs(dense_array(global_i)) > epsilon(1.0_real64)) then
                local_nnz = local_nnz + 1
            end if
        end do

        allocate (local_indexes(local_nnz), local_values(local_nnz))

        local_nnz = 0

        do global_i = local_i_offset + 1_int64, local_i_offset + local_i
            if (abs(dense_array(global_i)) > epsilon(1.0_real64)) then
                local_nnz = local_nnz + 1
                local_indexes(local_nnz) = global_i
                local_values(local_nnz) = dense_array(global_i)
            end if
        end do

        call MPI_Comm_rank(MPI_communicator, rank, ierr)
        call MPI_Comm_size(MPI_communicator, flock, ierr)

        allocate (counts(flock))
        call MPI_Allgather(local_nnz, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, MPI_communicator, ierr)

        nnz = sum(counts)

        allocate (indexes(nnz), values(nnz), disps(flock))

        disps(1) = 0
        do i = 2, flock
            disps(i) = disps(i - 1) + counts(i - 1)
        end do

    call MPI_Allgatherv(local_indexes, local_nnz, MPI_INTEGER8, indexes, counts, disps, MPI_INTEGER8, MPI_communicator, ierr)
     call MPI_Allgatherv(local_values, local_nnz, MPI_DOUBLE, values, counts, disps, MPI_DOUBLE, MPI_communicator, ierr)

    end subroutine to_sparse_vector

end module sparse_vector
