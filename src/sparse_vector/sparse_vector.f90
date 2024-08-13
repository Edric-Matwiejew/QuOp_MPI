module sparse_vector
    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, qp => real128
    use mpi

    implicit none

    contains

subroutine to_sparse_vector(dense_array, nnz, indexes, values, local_i, local_i_offset, MPI_communicator)
    real(dp), dimension(:), intent(in) :: dense_array
    integer(sp), intent(out) :: nnz
    integer(dp), dimension(:), allocatable, intent(out) :: indexes
    real(dp), dimension(:), allocatable, intent(out) :: values
    integer(dp), intent(in) :: local_i, local_i_offset
    integer(sp), intent(in) :: MPI_communicator

    integer(dp), dimension(:), allocatable :: local_indexes
    real(dp), dimension(:), allocatable :: local_values
    integer(sp) :: local_nnz = 0

    integer(sp) :: rank, flock
    integer(sp), dimension(:), allocatable :: counts, disps
    integer(sp) :: i, ierr

    ! Count the number of non-zero elements in the local portion of dense_array
    do i = local_i_offset + 1, local_i_offset + local_i
        if (abs(dense_array(i)) > epsilon(1.0d0)) then
            local_nnz = local_nnz + 1
        endif
    enddo

    ! Allocate local arrays
    allocate(local_indexes(local_nnz), local_values(local_nnz))

    ! Initialize index for storing local values
    local_nnz = 0
    
    ! Fill local arrays with non-zero elements
    do i = local_i_offset + 1, local_i_offset + local_i
        if (abs(dense_array(i)) > epsilon(1.0d0)) then
            local_nnz = local_nnz + 1
            local_indexes(local_nnz) = i
            local_values(local_nnz) = dense_array(i)
        endif
    enddo

    ! Initialize MPI
    call MPI_Comm_rank(MPI_communicator, rank, ierr)
    call MPI_Comm_size(MPI_communicator, flock, ierr)

    ! Gather the counts of non-zero elements from all processes
    allocate(counts(flock))
    call MPI_Allgather(local_nnz, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, MPI_communicator, ierr)

    ! Calculate the total number of non-zero elements
    nnz = sum(counts)

    ! Allocate arrays for the global result
    allocate(indexes(nnz), values(nnz), disps(flock))

    ! Calculate displacements for Allgatherv
    disps(1) = 0
    do i = 2, flock
        disps(i) = disps(i - 1) + counts(i - 1)
    enddo

    ! Gather all local indexes and values to form global arrays
    call MPI_Allgatherv(local_indexes, local_nnz, MPI_LONG, indexes, counts, disps, MPI_LONG, MPI_communicator, ierr)
    call MPI_Allgatherv(local_values, local_nnz, MPI_DOUBLE, values, counts, disps, MPI_DOUBLE, MPI_communicator, ierr)

end subroutine to_sparse_vector

end module sparse_vector