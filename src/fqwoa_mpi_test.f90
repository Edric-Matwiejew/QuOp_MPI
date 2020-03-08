module fqwoa_mpi

    contains

    include 'fqwoa_mpi.f90'

end module fqwoa_mpi

program complex_write

    use :: MPI
    use :: fqwoa_mpi
    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

    implicit none

    character(len = 128) :: file_name, group_name, dataset_name

    integer(dp) :: N
    integer(sp) :: local_i, local_i_offset

    complex(dp) :: complex_array(5)

    integer :: i

    integer :: flock
    integer :: rank
    integer :: ierr

    call MPI_init(ierr)

    file_name = "test_write"
    group_name = "test_group"
    dataset_name = "test_complex"

    call MPI_comm_size(MPI_COMM_WORLD, flock, ierr)
    call MPI_comm_rank(MPI_COMM_WORLD, rank, ierr)
    
    local_i = 5
    N = flock * local_i

    local_i_offset = local_i * (rank)

    write(*,*) rank, local_i, local_i_offset, N

    do i = 1, local_i
        complex_array(i) = cmplx(i, i, dp)
    enddo

    call save_dist_complex( file_name, &
                            group_name, &
                            dataset_name, &
                            "w", &
                            N, &
                            local_i, &
                            local_i_offset, &
                            complex_array, &
                            MPI_COMM_WORLD)

    write(*,*) "DONE"
    call MPI_finalize(ierr)

end program complex_write
