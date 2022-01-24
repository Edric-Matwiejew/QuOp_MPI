program eigen

    use :: sparse
    use :: sparse_graphs
    use :: MPI

    implicit none

    integer :: cols = 2**20
    type(CSR) :: A
    complex(dp), dimension(:), allocatable :: c
    integer :: i, j, el = 20

    integer, dimension(:), allocatable :: partition_table
    integer :: ub, lb

    integer :: start_it = 1, max_it = 500, current_it = 1

    integer :: num_simulations
    real(dp) :: b_k1_norm
    complex(dp), dimension(:), allocatable :: b_k, b_k1

    integer :: n
    integer, dimension(:), allocatable :: seed

    complex(dp) :: lambda, pi_coef

    !MPI
    integer :: rank, flock
    integer :: ierr

    call MPI_init(ierr)

    call MPI_comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_comm_size(MPI_COMM_WORLD, flock, ierr)

    call Generate_Partition_Table(cols, partition_table, MPI_COMM_WORLD)

    lb = partition_table(rank + 1)
    ub = partition_table(rank + 2) - 1

    allocate(c(el)) 

    do i = 1, el
        c(i) = cmplx(1,0)
    enddo

    call hermitian_circulant(A, c, cols, partition_table, MPI_COMM_WORLD)

    call Reconcile_Communications(  A, &
                                    partition_table, &
                                    MPI_COMM_WORLD)
    allocate(b_k(lb:ub))
    allocate(b_k1(lb:ub))

    pi_coef = complex(0.0_dp,8.0_dp*atan(1.0_dp)/real(cols,dp))

    do i = lb, ub
        b_k(i) = real(i)/10
    enddo

    do i = 1, max_it

        call SpMV_Series(   A, &
                            b_k, &
                            partition_table, &
                            start_it, &
                            current_it, &
                            max_it, &
                            rank, &
                            b_k1, &
                            MPI_COMM_WORLD)

        b_k1_norm = 0
        do j = lb, ub
                b_k1_norm = b_k1_norm + abs(b_k1(j))**2
        enddo

        call MPI_allreduce( MPI_IN_PLACE, &
                            b_k1_norm, &
                            1, &
                            MPI_DOUBLE, &
                            MPI_SUM, &
                            MPI_COMM_WORLD, &
                            ierr)

        b_k1_norm = sqrt(b_k1_norm)

        do j = lb, ub
                b_k(j) = b_k1(j)/b_k1_norm
        enddo

    enddo

    if (rank == 0) then

        j = 0
        lambda = 0
        do i = A%row_starts(1), A%row_starts(2) - 1
            lambda = lambda + exp(pi_coef)**((A%col_indexes(i) - 1)*j)*A%values(i)
        enddo
        if (abs(b_k1_norm) - abs(lambda) < 1e-3 ) then
                write(*,'(A30,I2,A16)') "SpMV_Series test PASSED with ", flock, " MPI processes."
        else
                write(*,'(A30,I2,A16)') "SpMV_Series test FAILED with ", flock, " MPI processes."
                write(*,*) lambda, b_k1_norm
        endif

    endif

    deallocate(b_k)
    deallocate(b_k1)
    deallocate(A%col_indexes)
    deallocate(A%values)
    deallocate(A%row_starts)

    call MPI_finalize(ierr)

end program eigen
