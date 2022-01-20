program spmmv_series

    use :: sparse
    use :: MPI

    implicit none

    type(CSR) :: A, A_local
    integer :: rows = 100, columns = 100
    integer :: nonzeros_per_row = 10
    integer :: nonzeros
    integer :: block = 0, n_blocks
    integer :: i, j

    integer, dimension(:), allocatable :: partition_table
    integer :: ub, lb

    complex(dp), dimension(:), allocatable :: u_local, v_local
    integer :: rows_local
    integer :: start_it = 1, max_it = 1000, current_it = 1

    !MPI
    integer :: root = 0
    integer :: rank
    integer :: ierr

    call MPI_init(ierr)

    call MPI_comm_rank(MPI_COMM_WORLD, rank, ierr)

    if (rank == 0) then

            nonzeros = nonzeros_per_row * rows
            A%rows = rows
            A%columns = columns

            allocate(A%row_starts(rows + 1))
            allocate(A%col_indexes(nonzeros))
            allocate(A%values(nonzeros))

            A%row_starts(1) = 1

            do i = 1, rows
                    A%row_starts(i + 1) = A%row_starts(i) + nonzeros_per_row
            enddo

            A%values = cmplx(1, 1, dp)

            do i = 1, rows

                do j = 1, nonzeros_per_row
                    A%col_indexes((i-1) * nonzeros_per_row + j) = j
                enddo

            enddo
    endif

    call Generate_Partition_Table(rows, partition_table, MPI_COMM_WORLD)


    lb = partition_table(rank + 1)
    ub = partition_table(rank + 2) - 1

    allocate(u_local(lb:ub))
    allocate(v_local(lb:ub))

    u_local = cmplx(1, 1, dp)
    v_local = cmplx(0, 0, dp)

    call Distribute_CSR_Matrix(   A, &
                                  partition_table, &
                                  root, &
                                  A_local, &
                                  MPI_COMM_WORLD)


    call Reconcile_Communications(  A_local, &
                                    partition_table, &
                                    MPI_COMM_WORLD)

    do i = 1, max_it

        call SpMV_Series(   A_local, &
                            u_local, &
                            partition_table, &
                            start_it, &
                            current_it, &
                            max_it, &
                            rank, &
                            v_local, &
                            MPI_COMM_WORLD)

    enddo

    deallocate(u_local)
    deallocate(v_local)
    deallocate(A_local%col_indexes)
    deallocate(A_local%values)
    deallocate(A_local%row_starts)

    if (rank == 0) then
        deallocate(A%col_indexes)
        deallocate(A%values)
        deallocate(A%row_starts)
    endif

    call MPI_finalize(ierr)

end program spmmv_series
