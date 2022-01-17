program spmmv_series

        use :: sparse
        use :: MPI

        implicit none

        type(CSR) :: A, A_local
        integer :: rows = 10000, columns = 10000
        integer :: nonzeros_per_row = 100
        integer :: nonzeros
        integer :: block = 0
        integer :: i, j

        integer, dimension(:), allocatable :: partition_table

        complex(dp), dimension(:), allocatable :: u_local, v_local
        integer :: rows_local
        integer :: start_it = 1, max_it = 1000, current_it = 1

        !MPI
        integer :: root
        integer :: rank
        integer :: ierr

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

                A%values = 1

                do i = 1, rows

                        if ( mod(i, nonzeros_per_row) == 0 ) then
                                block = block + 1
                        endif        

                        do j = 1, columns
                                A%col_indexes(i * nonzeros_per_row + j) = columns + block * nonzeros_per_row  
                        enddo

                enddo

        endif

        call Generate_Partition_Table(rows, partition_table, MPI_COMM_WORLD)


        rows_local = partition_table(rank + 1) - partition_table(rank)

        allocate(u_local(rows_local))
        allocate(v_local(rows_local))

        call Distribute_CSR_Matrix(   A, &
                                      partition_table, &
                                      root, &
                                      A_local, &
                                      MPI_COMM_WORLD)


        call Reconcile_Communications(  A, &
                                        partition_table, &
                                        MPI_COMM_WORLD)

        call SpMV_Series(   A, &
                            u_local, &
                            partition_table, &
                            start_it, &
                            current_it, &
                            max_it, &
                            rank, &
                            v_local, &
                            MPI_COMM_WORLD)

end program spmmv_series
