program dagger

        use :: sparse_graphs
        use :: MPI

        implicit none

        type(CSR) :: A, A_dagger
        complex(dp), dimension(:), allocatable :: c
        integer, dimension(:), allocatable :: partition_table
        
        integer :: cols = 2 ** 16, n = 20
        integer :: i, j

        !MPI
        integer :: ierr
        integer :: rank, flock

        call MPI_init(ierr)

        call Generate_Partition_Table(cols, partition_table, MPI_COMM_WORLD)

        allocate(c(n))

        do i = 1, n
                c(i) = cmplx(i,i)
        enddo

        call hermitian_circulant(A, c, cols, partition_table, MPI_COMM_WORLD)

        call CSR_Dagger(A, partition_table, A_dagger, MPI_COMM_WORLD)

        call MPI_comm_rank(MPI_COMM_WORLD, rank, ierr) 

        do i = partition_table(rank + 1), partition_table(rank + 2) - 1
                do j = A%row_starts(i), A%row_starts(i + 1) - 1
                        if (A%col_indexes(j) /= A_dagger%col_indexes(j)) then
                                write(*,*) "CSR_Dagger test FAILED: A /= A_dagger"
                                call exit(1)
                        elseif (A%values(j) /= A_dagger%values(j)) then
                                write(*,*) "CSR_Dagger test FAILED: A /= A_dagger"
                                call exit(1)
                        endif
                 enddo
        enddo

        call MPI_comm_size(MPI_COMM_WORLD, flock, ierr)

        if (rank == 0) then
                write(*,'(A29,I2,A16)') "CSR_Dagger test PASSED with ", flock, " MPI processes."
        endif

        call MPI_finalize(ierr)

 end program dagger
