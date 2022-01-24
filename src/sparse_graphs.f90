module sparse_graphs

        use :: ISO_Precisions
        use :: sparse
        use :: MPI

        implicit None

        contains

        subroutine hermitian_circulant(A, c, cols, partition_table, MPI_Communicator)

                type(CSR), intent(inout) :: A
                complex(dp), dimension(:), allocatable, intent(in) :: c
                integer, intent(in) :: cols
                integer, dimension(:), allocatable, intent(in) :: partition_table
                integer, intent(in) :: MPI_Communicator

                integer :: n_row_nz, n_local_rows, idx
                integer :: lb, ub, elements_lb, elements_ub
                integer :: i, j

                integer, dimension(:), allocatable :: base_cols

                !MPI
                integer :: rank
                integer :: ierr

                idx = size(c)

                if (idx > cols /2) then
                        write(*,*) "ERROR: Input array greater than cols / 2."
                        call exit(1) 
                else
                        n_row_nz = 2 * idx
                endif 

                A%rows = cols
                A%columns = cols

                call MPI_comm_rank(MPI_Communicator, rank, ierr)

                lb = partition_table(rank + 1)
                ub = partition_table(rank + 2) - 1

                n_local_rows = ub - lb + 1

                allocate(A%row_starts(lb:ub + 1))

                do i = lb, ub + 1
                        A%row_starts(i) = (i - 1) * n_row_nz + 1
                enddo

                elements_lb = A%row_starts(lb)
                elements_ub = A%row_starts(ub + 1) - 1

                allocate(A%col_indexes(elements_lb:elements_ub))
                allocate(A%values(elements_lb:elements_ub))

                allocate(base_cols(n_row_nz))

                do i = 1, idx
                        base_cols(i) = i
                        base_cols(n_row_nz - i + 1) = cols - i
                enddo

                do i = lb, ub
                        do j = 1, idx

                                A%col_indexes(j + (i - 1) * n_row_nz) = mod(base_cols(j) + i - 1, cols) + 1

                                if (A%col_indexes(j + (i - 1) * n_row_nz) > i) then
                                        A%values((i - 1) *  n_row_nz + j) = c(j)
                                else
                                        A%values((i - 1) *  n_row_nz + j) = conjg(c(j))
                                endif

                        enddo
                        do j = 1, idx

                                A%col_indexes(j + idx + (i - 1) * n_row_nz) = mod(base_cols(j + idx) + i - 1, cols) + 1

                                if (A%col_indexes(j + idx + (i - 1) * n_row_nz) > i) then
                                        A%values((i - 1) *  n_row_nz + j + idx) = c(idx - j + 1)
                                else
                                        A%values((i - 1) *  n_row_nz + j + idx) = conjg(c(idx - j + 1))
                                endif

                        enddo 
                enddo

                call Sort_CSR(A)

        end subroutine hermitian_circulant

end module sparse_graphs
