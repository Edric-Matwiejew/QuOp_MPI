program debug_hypercube
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use csr_generators, only: hypercube
    use Sparse, only: Generate_Partition_Table
    implicit none

    integer :: rank, nprocs, ierr, n_qubits
    integer :: system_size, n_local, lb, ub
    integer, allocatable :: partition_table(:)
    integer(dp) :: lb_dp, ub_dp, elem_lb, elem_ub, i
    integer(dp), allocatable :: row_starts(:), col_indexes(:)
    complex(dp), allocatable :: values(:)
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    n_qubits = 8
    system_size = 2**n_qubits
    
    call Generate_Partition_Table(system_size, partition_table, MPI_COMM_WORLD)
    
    lb = partition_table(rank + 1)
    ub = partition_table(rank + 2) - 1
    n_local = ub - lb + 1
    lb_dp = lb
    ub_dp = ub
    
    elem_lb = n_qubits * (lb_dp - 1) + 1
    elem_ub = n_qubits * ub_dp
    
    print '(A,I0,A,I0,A,I0)', 'Rank ', rank, ': lb=', lb, ' ub=', ub
    print '(A,I0,A,I0,A,I0)', 'Rank ', rank, ': elem_lb=', elem_lb, ' elem_ub=', elem_ub
    
    allocate(row_starts(n_local + 1))
    allocate(col_indexes(elem_lb:elem_ub))
    allocate(values(elem_lb:elem_ub))
    
    call hypercube(int(n_qubits, dp), lb_dp, ub_dp, row_starts, col_indexes, values)
    
    if (rank == 0) then
        print '(A)', 'Rank 0 row_starts (first 4 rows):'
        do i = 1, min(5, n_local + 1)
            print '(A,I0,A,I0)', '  row_starts(', i, ') = ', row_starts(i)
        end do
        print '(A)', 'Rank 0 col_indexes (first 16 elements):'
        do i = elem_lb, min(elem_lb + 15, elem_ub)
            print '(A,I0,A,I0)', '  col_indexes(', i, ') = ', col_indexes(i)
        end do
    end if
    
    if (rank == 1) then
        print '(A)', 'Rank 1 row_starts (first 4 rows):'
        do i = 1, min(5, n_local + 1)
            print '(A,I0,A,I0)', '  row_starts(', i, ') = ', row_starts(i)
        end do
        print '(A)', 'Rank 1 col_indexes (first 16 elements, starting at 1025):'
        do i = elem_lb, min(elem_lb + 15, elem_ub)
            print '(A,I0,A,I0)', '  col_indexes(', i, ') = ', col_indexes(i)
        end do
    end if
    
    call MPI_Finalize(ierr)
end program debug_hypercube
