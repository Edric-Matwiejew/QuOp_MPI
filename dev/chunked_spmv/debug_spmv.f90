program debug_spmv
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use csr_generators, only: hypercube
    use Sparse, only: CSR, SpMV_Series, Reconcile_Communications, Generate_Partition_Table
    implicit none

    integer :: rank, nprocs, ierr, n_qubits
    integer :: system_size, n_local, lb, ub
    integer, allocatable :: partition_table(:)
    type(CSR) :: A
    complex(dp), allocatable :: u(:), v(:)
    integer(dp) :: lb_dp, ub_dp, elem_lb, elem_ub, i, j, col
    integer(dp), allocatable :: row_starts_temp(:), col_indexes_temp(:)
    complex(dp), allocatable :: values_temp(:)
    integer :: local_count
    
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
    
    allocate(row_starts_temp(n_local + 1))
    allocate(col_indexes_temp(elem_lb:elem_ub))
    allocate(values_temp(elem_lb:elem_ub))
    
    call hypercube(int(n_qubits, dp), lb_dp, ub_dp, row_starts_temp, col_indexes_temp, values_temp)
    
    A%rows = system_size
    A%columns = system_size
    A%structure = 'SY'
    A%is_unit_valued = .true.
    
    allocate(A%row_starts(lb:ub+1))
    allocate(A%col_indexes(elem_lb:elem_ub))
    allocate(A%values(elem_lb:elem_ub))
    
    do i = 1, n_local + 1
        A%row_starts(lb + i - 1) = row_starts_temp(i)
    end do
    A%col_indexes = col_indexes_temp
    A%values = values_temp
    
    ! Debug: check what SpMV will iterate over for row 129 (rank 1's first row)
    if (rank == 1) then
        print '(A,I0)', 'A%row_starts(129) = ', A%row_starts(129)
        print '(A,I0)', 'A%row_starts(130) = ', A%row_starts(130)
        print '(A)', 'Columns for row 129:'
        local_count = 0
        do j = A%row_starts(129), A%row_starts(130)-1
            col = A%col_indexes(j)
            if (col >= lb .and. col <= ub) then
                local_count = local_count + 1
                print '(A,I0,A,I0,A)', '  j=', j, ' col=', col, ' LOCAL'
            else
                print '(A,I0,A,I0,A)', '  j=', j, ' col=', col, ' REMOTE'
            end if
        end do
        print '(A,I0)', 'Local columns: ', local_count
    end if
    
    call Reconcile_Communications(A, partition_table, MPI_COMM_WORLD)
    
    allocate(u(lb:ub), v(lb:ub))
    u = (1.0_dp, 0.0_dp)
    
    call SpMV_Series(A, u, partition_table, 1, 1, 1, rank, v, MPI_COMM_WORLD)
    
    if (rank == 1) then
        print '(A,2F8.4)', 'v(129) = ', v(129)
        print '(A,F8.4)', 'Expected: ', real(n_qubits)
    end if
    
    call MPI_Finalize(ierr)
end program debug_spmv
