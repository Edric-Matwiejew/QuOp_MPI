program debug_quop
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use Sparse, only: CSR, SpMV_Series, Reconcile_Communications, Generate_Partition_Table
    implicit none

    integer :: rank, nprocs, ierr, k, n_qubits
    integer :: system_size, n_local, lb, ub
    integer, allocatable :: partition_table(:)
    type(CSR) :: A
    complex(dp), allocatable :: u(:), v(:)
    integer(dp) :: i, idx, global_row, local_nnz
    integer(dp) :: temp_cols(20)
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    n_qubits = 8
    system_size = 2**n_qubits
    
    call Generate_Partition_Table(system_size, partition_table, MPI_COMM_WORLD)
    
    lb = partition_table(rank + 1)
    ub = partition_table(rank + 2) - 1
    n_local = ub - lb + 1
    local_nnz = int(n_local, dp) * n_qubits
    
    print '(A,I0,A,I0,A,I0)', 'Rank ', rank, ': lb=', lb, ' ub=', ub
    
    ! Build CSR matrix
    A%rows = system_size
    A%columns = system_size
    A%structure = 'SY'
    A%is_unit_valued = .true.
    
    allocate(A%row_starts(lb:ub+1))
    allocate(A%col_indexes(local_nnz))
    allocate(A%values(local_nnz))
    
    idx = 1
    do i = lb, ub
        A%row_starts(i) = idx
        global_row = i - 1  ! 0-based for XOR
        
        do k = 1, n_qubits
            temp_cols(k) = ieor(global_row, ishft(1_int64, k - 1)) + 1  ! 1-based
        end do
        
        do k = 1, n_qubits
            A%col_indexes(idx) = temp_cols(k)
            A%values(idx) = (1.0_dp, 0.0_dp)
            idx = idx + 1
        end do
    end do
    A%row_starts(ub + 1) = idx
    
    ! Debug output for rank 0
    if (rank == 0) then
        print '(A)', 'Row starts:'
        do i = lb, min(lb+3, ub)
            print '(A,I0,A,I0,A,I0)', '  row_starts(', i, ') = ', A%row_starts(i), &
                  '  to ', A%row_starts(i+1)-1
        end do
        print '(A)', 'First few col_indexes:'
        do i = 1, min(16, local_nnz)
            print '(A,I0,A,I0)', '  col_indexes(', i, ') = ', A%col_indexes(i)
        end do
    end if
    
    call Reconcile_Communications(A, partition_table, MPI_COMM_WORLD)
    
    ! Allocate vectors
    allocate(u(lb:ub), v(lb:ub))
    u = (1.0_dp, 0.0_dp)
    
    call SpMV_Series(A, u, partition_table, 1, 1, 1, rank, v, MPI_COMM_WORLD)
    
    ! Check result
    print '(A,I0,A,2F8.4)', 'Rank ', rank, ': v(lb) = ', v(lb)
    print '(A,I0,A,F8.4)', 'Rank ', rank, ': expected = ', real(n_qubits)
    
    call MPI_Finalize(ierr)
end program debug_quop
