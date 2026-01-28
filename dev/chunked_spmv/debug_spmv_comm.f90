program debug_spmv_comm
    ! Debug the exact communication pattern used in our SpMV
    use mpi
    use iso_fortran_env, only: int64, dp => real64
    use chunked_spmv_mod, only: generate_partition_table, build_hypercube_csr, &
                                 setup_graph_comm
    implicit none
    
    integer :: rank, nprocs, ierr, i
    integer :: n_qubits
    integer(int64) :: system_size, n_local, local_nnz, lb, ub
    integer(int64), allocatable :: partition_table(:)
    integer(int64), allocatable :: row_starts(:), col_indexes(:)
    integer :: graph_comm, total_recv, total_send
    integer(int64), allocatable :: recv_indices_sorted(:), send_offsets(:)
    integer, allocatable :: sort_perm(:), recv_counts(:), recv_disps(:)
    integer, allocatable :: send_counts(:), send_disps(:)
    integer, allocatable :: in_neighbors(:), out_neighbors(:)
    complex(dp), allocatable :: send_buf(:), recv_buf(:)
    integer :: request
    integer :: status(MPI_STATUS_SIZE)
    real(dp) :: t_start, t_end
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    n_qubits = 16
    system_size = 2_int64**n_qubits
    
    if (rank == 0) print '(A,I0,A,I0)', 'Testing with ', nprocs, ' ranks, ', n_qubits, ' qubits'
    
    ! Generate partition table (0-based)
    call generate_partition_table(system_size, partition_table)
    
    ! Build hypercube CSR (0-based columns, sorted rows)
    call build_hypercube_csr(n_qubits, partition_table, row_starts, col_indexes, n_local, local_nnz)
    
    lb = partition_table(rank + 1)
    ub = partition_table(rank + 2) - 1
    
    ! Setup graph communicator (redirect debug output)
    if (rank == 0) print *, 'Calling setup_graph_comm...'
    call setup_graph_comm(row_starts, col_indexes, partition_table, &
                          graph_comm, recv_indices_sorted, sort_perm, &
                          recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                          in_neighbors, out_neighbors, total_recv, total_send, lb, ub)
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (rank == 0) print *, 'Setup complete'
    
    ! Print neighbor info
    print '(A,I0,A,I0,A,I0)', 'Rank ', rank, ': n_out (indegree)=', size(out_neighbors), &
          ' n_in (outdegree)=', size(in_neighbors)
    print '(A,I0,A,10I4)', 'Rank ', rank, ': out_neighbors (sources)=', out_neighbors
    print '(A,I0,A,10I4)', 'Rank ', rank, ': in_neighbors (destinations)=', in_neighbors
    print '(A,I0,A,10I6)', 'Rank ', rank, ': recv_counts=', recv_counts
    print '(A,I0,A,10I6)', 'Rank ', rank, ': send_counts=', send_counts
    print '(A,I0,A,I0,A,I0)', 'Rank ', rank, ': total_recv=', total_recv, ' total_send=', total_send
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    ! Allocate buffers
    allocate(send_buf(max(total_send, 1)), recv_buf(max(total_recv, 1)))
    
    ! Fill send buffer with test values
    do i = 1, total_send
        send_buf(i) = cmplx(rank * 1000 + i, 0.0_dp, dp)
    end do
    recv_buf = (-1.0_dp, 0.0_dp)
    
    ! Time the neighbor alltoallv - run multiple times with overlapped work
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    block
        real(dp) :: dummy
        integer :: k
        dummy = 0.0_dp
        
        do i = 1, 10
            t_start = MPI_Wtime()
            
            call MPI_Ineighbor_alltoallv(send_buf, send_counts, send_disps, MPI_DOUBLE_COMPLEX, &
                                          recv_buf, recv_counts, recv_disps, MPI_DOUBLE_COMPLEX, &
                                          graph_comm, request, ierr)
            
            ! Do some overlapped work (simulating local computation)
            do k = 1, 1000000
                dummy = dummy + 1.0_dp  ! Dummy work NOT touching MPI buffers
            end do
            
            call MPI_Wait(request, status, ierr)
            
            t_end = MPI_Wtime()
            
            if ((t_end - t_start)*1000 > 10.0_dp) then
                print '(A,I0,A,I0,A,F10.3,A)', 'Rank ', rank, ' iter ', i, &
                      ': SLOW Neighbor_alltoallv time: ', (t_end - t_start)*1000, ' ms'
            end if
        end do
        
        if (dummy < 0) print *, 'never'  ! Prevent optimization
    end block
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (rank == 0) print *, 'All iterations complete'
    
    ! Verify received data
    print '(A,I0,A,5F8.0,A)', 'Rank ', rank, ': recv_buf(1:5)=', real(recv_buf(1:min(5,total_recv))), '...'
    
    deallocate(send_buf, recv_buf)
    call MPI_Comm_free(graph_comm, ierr)
    call MPI_Finalize(ierr)
end program debug_spmv_comm
