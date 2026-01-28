program test_neighbor_spmv
    ! Test that MPI_Ineighbor_alltoallv with a graph communicator
    ! correctly exchanges data between neighbors, simulating the SpMV pattern
    use mpi
    use iso_fortran_env, only: int64
    implicit none
    
    integer :: rank, nprocs, ierr
    integer :: graph_comm
    integer :: n_out, n_in  ! indegree, outdegree
    integer, allocatable :: out_neighbors(:), in_neighbors(:)  ! sources, destinations
    integer, allocatable :: out_weights(:), in_weights(:)
    integer, allocatable :: recv_counts(:), recv_disps(:)
    integer, allocatable :: send_counts(:), send_disps(:)
    integer, allocatable :: sendbuf(:), recvbuf(:)
    integer :: total_send, total_recv, request
    integer :: status(MPI_STATUS_SIZE)
    integer :: i, j, expected
    logical :: passed
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    ! Hypercube-like pattern: each rank exchanges with all others
    ! (simplified version of actual SpMV pattern)
    
    ! out_neighbors = ranks we RECEIVE from (indegree)
    ! in_neighbors = ranks we SEND to (outdegree)
    ! For hypercube, these are the same set (symmetric)
    
    n_out = nprocs - 1  ! Receive from everyone except self
    n_in = nprocs - 1   ! Send to everyone except self
    
    allocate(out_neighbors(n_out), in_neighbors(n_in))
    allocate(out_weights(n_out), in_weights(n_in))
    
    j = 1
    do i = 0, nprocs - 1
        if (i /= rank) then
            out_neighbors(j) = i
            in_neighbors(j) = i
            j = j + 1
        end if
    end do
    
    out_weights = 1
    in_weights = 1
    
    if (rank == 0) then
        print '(A)', '=== Graph Neighbor Alltoallv Test ==='
        print '(A,I0,A)', 'Running with ', nprocs, ' ranks'
    end if
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    print '(A,I0,A,10I3)', 'Rank ', rank, ': out_neighbors (recv from) = ', out_neighbors
    print '(A,I0,A,10I3)', 'Rank ', rank, ': in_neighbors (send to) = ', in_neighbors
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    ! Create graph communicator
    ! sources = out_neighbors (ranks we receive FROM)
    ! destinations = in_neighbors (ranks we send TO)
    call MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, &
            n_out, out_neighbors, out_weights, &
            n_in, in_neighbors, in_weights, &
            MPI_INFO_NULL, .false., graph_comm, ierr)
    
    if (rank == 0) print '(A)', 'Graph communicator created'
    
    ! Set up counts: send 2 values to each destination, receive 2 from each source
    allocate(recv_counts(n_out), recv_disps(n_out))
    allocate(send_counts(n_in), send_disps(n_in))
    
    recv_counts = 2
    send_counts = 2
    
    recv_disps(1) = 0
    do i = 2, n_out
        recv_disps(i) = recv_disps(i-1) + recv_counts(i-1)
    end do
    
    send_disps(1) = 0
    do i = 2, n_in
        send_disps(i) = send_disps(i-1) + send_counts(i-1)
    end do
    
    total_recv = sum(recv_counts)
    total_send = sum(send_counts)
    
    allocate(sendbuf(total_send), recvbuf(total_recv))
    recvbuf = -1
    
    ! Pack send buffer: send (rank*100 + dest*10 + 1) and (rank*100 + dest*10 + 2) to each dest
    do i = 1, n_in
        sendbuf(send_disps(i) + 1) = rank * 100 + in_neighbors(i) * 10 + 1
        sendbuf(send_disps(i) + 2) = rank * 100 + in_neighbors(i) * 10 + 2
    end do
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    print '(A,I0,A,10I5)', 'Rank ', rank, ': sendbuf = ', sendbuf
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (rank == 0) print '(A)', ''
    if (rank == 0) print '(A)', 'Calling MPI_Ineighbor_alltoallv...'
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    ! MPI_Ineighbor_alltoallv:
    ! sendbuf indexed by outdegree (destinations = in_neighbors)
    ! recvbuf indexed by indegree (sources = out_neighbors)
    call MPI_Ineighbor_alltoallv(sendbuf, send_counts, send_disps, MPI_INTEGER, &
                                  recvbuf, recv_counts, recv_disps, MPI_INTEGER, &
                                  graph_comm, request, ierr)
    
    call MPI_Wait(request, status, ierr)
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (rank == 0) print '(A)', 'Communication complete'
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    print '(A,I0,A,10I5)', 'Rank ', rank, ': recvbuf = ', recvbuf
    
    ! Verify: recvbuf for out_neighbors(i) should be (out_neighbors(i)*100 + rank*10 + {1,2})
    passed = .true.
    do i = 1, n_out
        expected = out_neighbors(i) * 100 + rank * 10 + 1
        if (recvbuf(recv_disps(i) + 1) /= expected) then
            print '(A,I0,A,I0,A,I0,A,I0)', 'Rank ', rank, ': source ', i, &
                  ' val1 WRONG: got ', recvbuf(recv_disps(i) + 1), ' expected ', expected
            passed = .false.
        end if
        expected = out_neighbors(i) * 100 + rank * 10 + 2
        if (recvbuf(recv_disps(i) + 2) /= expected) then
            print '(A,I0,A,I0,A,I0,A,I0)', 'Rank ', rank, ': source ', i, &
                  ' val2 WRONG: got ', recvbuf(recv_disps(i) + 2), ' expected ', expected
            passed = .false.
        end if
    end do
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    if (passed) then
        print '(A,I0,A)', 'Rank ', rank, ': ALL CORRECT'
    else
        print '(A,I0,A)', 'Rank ', rank, ': FAILED'
    end if
    
    call MPI_Comm_free(graph_comm, ierr)
    deallocate(out_neighbors, in_neighbors, out_weights, in_weights)
    deallocate(recv_counts, recv_disps, send_counts, send_disps)
    deallocate(sendbuf, recvbuf)
    
    call MPI_Finalize(ierr)
end program test_neighbor_spmv
