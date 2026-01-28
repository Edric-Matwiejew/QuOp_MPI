program test_graph_comm
    use mpi
    implicit none
    
    integer :: rank, nprocs, ierr
    integer :: graph_comm
    integer :: indegree, outdegree
    integer, allocatable :: sources(:), destinations(:)
    integer, allocatable :: sourceweights(:), destweights(:)
    integer, allocatable :: sendcounts(:), sdispls(:)
    integer, allocatable :: recvcounts(:), rdispls(:)
    integer, allocatable :: sendbuf(:), recvbuf(:)
    integer :: i, total_send, total_recv
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    if (nprocs < 3) then
        if (rank == 0) print *, 'Need at least 3 processes'
        call MPI_Finalize(ierr)
        stop
    end if
    
    ! Simple test pattern for 4 ranks:
    ! Rank 0: sends to 1,2    receives from 1
    ! Rank 1: sends to 0,2    receives from 0,2,3
    ! Rank 2: sends to 1,3    receives from 0,1
    ! Rank 3: sends to 1      receives from 2
    
    ! Each rank will send its rank*10+dest to each destination
    ! e.g., rank 0 sends 01 to rank 1, 02 to rank 2
    
    select case (rank)
    case (0)
        indegree = 1
        outdegree = 2
        allocate(sources(1), destinations(2))
        sources = [1]           ! I receive FROM rank 1
        destinations = [1, 2]   ! I send TO ranks 1, 2
        
    case (1)
        indegree = 3
        outdegree = 2
        allocate(sources(3), destinations(2))
        sources = [0, 2, 3]     ! I receive FROM ranks 0, 2, 3
        destinations = [0, 2]   ! I send TO ranks 0, 2
        
    case (2)
        indegree = 2
        outdegree = 2
        allocate(sources(2), destinations(2))
        sources = [0, 1]        ! I receive FROM ranks 0, 1
        destinations = [1, 3]   ! I send TO ranks 1, 3
        
    case (3)
        indegree = 1
        outdegree = 1
        allocate(sources(1), destinations(1))
        sources = [2]           ! I receive FROM rank 2
        destinations = [1]      ! I send TO rank 1
        
    case default
        indegree = 0
        outdegree = 0
        allocate(sources(1), destinations(1))
    end select
    
    allocate(sourceweights(max(indegree,1)), destweights(max(outdegree,1)))
    sourceweights = 1
    destweights = 1
    
    print '(A,I0,A,I0,A,I0)', 'Rank ', rank, ': indegree=', indegree, ' outdegree=', outdegree
    if (indegree > 0) print '(A,I0,A,10I3)', 'Rank ', rank, ': sources=', sources
    if (outdegree > 0) print '(A,I0,A,10I3)', 'Rank ', rank, ': destinations=', destinations
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    ! Create graph communicator
    ! sources = ranks that send data TO me (I receive FROM them)
    ! destinations = ranks I send data TO (they receive FROM me)
    call MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, &
            indegree, sources, sourceweights, &
            outdegree, destinations, destweights, &
            MPI_INFO_NULL, .false., graph_comm, ierr)
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (rank == 0) print *, ''
    if (rank == 0) print *, '=== Testing MPI_Neighbor_alltoallv ==='
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    ! Now test neighbor alltoallv
    ! For MPI_Neighbor_alltoallv:
    !   sendbuf is sent TO destinations (outdegree entries)
    !   recvbuf receives FROM sources (indegree entries)
    !   sendcounts/sdispls are indexed by destination order (0..outdegree-1)
    !   recvcounts/rdispls are indexed by source order (0..indegree-1)
    
    ! Each destination gets 1 integer: rank*10 + dest_rank
    allocate(sendcounts(max(outdegree,1)), sdispls(max(outdegree,1)))
    allocate(recvcounts(max(indegree,1)), rdispls(max(indegree,1)))
    
    ! Send 1 element to each destination
    sendcounts = 1
    sdispls(1) = 0
    do i = 2, outdegree
        sdispls(i) = sdispls(i-1) + sendcounts(i-1)
    end do
    total_send = sum(sendcounts(1:outdegree))
    
    ! Receive 1 element from each source  
    recvcounts = 1
    rdispls(1) = 0
    do i = 2, indegree
        rdispls(i) = rdispls(i-1) + recvcounts(i-1)
    end do
    total_recv = sum(recvcounts(1:indegree))
    
    allocate(sendbuf(max(total_send,1)), recvbuf(max(total_recv,1)))
    sendbuf = 0
    recvbuf = -1  ! Initialize to -1 to see what gets filled
    
    ! Fill send buffer: rank*10 + destination_rank
    do i = 1, outdegree
        sendbuf(sdispls(i) + 1) = rank * 10 + destinations(i)
    end do
    
    print '(A,I0,A,10I4)', 'Rank ', rank, ': sendbuf=', sendbuf(1:total_send)
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    call MPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, MPI_INTEGER, &
                                 recvbuf, recvcounts, rdispls, MPI_INTEGER, &
                                 graph_comm, ierr)
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    print '(A,I0,A,10I4)', 'Rank ', rank, ': recvbuf=', recvbuf(1:total_recv)
    
    ! Verify: recvbuf[i] should be sources[i]*10 + rank
    if (rank == 0) print *, ''
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    do i = 1, indegree
        if (recvbuf(rdispls(i) + 1) == sources(i) * 10 + rank) then
            print '(A,I0,A,I0,A)', 'Rank ', rank, ': recv from source ', i-1, ' CORRECT'
        else
            print '(A,I0,A,I0,A,I0,A,I0)', 'Rank ', rank, ': recv from source ', i-1, &
                  ' WRONG: got ', recvbuf(rdispls(i) + 1), ' expected ', sources(i) * 10 + rank
        end if
    end do
    
    call MPI_Comm_free(graph_comm, ierr)
    deallocate(sources, destinations, sourceweights, destweights)
    deallocate(sendcounts, sdispls, recvcounts, rdispls)
    deallocate(sendbuf, recvbuf)
    
    call MPI_Finalize(ierr)
end program test_graph_comm
