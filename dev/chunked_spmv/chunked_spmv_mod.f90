!------------------------------------------------------------------------------
! Memory-efficient SpMV module for unit-valued matrices using MPI graph communicators
!
! Features:
! 1. Graph communicator setup for O(neighbors) scaling
! 2. Hash table for O(1) average remote column lookup
! 3. O(unique_remote) memory instead of O(nnz)
!
! Usage:
!   use chunked_spmv_mod
!------------------------------------------------------------------------------

module chunked_spmv_mod
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    implicit none
    
    private
    
    ! Hash table constants (Knuth's golden ratio multiplier)
    integer(int64), parameter :: HASH_MULT = 2654435769_int64
    integer(int64), parameter :: MASK32 = int(Z'FFFFFFFF', int64)
    
    ! Public routines
    public :: generate_partition_table
    public :: build_hypercube_csr
    public :: setup_graph_comm
    public :: build_hash_table
    public :: spmv_sorted_rows
    public :: cleanup_graph_comm
    
    ! Public helper functions (useful for testing)
    public :: lower_bound
    public :: upper_bound
    public :: find_owner
    
contains

    !--------------------------------------------------------------------------
    ! Find first position where arr(pos) >= val (lower bound)
    ! Uses assumed-size array to avoid array slice temporaries in hot loops
    !--------------------------------------------------------------------------
    pure function lower_bound(arr, lo_in, hi_in, val) result(pos)
        integer(int64), intent(in) :: arr(*)
        integer(int64), intent(in) :: lo_in, hi_in, val
        integer(int64) :: pos
        
        integer(int64) :: lo, hi, mid
        
        lo = lo_in
        hi = hi_in + 1  ! One past end
        
        do while (lo < hi)
            mid = (lo + hi) / 2
            if (arr(mid) < val) then
                lo = mid + 1
            else
                hi = mid
            end if
        end do
        pos = lo
    end function lower_bound
    
    !--------------------------------------------------------------------------
    ! Find first position where arr(pos) > val (upper bound)
    ! Uses assumed-size array to avoid array slice temporaries in hot loops
    !--------------------------------------------------------------------------
    pure function upper_bound(arr, lo_in, hi_in, val) result(pos)
        integer(int64), intent(in) :: arr(*)
        integer(int64), intent(in) :: lo_in, hi_in, val
        integer(int64) :: pos
        
        integer(int64) :: lo, hi, mid
        
        lo = lo_in
        hi = hi_in + 1
        
        do while (lo < hi)
            mid = (lo + hi) / 2
            if (arr(mid) <= val) then
                lo = mid + 1
            else
                hi = mid
            end if
        end do
        pos = lo
    end function upper_bound

    !--------------------------------------------------------------------------
    ! Compute hash index for a column using XOR-folding and golden ratio multiply
    ! Folds 64-bit key to 32-bit to avoid overflow, then applies multiplicative hash
    !--------------------------------------------------------------------------
    pure function compute_hash(col, hash_size) result(hash_pos)
        integer(int64), intent(in) :: col, hash_size
        integer(int64) :: hash_pos
        
        integer(int64) :: folded
        
        folded = iand(ieor(col, ishft(col, -32)), MASK32)
        hash_pos = mod(folded * HASH_MULT, hash_size) + 1_int64
        if (hash_pos < 1) hash_pos = hash_pos + hash_size
    end function compute_hash

    !--------------------------------------------------------------------------
    ! Look up a column in the hash table, returns position in sorted array or 0
    !--------------------------------------------------------------------------
    pure function hash_lookup(col, hash_keys, hash_vals, hash_size) result(pos)
        integer(int64), intent(in) :: col
        integer(int64), intent(in) :: hash_keys(:), hash_vals(:)
        integer(int64), intent(in) :: hash_size
        integer(int64) :: pos
        
        integer(int64) :: hash_pos, probe
        
        pos = 0_int64
        hash_pos = compute_hash(col, hash_size)
        
        do probe = 0, hash_size - 1
            if (hash_keys(hash_pos) == col) then
                pos = hash_vals(hash_pos)
                return
            else if (hash_keys(hash_pos) < 0) then
                return  ! Not found
            end if
            hash_pos = mod(hash_pos, hash_size) + 1_int64
        end do
    end function hash_lookup

    !--------------------------------------------------------------------------
    ! Find owner rank for a column index
    !--------------------------------------------------------------------------
    pure function find_owner(col, partition_table) result(owner)
        integer(int64), intent(in) :: col
        integer(int64), intent(in) :: partition_table(:)
        integer :: owner
        
        integer :: lo, hi, mid
        
        lo = 1
        hi = size(partition_table) - 1
        
        do while (lo < hi)
            mid = (lo + hi + 1) / 2
            if (partition_table(mid) <= col) then
                lo = mid
            else
                hi = mid - 1
            end if
        end do
        
        owner = lo - 1  ! Convert to 0-based rank
    end function find_owner

    !--------------------------------------------------------------------------
    ! Generate partition table (0-based column indices, Fortran 1-based array)
    !--------------------------------------------------------------------------
    subroutine generate_partition_table(system_size, partition_table)
        integer(int64), intent(in) :: system_size
        integer(int64), allocatable, intent(out) :: partition_table(:)
        
        integer :: nprocs, i, ierr
        integer(int64) :: base_rows, remainder
        
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        allocate(partition_table(nprocs + 1))
        partition_table(1) = 0
        
        base_rows = system_size / nprocs
        remainder = mod(system_size, int(nprocs, int64))
        
        do i = 1, nprocs
            if (i <= remainder) then
                partition_table(i + 1) = partition_table(i) + base_rows + 1
            else
                partition_table(i + 1) = partition_table(i) + base_rows
            end if
        end do
    end subroutine generate_partition_table

    !--------------------------------------------------------------------------
    ! Build hypercube adjacency CSR (unit-valued, sorted rows)
    !--------------------------------------------------------------------------
    subroutine build_hypercube_csr(n_qubits, partition_table, &
                                    row_starts, col_indexes, n_local, local_nnz)
        integer, intent(in) :: n_qubits
        integer(int64), intent(in) :: partition_table(:)
        integer(int64), allocatable, intent(out) :: row_starts(:), col_indexes(:)
        integer(int64), intent(out) :: n_local, local_nnz
        
        integer :: rank, ierr, k
        integer(int64) :: lb, ub, i, idx, global_row
        integer(int64) :: temp_cols(n_qubits)
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        
        lb = partition_table(rank + 1)      ! 0-based start
        ub = partition_table(rank + 2) - 1  ! 0-based end (inclusive)
        n_local = ub - lb + 1
        local_nnz = n_local * n_qubits
        
        allocate(row_starts(n_local + 1))
        allocate(col_indexes(local_nnz))
        
        idx = 1
        do i = 1, n_local
            row_starts(i) = idx
            global_row = lb + i - 1  ! 0-based global row
            
            ! Generate neighbors and sort within row
            do k = 1, n_qubits
                temp_cols(k) = ieor(global_row, ishft(1_int64, k - 1))
            end do
            
            ! Simple insertion sort for small array
            call sort_int64(temp_cols)
            
            do k = 1, n_qubits
                col_indexes(idx) = temp_cols(k)
                idx = idx + 1
            end do
        end do
        row_starts(n_local + 1) = idx
    end subroutine build_hypercube_csr

    !--------------------------------------------------------------------------
    ! Sort small int64 array (insertion sort)
    !--------------------------------------------------------------------------
    pure subroutine sort_int64(arr)
        integer(int64), intent(inout) :: arr(:)
        integer :: i, j, n
        integer(int64) :: key
        
        n = size(arr)
        do i = 2, n
            key = arr(i)
            j = i - 1
            do while (j >= 1 .and. arr(j) > key)
                arr(j + 1) = arr(j)
                j = j - 1
            end do
            arr(j + 1) = key
        end do
    end subroutine sort_int64
    
    !--------------------------------------------------------------------------
    ! Sort int64 array with permutation tracking (int64 permutation)
    !--------------------------------------------------------------------------
    pure subroutine sort_with_perm(arr, perm)
        integer(int64), intent(inout) :: arr(:)
        integer(int64), intent(inout) :: perm(:)
        
        integer(int64) :: i, j, n, temp_p
        integer(int64) :: key
        
        n = size(arr)
        ! Insertion sort (fine for moderate sizes)
        do i = 2, n
            key = arr(i)
            temp_p = perm(i)
            j = i - 1
            do while (j >= 1 .and. arr(j) > key)
                arr(j + 1) = arr(j)
                perm(j + 1) = perm(j)
                j = j - 1
            end do
            arr(j + 1) = key
            perm(j + 1) = temp_p
        end do
    end subroutine sort_with_perm

    !--------------------------------------------------------------------------
    ! Merge sort for int64 array - O(n log n)
    !--------------------------------------------------------------------------
    recursive subroutine merge_sort_int64(arr)
        integer(int64), intent(inout) :: arr(:)
        integer :: n, mid
        integer(int64), allocatable :: left(:), right(:)
        
        n = size(arr)
        if (n <= 1) return
        
        mid = n / 2
        allocate(left(mid), right(n - mid))
        left = arr(1:mid)
        right = arr(mid+1:n)
        
        call merge_sort_int64(left)
        call merge_sort_int64(right)
        call merge_arrays(left, right, arr)
        
        deallocate(left, right)
    end subroutine merge_sort_int64
    
    !--------------------------------------------------------------------------
    ! Merge two sorted arrays into one
    !--------------------------------------------------------------------------
    pure subroutine merge_arrays(left, right, arr)
        integer(int64), intent(in) :: left(:), right(:)
        integer(int64), intent(out) :: arr(:)
        integer :: i, j, k
        
        i = 1; j = 1; k = 1
        do while (i <= size(left) .and. j <= size(right))
            if (left(i) <= right(j)) then
                arr(k) = left(i)
                i = i + 1
            else
                arr(k) = right(j)
                j = j + 1
            end if
            k = k + 1
        end do
        
        do while (i <= size(left))
            arr(k) = left(i)
            i = i + 1
            k = k + 1
        end do
        
        do while (j <= size(right))
            arr(k) = right(j)
            j = j + 1
            k = k + 1
        end do
    end subroutine merge_arrays

    !--------------------------------------------------------------------------
    ! Setup graph communicator with O(unique_remote) storage
    !--------------------------------------------------------------------------
    subroutine setup_graph_comm(row_starts, col_indexes, partition_table, &
                                 graph_comm, &
                                 recv_indices_sorted, sort_perm, &
                                 recv_counts, recv_disps, &
                                 send_offsets, send_counts, send_disps, &
                                 in_neighbors, out_neighbors, &
                                 total_recv, total_send, lb, ub)
        integer(int64), intent(in) :: row_starts(:), col_indexes(:)
        integer(int64), intent(in) :: partition_table(:)
        integer, intent(out) :: graph_comm
        integer(int64), allocatable, intent(out) :: recv_indices_sorted(:)
        integer(int64), allocatable, intent(out) :: sort_perm(:)
        integer, allocatable, intent(out) :: recv_counts(:), recv_disps(:)
        integer(int64), allocatable, intent(out) :: send_offsets(:)
        integer, allocatable, intent(out) :: send_counts(:), send_disps(:)
        integer, allocatable, intent(out) :: in_neighbors(:), out_neighbors(:)
        integer(int64), intent(out) :: total_recv, total_send
        integer(int64), intent(out) :: lb, ub
        
        integer :: rank, nprocs, ierr, i, r, owner, n_out, n_in, idx, pos
        integer(int64) :: col, n_local, j
        integer, allocatable :: in_weights(:), out_weights(:)
        integer, allocatable :: in_neighbor_list(:), out_neighbor_list(:)
        integer(int64), allocatable :: all_recv_indices(:), requested(:)
        integer(int64), allocatable :: temp_sort_perm(:)
        logical, allocatable :: is_out_neighbor(:), is_in_neighbor(:)
        integer(int64), allocatable :: seen_cols(:)
        integer(int64) :: n_seen
        logical :: found
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1
        
        ! Step 1: Identify out_neighbors (ranks we need data from)
        allocate(is_out_neighbor(0:nprocs-1))
        is_out_neighbor = .false.
        
        do j = 1, size(col_indexes)
            col = col_indexes(j)
            if (col < lb .or. col > ub) then
                owner = find_owner(col, partition_table)
                is_out_neighbor(owner) = .true.
            end if
        end do
        is_out_neighbor(rank) = .false.
        
        n_out = count(is_out_neighbor)
        allocate(out_neighbor_list(max(n_out, 1)))
        allocate(out_neighbors(max(n_out, 1)))
        idx = 1
        do r = 0, nprocs - 1
            if (is_out_neighbor(r)) then
                out_neighbor_list(idx) = r
                out_neighbors(idx) = r
                idx = idx + 1
            end if
        end do
        
        ! Step 2: Exchange to find in_neighbors
        allocate(is_in_neighbor(0:nprocs-1))
        is_in_neighbor = .false.
        
        call MPI_Alltoall(is_out_neighbor, 1, MPI_LOGICAL, &
                          is_in_neighbor, 1, MPI_LOGICAL, MPI_COMM_WORLD, ierr)
        
        n_in = count(is_in_neighbor)
        allocate(in_neighbor_list(max(n_in, 1)))
        allocate(in_neighbors(max(n_in, 1)))
        idx = 1
        do r = 0, nprocs - 1
            if (is_in_neighbor(r)) then
                in_neighbor_list(idx) = r
                in_neighbors(idx) = r
                idx = idx + 1
            end if
        end do
        
        ! Step 3: Create graph communicator
        allocate(in_weights(max(n_in, 1)), out_weights(max(n_out, 1)))
        in_weights = 1
        out_weights = 1
        
        ! MPI_Dist_graph_create_adjacent(comm, indegree, sources, srcweights, 
        !                                 outdegree, destinations, destweights, ...)
        ! sources = ranks that send TO us = out_neighbor_list (we receive FROM them)
        ! destinations = ranks we send TO = in_neighbor_list (they receive FROM us)
        call MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, &
                n_out, out_neighbor_list, out_weights, &
                n_in, in_neighbor_list, in_weights, &
                MPI_INFO_NULL, .false., graph_comm, ierr)
        
        deallocate(in_weights, out_weights)
        
        ! Step 4: Collect all remote columns, sort, deduplicate - O(nnz log nnz)
        ! First count remote columns
        n_seen = 0
        do j = 1, size(col_indexes)
            col = col_indexes(j)
            if (col < lb .or. col > ub) then
                n_seen = n_seen + 1
            end if
        end do
        
        ! Collect remote columns with their owners
        allocate(seen_cols(max(n_seen, 1)))
        idx = 1
        do j = 1, size(col_indexes)
            col = col_indexes(j)
            if (col < lb .or. col > ub) then
                seen_cols(idx) = col
                idx = idx + 1
            end if
        end do
        
        ! Sort remote columns
        if (n_seen > 1) call merge_sort_int64(seen_cols(1:n_seen))
        
        ! Count unique and build unique list
        if (n_seen > 0) then
            total_recv = 1
            do j = 2, n_seen
                if (seen_cols(j) /= seen_cols(j-1)) then
                    total_recv = total_recv + 1
                end if
            end do
        else
            total_recv = 0
        end if
        
        allocate(all_recv_indices(max(total_recv, 1)))
        if (n_seen > 0) then
            all_recv_indices(1) = seen_cols(1)
            idx = 1
            do j = 2, n_seen
                if (seen_cols(j) /= seen_cols(j-1)) then
                    idx = idx + 1
                    all_recv_indices(idx) = seen_cols(j)
                end if
            end do
        end if
        
        deallocate(seen_cols)
        
        ! Step 5: Count per neighbor and build displacements
        allocate(recv_counts(max(n_out, 1)))
        recv_counts = 0
        
        do j = 1, total_recv
            owner = find_owner(all_recv_indices(j), partition_table)
            ! Find which neighbor index this owner corresponds to
            do i = 1, n_out
                if (out_neighbor_list(i) == owner) then
                    recv_counts(i) = recv_counts(i) + 1
                    exit
                end if
            end do
        end do
        
        allocate(recv_disps(max(n_out, 1)))
        if (n_out > 0) then
            recv_disps(1) = 0
            do i = 2, n_out
                recv_disps(i) = recv_disps(i-1) + recv_counts(i-1)
            end do
        end if
        
        ! Reorder all_recv_indices to be grouped by neighbor
        ! (they're already sorted globally, just need to group by owner)
        block
            integer(int64), allocatable :: temp_indices(:)
            integer, allocatable :: neighbor_pos(:)
            
            allocate(temp_indices(max(total_recv, 1)))
            allocate(neighbor_pos(max(n_out, 1)))
            neighbor_pos = recv_disps + 1  ! Current position for each neighbor
            
            do j = 1, total_recv
                owner = find_owner(all_recv_indices(j), partition_table)
                do i = 1, n_out
                    if (out_neighbor_list(i) == owner) then
                        temp_indices(neighbor_pos(i)) = all_recv_indices(j)
                        neighbor_pos(i) = neighbor_pos(i) + 1
                        exit
                    end if
                end do
            end do
            
            all_recv_indices(1:total_recv) = temp_indices(1:total_recv)
            deallocate(temp_indices, neighbor_pos)
        end block
        
        ! Step 6: Sort recv indices for binary search and create permutation
        allocate(recv_indices_sorted(max(total_recv, 1)))
        allocate(sort_perm(max(total_recv, 1)))
        allocate(temp_sort_perm(max(total_recv, 1)))
        
        ! Initialize permutation
        do i = 1, max(total_recv, 1)
            temp_sort_perm(i) = i
        end do
        if (total_recv > 0) then
            recv_indices_sorted(1:total_recv) = all_recv_indices(1:total_recv)
            
            ! Sort with permutation tracking
            call sort_with_perm(recv_indices_sorted(1:total_recv), temp_sort_perm(1:total_recv))
            
            sort_perm(1:total_recv) = temp_sort_perm(1:total_recv)
        end if
        
        deallocate(temp_sort_perm)
        
        ! Step 7: Exchange counts to set up send side
        ! Use temporary full-sized arrays just for count exchange
        block
            integer, allocatable :: all_recv_counts(:), all_send_counts(:)
            
            allocate(all_recv_counts(nprocs), all_send_counts(nprocs))
            all_recv_counts = 0
            
            ! Fill in counts for our out_neighbors (sources of data for us)
            do i = 1, n_out
                all_recv_counts(out_neighbor_list(i) + 1) = recv_counts(i)
            end do
            
            ! Exchange: send what we need, receive what others need from us
            call MPI_Alltoall(all_recv_counts, 1, MPI_INTEGER, &
                              all_send_counts, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
            
            ! Extract send_counts for our in_neighbors (destinations for our data)
            allocate(send_counts(max(n_in, 1)))
            do i = 1, n_in
                send_counts(i) = all_send_counts(in_neighbor_list(i) + 1)
            end do
            
            deallocate(all_recv_counts, all_send_counts)
        end block
        
        total_send = sum(send_counts)
        
        allocate(send_disps(max(n_in, 1)))
        if (n_in > 0) then
            send_disps(1) = 0
            do i = 2, n_in
                send_disps(i) = send_disps(i-1) + send_counts(i-1)
            end do
        end if
        
        ! Step 8: Exchange indices to know what neighbors need from us
        ! Use temporary full-sized arrays for index exchange
        block
            integer, allocatable :: all_recv_counts(:), all_send_counts(:)
            integer, allocatable :: all_recv_disps(:), all_send_disps(:)
            integer(int64), allocatable :: all_send_indices(:), all_recv_requested(:)
            integer :: total_all_send, total_all_recv
            
            allocate(all_recv_counts(nprocs), all_send_counts(nprocs))
            all_recv_counts = 0
            all_send_counts = 0
            
            ! Build full-sized counts
            do i = 1, n_out
                all_recv_counts(out_neighbor_list(i) + 1) = recv_counts(i)
            end do
            do i = 1, n_in
                all_send_counts(in_neighbor_list(i) + 1) = send_counts(i)
            end do
            
            ! Build displacements
            allocate(all_recv_disps(nprocs), all_send_disps(nprocs))
            all_recv_disps(1) = 0
            all_send_disps(1) = 0
            do i = 2, nprocs
                all_recv_disps(i) = all_recv_disps(i-1) + all_recv_counts(i-1)
                all_send_disps(i) = all_send_disps(i-1) + all_send_counts(i-1)
            end do
            
            total_all_send = sum(all_recv_counts)  ! We send our recv indices
            total_all_recv = sum(all_send_counts)  ! We receive others' requests
            
            ! Pack send buffer: send recv_indices to respective owners
            allocate(all_send_indices(max(total_all_send, 1)))
            do i = 1, n_out
                do j = 1, recv_counts(i)
                    all_send_indices(all_recv_disps(out_neighbor_list(i) + 1) + j) = &
                        all_recv_indices(recv_disps(i) + j)
                end do
            end do
            
            allocate(all_recv_requested(max(total_all_recv, 1)))
            
            ! Use all_recv_counts as sendcounts (what we want = what we send)
            ! Use all_send_counts as recvcounts (what they want from us)
            call MPI_Alltoallv(all_send_indices, all_recv_counts, all_recv_disps, MPI_INTEGER8, &
                               all_recv_requested, all_send_counts, all_send_disps, MPI_INTEGER8, &
                               MPI_COMM_WORLD, ierr)
            
            ! Unpack received indices (what others need from us)
            allocate(requested(max(total_send, 1)))
            do i = 1, n_in
                do j = 1, send_counts(i)
                    requested(send_disps(i) + j) = &
                        all_recv_requested(all_send_disps(in_neighbor_list(i) + 1) + j)
                end do
            end do
            
            deallocate(all_recv_counts, all_send_counts)
            deallocate(all_recv_disps, all_send_disps)
            deallocate(all_send_indices, all_recv_requested)
        end block
        
        ! Convert to local offsets
        allocate(send_offsets(max(total_send, 1)))
        do i = 1, total_send
            send_offsets(i) = requested(i) - lb + 1  ! 1-based local index
        end do
        
        deallocate(all_recv_indices, requested)
        deallocate(is_out_neighbor, is_in_neighbor)
        deallocate(in_neighbor_list, out_neighbor_list)
    end subroutine setup_graph_comm

    !--------------------------------------------------------------------------
    ! Cleanup graph communicator and associated arrays
    !--------------------------------------------------------------------------
    subroutine cleanup_graph_comm(graph_comm, recv_indices_sorted, sort_perm, &
                                   recv_counts, recv_disps, send_offsets, &
                                   send_counts, send_disps, in_neighbors, out_neighbors)
        integer, intent(inout) :: graph_comm
        integer(int64), allocatable, intent(inout) :: recv_indices_sorted(:), send_offsets(:)
        integer(int64), allocatable, intent(inout) :: sort_perm(:)
        integer, allocatable, intent(inout) :: recv_counts(:), recv_disps(:)
        integer, allocatable, intent(inout) :: send_counts(:), send_disps(:)
        integer, allocatable, intent(inout) :: in_neighbors(:), out_neighbors(:)
        
        integer :: ierr
        
        if (graph_comm /= MPI_COMM_NULL) then
            call MPI_Comm_free(graph_comm, ierr)
            graph_comm = MPI_COMM_NULL
        end if
        
        if (allocated(recv_indices_sorted)) deallocate(recv_indices_sorted)
        if (allocated(sort_perm)) deallocate(sort_perm)
        if (allocated(recv_counts)) deallocate(recv_counts)
        if (allocated(recv_disps)) deallocate(recv_disps)
        if (allocated(send_offsets)) deallocate(send_offsets)
        if (allocated(send_counts)) deallocate(send_counts)
        if (allocated(send_disps)) deallocate(send_disps)
        if (allocated(in_neighbors)) deallocate(in_neighbors)
        if (allocated(out_neighbors)) deallocate(out_neighbors)
    end subroutine cleanup_graph_comm

    !--------------------------------------------------------------------------
    ! Build hash table for O(1) average remote column lookup
    ! Call once after setup_graph_comm, before SpMV iterations
    !--------------------------------------------------------------------------
    subroutine build_hash_table(recv_indices_sorted, sort_perm, total_recv, &
                                 hash_keys, hash_vals, hash_size)
        integer(int64), intent(in) :: recv_indices_sorted(:)
        integer(int64), intent(in) :: sort_perm(:)
        integer(int64), intent(in) :: total_recv
        integer(int64), allocatable, intent(out) :: hash_keys(:)
        integer(int64), allocatable, intent(out) :: hash_vals(:)
        integer(int64), intent(out) :: hash_size
        
        integer(int64) :: i, hash_pos, probe
        
        ! Size hash table with load factor ~0.5 for good performance
        hash_size = 2_int64 * total_recv + 1_int64
        ! Make hash_size odd for better distribution
        if (mod(hash_size, 2_int64) == 0) hash_size = hash_size + 1_int64
        
        allocate(hash_keys(hash_size), hash_vals(hash_size))
        hash_keys = -1_int64  ! Empty marker
        hash_vals = 0_int64
        
        ! Build hash table with linear probing
        do i = 1, total_recv
            hash_pos = compute_hash(recv_indices_sorted(i), hash_size)
            do probe = 0, hash_size - 1
                if (hash_keys(hash_pos) < 0) then
                    hash_keys(hash_pos) = recv_indices_sorted(i)
                    hash_vals(hash_pos) = i  ! Position in sorted array
                    exit
                end if
                hash_pos = mod(hash_pos, hash_size) + 1_int64
            end do
        end do
    end subroutine build_hash_table

    !--------------------------------------------------------------------------
    ! SpMV for unit-valued matrix using prebuilt hash table
    ! Optional chunking: if max_recv_chunk > 0 and total_recv > max_recv_chunk,
    ! process recv buffer in chunks to limit peak memory usage
    ! Uses MPI_Ineighbor_alltoallv with graph communicator for memory efficiency
    !--------------------------------------------------------------------------
    subroutine spmv_sorted_rows(row_starts, col_indexes, u, v, scalar, &
                                 graph_comm, &
                                 recv_indices_sorted, sort_perm, &
                                 recv_counts, recv_disps, &
                                 send_offsets, send_counts, send_disps, &
                                 total_recv, total_send, lb, ub, &
                                 send_buf, recv_buf, &
                                 hash_keys, hash_vals, hash_size, &
                                 max_recv_chunk)
        integer(int64), intent(in) :: row_starts(:), col_indexes(:)
        complex(dp), intent(in) :: u(:)
        complex(dp), intent(out) :: v(:)
        complex(dp), intent(in) :: scalar
        integer, intent(in) :: graph_comm
        integer(int64), intent(in) :: recv_indices_sorted(:)
        integer(int64), intent(in) :: sort_perm(:)
        integer, intent(in) :: recv_counts(:), recv_disps(:)
        integer(int64), intent(in) :: send_offsets(:)
        integer, intent(in) :: send_counts(:), send_disps(:)
        integer(int64), intent(in) :: total_recv, total_send
        integer(int64), intent(in) :: lb, ub
        complex(dp), intent(inout) :: send_buf(:), recv_buf(:)
        integer(int64), intent(in) :: hash_keys(:)
        integer(int64), intent(in) :: hash_vals(:)
        integer(int64), intent(in) :: hash_size
        integer(int64), intent(in), optional :: max_recv_chunk
        
        integer :: ierr, request
        integer(int64) :: i, n_local, col, start_j, end_j, j
        integer(int64) :: local_start, local_end
        integer(int64) :: hash_pos, probe
        complex(dp) :: row_sum
        integer :: status(MPI_STATUS_SIZE)
        complex(dp), allocatable :: recv_buf_sorted(:)
        
        ! Chunking variables
        integer(int64) :: chunk_size, chunk_start, chunk_end, n_chunks, chunk
        integer(int64) :: actual_chunk_size
        logical :: use_chunking
        
        n_local = ub - lb + 1
        
        ! Determine if chunking is needed
        use_chunking = .false.
        if (present(max_recv_chunk)) then
            if (max_recv_chunk > 0 .and. total_recv > max_recv_chunk) then
                use_chunking = .true.
                chunk_size = max_recv_chunk
            end if
        end if
        
        ! Pack send buffer
        !$omp parallel do
        do i = 1, total_send
            send_buf(i) = u(send_offsets(i))
        end do
        !$omp end parallel do
        
        ! Start non-blocking exchange using graph communicator
        ! Graph: sources = out_neighbors (indegree), destinations = in_neighbors (outdegree)
        ! sendbuf is sent TO destinations, indexed by outdegree order (send_counts/disps)
        ! recvbuf receives FROM sources, indexed by indegree order (recv_counts/disps)
        call MPI_Ineighbor_alltoallv(send_buf, send_counts, send_disps, MPI_DOUBLE_COMPLEX, &
                                     recv_buf, recv_counts, recv_disps, MPI_DOUBLE_COMPLEX, &
                                     graph_comm, request, ierr)
        
        ! Compute LOCAL contributions while communication proceeds
        !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end)
        do i = 1, n_local
            start_j = row_starts(i)
            end_j = row_starts(i + 1) - 1
            row_sum = (0.0_dp, 0.0_dp)
            
            ! Find local range boundaries using binary search
            local_start = lower_bound(col_indexes, start_j, end_j, lb)
            local_end = upper_bound(col_indexes, start_j, end_j, ub) - 1
            
            ! Sum local columns (contiguous range)
            do j = local_start, local_end
                col = col_indexes(j)
                row_sum = row_sum + u(col - lb + 1)
            end do
            
            v(i) = row_sum
        end do
        !$omp end parallel do
        
        ! Wait for communication to complete
        call MPI_Wait(request, status, ierr)
        
        if (.not. use_chunking) then
            ! Fast path: process all remote data at once
            call process_remote_unchunked()
        else
            ! Memory-bounded path: process recv buffer in chunks
            call process_remote_chunked()
        end if
        
    contains
    
        subroutine process_remote_unchunked()
            integer(int64) :: sorted_pos
            
            ! Allocate full recv_buf_sorted
            allocate(recv_buf_sorted(max(total_recv, 1_int64)))
            
            ! Reorder recv_buf to sorted order
            !$omp parallel do
            do i = 1, total_recv
                recv_buf_sorted(i) = recv_buf(sort_perm(i))
            end do
            !$omp end parallel do
            
            ! Add REMOTE contributions using prebuilt hash table
            !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end, sorted_pos)
            do i = 1, n_local
                start_j = row_starts(i)
                end_j = row_starts(i + 1) - 1
                
                ! Quick check: if row has no non-local columns, skip
                if (col_indexes(start_j) >= lb .and. col_indexes(end_j) <= ub) then
                    v(i) = scalar * v(i)
                    cycle
                end if
                
                row_sum = v(i)
                
                ! Find local boundaries
                local_start = lower_bound(col_indexes, start_j, end_j, lb)
                local_end = upper_bound(col_indexes, start_j, end_j, ub) - 1
                
                ! Remote columns before local range
                do j = start_j, local_start - 1
                    col = col_indexes(j)
                    sorted_pos = hash_lookup(col, hash_keys, hash_vals, hash_size)
                    if (sorted_pos > 0) row_sum = row_sum + recv_buf_sorted(sorted_pos)
                end do
                
                ! Remote columns after local range
                do j = local_end + 1, end_j
                    col = col_indexes(j)
                    sorted_pos = hash_lookup(col, hash_keys, hash_vals, hash_size)
                    if (sorted_pos > 0) row_sum = row_sum + recv_buf_sorted(sorted_pos)
                end do
                
                v(i) = scalar * row_sum
            end do
            !$omp end parallel do
            
            deallocate(recv_buf_sorted)
        end subroutine process_remote_unchunked
        
        subroutine process_remote_chunked()
            integer :: sorted_idx
            
            ! Calculate number of chunks
            n_chunks = (total_recv + chunk_size - 1) / chunk_size
            
            ! Allocate chunk-sized buffer (reused across chunks)
            allocate(recv_buf_sorted(chunk_size))
            
            ! Process each chunk
            do chunk = 1, n_chunks
                chunk_start = (chunk - 1) * chunk_size + 1
                chunk_end = min(chunk * chunk_size, total_recv)
                actual_chunk_size = chunk_end - chunk_start + 1
                
                ! Reorder this chunk of recv_buf to sorted order
                !$omp parallel do private(sorted_idx)
                do i = 1, actual_chunk_size
                    sorted_idx = chunk_start + i - 1
                    recv_buf_sorted(i) = recv_buf(sort_perm(sorted_idx))
                end do
                !$omp end parallel do
                
                ! Add contributions from this chunk to all rows
                ! On first chunk, don't apply scalar yet; on last chunk, apply scalar
                if (chunk == 1) then
                    call add_chunk_contributions(chunk_start, chunk_end, .false.)
                else if (chunk == n_chunks) then
                    call add_chunk_contributions(chunk_start, chunk_end, .true.)
                else
                    call add_chunk_contributions(chunk_start, chunk_end, .false.)
                end if
            end do
            
            deallocate(recv_buf_sorted)
        end subroutine process_remote_chunked
        
        subroutine add_chunk_contributions(chunk_start, chunk_end, apply_scalar)
            integer(int64), intent(in) :: chunk_start, chunk_end
            logical, intent(in) :: apply_scalar
            
            integer(int64) :: row_idx, col_idx
            integer(int64) :: row_start_j, row_end_j
            integer(int64) :: local_start_j, local_end_j
            integer(int64) :: min_col, max_col
            integer(int64) :: remote_start, remote_end
            integer(int64) :: sorted_pos
            complex(dp) :: row_sum
            
            ! Column range for this chunk
            min_col = recv_indices_sorted(chunk_start)
            max_col = recv_indices_sorted(chunk_end)
            
            !$omp parallel do private(row_start_j, row_end_j, row_sum, col_idx, &
            !$omp&    local_start_j, local_end_j, remote_start, remote_end, sorted_pos)
            do row_idx = 1, n_local
                row_start_j = row_starts(row_idx)
                row_end_j = row_starts(row_idx + 1) - 1
                
                ! Quick check: if row has no non-local columns, apply scalar on last chunk
                if (col_indexes(row_start_j) >= lb .and. col_indexes(row_end_j) <= ub) then
                    if (apply_scalar) v(row_idx) = scalar * v(row_idx)
                    cycle
                end if
                
                row_sum = (0.0_dp, 0.0_dp)
                
                ! Find local boundaries
                local_start_j = lower_bound(col_indexes, row_start_j, row_end_j, lb)
                local_end_j = upper_bound(col_indexes, row_start_j, row_end_j, ub) - 1
                
                ! Remote columns before local range - narrow to [min_col, max_col]
                if (row_start_j <= local_start_j - 1) then
                    remote_start = lower_bound(col_indexes, row_start_j, local_start_j - 1, min_col)
                    remote_end = upper_bound(col_indexes, row_start_j, local_start_j - 1, max_col)
                    
                    do col_idx = remote_start, remote_end - 1
                        sorted_pos = hash_lookup(col_indexes(col_idx), hash_keys, hash_vals, hash_size)
                        if (sorted_pos > 0) then
                            row_sum = row_sum + recv_buf_sorted(sorted_pos - chunk_start + 1)
                        end if
                    end do
                end if
                
                ! Remote columns after local range - narrow to [min_col, max_col]
                if (local_end_j + 1 <= row_end_j) then
                    remote_start = lower_bound(col_indexes, local_end_j + 1, row_end_j, min_col)
                    remote_end = upper_bound(col_indexes, local_end_j + 1, row_end_j, max_col)
                    
                    do col_idx = remote_start, remote_end - 1
                        sorted_pos = hash_lookup(col_indexes(col_idx), hash_keys, hash_vals, hash_size)
                        if (sorted_pos > 0) then
                            row_sum = row_sum + recv_buf_sorted(sorted_pos - chunk_start + 1)
                        end if
                    end do
                end if
                
                v(row_idx) = v(row_idx) + row_sum
                if (apply_scalar) v(row_idx) = scalar * v(row_idx)
            end do
            !$omp end parallel do
        end subroutine add_chunk_contributions
        
    end subroutine spmv_sorted_rows

end module chunked_spmv_mod