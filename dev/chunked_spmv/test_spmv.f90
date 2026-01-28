!------------------------------------------------------------------------------
! Memory-efficient SpMV for unit-valued matrices using MPI graph communicators
!
! This standalone test program implements:
! 1. Graph communicator setup for O(neighbors) scaling
! 2. Sorted-rows SpMV with O(unique_remote) storage
! 3. Chunked SpMV for bounded memory at extreme scale
!
! Compile: mpif90 -O3 -o test_spmv test_spmv.f90
! Run: mpirun -np 4 ./test_spmv 16
!------------------------------------------------------------------------------

module chunked_spmv_mod
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    implicit none
    
contains

    !--------------------------------------------------------------------------
    ! Binary search in sorted array - returns position or 0 if not found
    ! Takes explicit bounds to avoid array slice temporaries
    !--------------------------------------------------------------------------
    pure function binary_search(arr, n, val) result(pos)
        integer(int64), intent(in) :: arr(*)  ! Assumed-size to avoid copy
        integer, intent(in) :: n              ! Number of elements to search
        integer(int64), intent(in) :: val
        integer :: pos
        
        integer :: lo, hi, mid
        
        lo = 1
        hi = n
        pos = 0  ! Not found
        
        do while (lo <= hi)
            mid = (lo + hi) / 2
            if (arr(mid) == val) then
                pos = mid
                return
            else if (arr(mid) < val) then
                lo = mid + 1
            else
                hi = mid - 1
            end if
        end do
    end function binary_search
    
    !--------------------------------------------------------------------------
    ! Find first position where arr(pos) >= val (lower bound)
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
    ! Sort int64 array with permutation tracking
    !--------------------------------------------------------------------------
    subroutine sort_with_perm(arr, perm)
        integer(int64), intent(inout) :: arr(:)
        integer, intent(inout) :: perm(:)
        
        integer :: i, j, n, temp_p
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
    
    subroutine merge_arrays(left, right, arr)
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
        integer, allocatable, intent(out) :: sort_perm(:)
        integer, allocatable, intent(out) :: recv_counts(:), recv_disps(:)
        integer(int64), allocatable, intent(out) :: send_offsets(:)
        integer, allocatable, intent(out) :: send_counts(:), send_disps(:)
        integer, allocatable, intent(out) :: in_neighbors(:), out_neighbors(:)
        integer, intent(out) :: total_recv, total_send
        integer(int64), intent(out) :: lb, ub
        
        integer :: rank, nprocs, ierr, i, j, r, owner, n_out, n_in, idx, pos
        integer(int64) :: col, n_local
        integer, allocatable :: in_weights(:), out_weights(:)
        integer, allocatable :: in_neighbor_list(:), out_neighbor_list(:)
        integer(int64), allocatable :: all_recv_indices(:), requested(:)
        integer, allocatable :: temp_sort_perm(:)
        logical, allocatable :: is_out_neighbor(:), is_in_neighbor(:)
        integer(int64), allocatable :: seen_cols(:)
        integer :: n_seen
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
        
        call MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, &
                n_in, in_neighbor_list, in_weights, &
                n_out, out_neighbor_list, out_weights, &
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
        allocate(send_counts(max(n_in, 1)))
        call MPI_Neighbor_alltoall(recv_counts, 1, MPI_INTEGER, &
                                   send_counts, 1, MPI_INTEGER, graph_comm, ierr)
        
        total_send = sum(send_counts)
        
        allocate(send_disps(max(n_in, 1)))
        if (n_in > 0) then
            send_disps(1) = 0
            do i = 2, n_in
                send_disps(i) = send_disps(i-1) + send_counts(i-1)
            end do
        end if
        
        ! Step 8: Exchange indices to know what neighbors need from us
        allocate(requested(max(total_send, 1)))
        call MPI_Neighbor_alltoallv(all_recv_indices, recv_counts, recv_disps, MPI_INTEGER8, &
                                    requested, send_counts, send_disps, MPI_INTEGER8, &
                                    graph_comm, ierr)
        
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
    ! SpMV for unit-valued matrix using sorted rows and binary search
    ! Optimized: exploits sorted rows to find local/remote boundaries once
    !--------------------------------------------------------------------------
    subroutine spmv_sorted_rows(row_starts, col_indexes, u, v, scalar, &
                                 graph_comm, &
                                 recv_indices_sorted, sort_perm, &
                                 recv_counts, recv_disps, &
                                 send_offsets, send_counts, send_disps, &
                                 total_recv, total_send, lb, ub, &
                                 send_buf, recv_buf)
        integer(int64), intent(in) :: row_starts(:), col_indexes(:)
        complex(dp), intent(in) :: u(:)
        complex(dp), intent(out) :: v(:)
        complex(dp), intent(in) :: scalar
        integer, intent(in) :: graph_comm
        integer(int64), intent(in) :: recv_indices_sorted(:)
        integer, intent(in) :: sort_perm(:)
        integer, intent(in) :: recv_counts(:), recv_disps(:)
        integer(int64), intent(in) :: send_offsets(:)
        integer, intent(in) :: send_counts(:), send_disps(:)
        integer, intent(in) :: total_recv, total_send
        integer(int64), intent(in) :: lb, ub
        complex(dp), intent(inout) :: send_buf(:), recv_buf(:)
        
        integer :: ierr, i, sorted_pos, recv_pos, request
        integer(int64) :: n_local, col, start_j, end_j, j
        integer(int64) :: local_start, local_end  ! Boundaries in sorted row
        complex(dp) :: row_sum
        integer :: status(MPI_STATUS_SIZE)
        
        n_local = ub - lb + 1
        
        ! Pack send buffer
        !$omp parallel do
        do i = 1, total_send
            send_buf(i) = u(send_offsets(i))
        end do
        !$omp end parallel do
        
        ! Start non-blocking exchange
        call MPI_Ineighbor_alltoallv(send_buf, send_counts, send_disps, MPI_DOUBLE_COMPLEX, &
                                     recv_buf, recv_counts, recv_disps, MPI_DOUBLE_COMPLEX, &
                                     graph_comm, request, ierr)
        
        ! Compute LOCAL contributions while communication proceeds
        ! Exploit sorted rows: find local range [lb, ub] with binary search
        !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end)
        do i = 1, n_local
            start_j = row_starts(i)
            end_j = row_starts(i + 1) - 1
            row_sum = (0.0_dp, 0.0_dp)
            
            ! Find first column >= lb (start of local range)
            local_start = lower_bound(col_indexes, start_j, end_j, lb)
            ! Find first column > ub (end of local range)  
            local_end = upper_bound(col_indexes, start_j, end_j, ub) - 1
            
            ! Sum local columns (contiguous range, no conditionals)
            do j = local_start, local_end
                col = col_indexes(j)
                row_sum = row_sum + u(col - lb + 1)
            end do
            
            v(i) = row_sum
        end do
        !$omp end parallel do
        
        ! Wait for communication to complete
        call MPI_Wait(request, status, ierr)
        
        ! Add REMOTE contributions (columns before local_start and after local_end)
        !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end, sorted_pos, recv_pos)
        do i = 1, n_local
            start_j = row_starts(i)
            end_j = row_starts(i + 1) - 1
            row_sum = v(i)
            
            ! Find local boundaries again
            local_start = lower_bound(col_indexes, start_j, end_j, lb)
            local_end = upper_bound(col_indexes, start_j, end_j, ub) - 1
            
            ! Remote columns before local range
            do j = start_j, local_start - 1
                col = col_indexes(j)
                sorted_pos = binary_search(recv_indices_sorted, total_recv, col)
                if (sorted_pos > 0) then
                    recv_pos = sort_perm(sorted_pos)
                    row_sum = row_sum + recv_buf(recv_pos)
                end if
            end do
            
            ! Remote columns after local range
            do j = local_end + 1, end_j
                col = col_indexes(j)
                sorted_pos = binary_search(recv_indices_sorted, total_recv, col)
                if (sorted_pos > 0) then
                    recv_pos = sort_perm(sorted_pos)
                    row_sum = row_sum + recv_buf(recv_pos)
                end if
            end do
            
            v(i) = scalar * row_sum
        end do
        !$omp end parallel do
        
    end subroutine spmv_sorted_rows

end module chunked_spmv_mod


!==============================================================================
! Main test program
!==============================================================================
program test_chunked_spmv
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use chunked_spmv_mod
    implicit none

    integer :: ierr, rank, nprocs
    integer :: n_qubits
    character(len=32) :: arg
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    ! Parse command line
    n_qubits = 14
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg)
        read(arg, *) n_qubits
    end if
    
    if (rank == 0) then
        print '(A)', '================================================='
        print '(A)', 'Memory-Efficient SpMV for Unit-Valued Matrices'
        print '(A)', '================================================='
        print '(A,I0)', 'MPI ranks: ', nprocs
        print '(A,I0)', 'Qubits: ', n_qubits
        print '(A,I0)', 'System size: ', 2_int64**n_qubits
        print '(A)', ''
    end if
    
    call test_correctness(n_qubits)
    call benchmark(n_qubits, 10)
    
    call MPI_Finalize(ierr)

contains

    !--------------------------------------------------------------------------
    ! Test correctness
    !--------------------------------------------------------------------------
    subroutine test_correctness(n_qubits)
        integer, intent(in) :: n_qubits
        
        integer :: rank, nprocs, ierr, k
        integer(int64) :: system_size, n_local, local_nnz, lb, ub, i
        integer(int64), allocatable :: partition_table(:), row_starts(:), col_indexes(:)
        complex(dp), allocatable :: u(:), v(:), expected(:)
        complex(dp), allocatable :: send_buf(:), recv_buf(:)
        integer :: graph_comm, total_recv, total_send
        integer(int64), allocatable :: recv_indices_sorted(:), send_offsets(:)
        integer, allocatable :: sort_perm(:), recv_counts(:), recv_disps(:)
        integer, allocatable :: send_counts(:), send_disps(:)
        integer, allocatable :: in_neighbors(:), out_neighbors(:)
        logical :: passed
        real(dp) :: max_err
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        system_size = 2_int64**n_qubits
        
        ! Generate partition and CSR
        call generate_partition_table(system_size, partition_table)
        call build_hypercube_csr(n_qubits, partition_table, row_starts, col_indexes, n_local, local_nnz)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        
        ! Setup graph communicator
        call setup_graph_comm(row_starts, col_indexes, partition_table, &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              in_neighbors, out_neighbors, total_recv, total_send, lb, ub)
        
        ! Preallocate buffers
        allocate(u(n_local), v(n_local), expected(n_local))
        allocate(send_buf(max(total_send, 1)), recv_buf(max(total_recv, 1)))
        
        ! Test 1: All-ones vector
        u = (1.0_dp, 0.0_dp)
        expected = cmplx(n_qubits, 0.0_dp, dp)
        
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf)
        
        max_err = maxval(abs(v - expected))
        passed = max_err < 1.0e-10_dp
        
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Test 1 (all ones): PASS'
            else
                print '(A,ES12.4)', 'Test 1 (all ones): FAIL, max_err=', max_err
            end if
        end if
        
        ! Test 2: Index vector
        do i = 1, n_local
            u(i) = cmplx(lb + i - 1, 0.0_dp, dp)
        end do
        
        expected = (0.0_dp, 0.0_dp)
        do i = 1, n_local
            do k = 1, n_qubits
                expected(i) = expected(i) + cmplx(ieor(lb + i - 1, ishft(1_int64, k - 1)), 0.0_dp, dp)
            end do
        end do
        
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf)
        
        max_err = maxval(abs(v - expected))
        passed = max_err < 1.0e-10_dp
        
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Test 2 (index vector): PASS'
            else
                print '(A,ES12.4)', 'Test 2 (index vector): FAIL, max_err=', max_err
            end if
        end if
        
        ! Test 3: Scalar multiplier
        u = (1.0_dp, 0.0_dp)
        expected = cmplx(0.0_dp, -real(n_qubits, dp), dp)  ! -i * n_qubits
        
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (0.0_dp, -1.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf)
        
        max_err = maxval(abs(v - expected))
        passed = max_err < 1.0e-10_dp
        
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Test 3 (scalar -i): PASS'
            else
                print '(A,ES12.4)', 'Test 3 (scalar -i): FAIL, max_err=', max_err
            end if
        end if
        
        ! Cleanup
        deallocate(u, v, expected, send_buf, recv_buf)
        deallocate(partition_table, row_starts, col_indexes)
        deallocate(recv_indices_sorted, sort_perm, recv_counts, recv_disps)
        deallocate(send_offsets, send_counts, send_disps, in_neighbors, out_neighbors)
        call MPI_Comm_free(graph_comm, ierr)
    end subroutine test_correctness

    !--------------------------------------------------------------------------
    ! Benchmark
    !--------------------------------------------------------------------------
    subroutine benchmark(n_qubits, n_iters)
        integer, intent(in) :: n_qubits, n_iters
        
        integer :: rank, nprocs, ierr, iter
        integer(int64) :: system_size, n_local, local_nnz, lb, ub
        integer(int64), allocatable :: partition_table(:), row_starts(:), col_indexes(:)
        complex(dp), allocatable :: u(:), v(:)
        complex(dp), allocatable :: send_buf(:), recv_buf(:)
        integer :: graph_comm, total_recv, total_send
        integer(int64), allocatable :: recv_indices_sorted(:), send_offsets(:)
        integer, allocatable :: sort_perm(:), recv_counts(:), recv_disps(:)
        integer, allocatable :: send_counts(:), send_disps(:)
        integer, allocatable :: in_neighbors(:), out_neighbors(:)
        real(dp) :: t_start, t_end, t_setup, t_spmv
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        system_size = 2_int64**n_qubits
        
        ! Setup timing
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        call generate_partition_table(system_size, partition_table)
        call build_hypercube_csr(n_qubits, partition_table, row_starts, col_indexes, n_local, local_nnz)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        
        call setup_graph_comm(row_starts, col_indexes, partition_table, &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              in_neighbors, out_neighbors, total_recv, total_send, lb, ub)
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_setup = MPI_Wtime() - t_start
        
        ! Allocate and initialize (buffers allocated once, reused)
        allocate(u(n_local), v(n_local))
        allocate(send_buf(max(total_send, 1)), recv_buf(max(total_recv, 1)))
        u = (1.0_dp, 0.0_dp)
        
        ! Warmup
        do iter = 1, 2
            call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                                  graph_comm, recv_indices_sorted, sort_perm, &
                                  recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                                  total_recv, total_send, lb, ub, send_buf, recv_buf)
        end do
        
        ! Timed runs
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        do iter = 1, n_iters
            call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                                  graph_comm, recv_indices_sorted, sort_perm, &
                                  recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                                  total_recv, total_send, lb, ub, send_buf, recv_buf)
        end do
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_spmv = MPI_Wtime() - t_start
        
        if (rank == 0) then
            print '(A)', ''
            print '(A,I0,A,I0,A,I0,A)', '=== Benchmark: ', n_qubits, ' qubits, ', nprocs, ' ranks, ', n_iters, ' iterations ==='
            print '(A,F10.2,A)', 'Setup time:     ', t_setup * 1000, ' ms'
            print '(A,F10.2,A)', 'SpMV time:      ', (t_spmv / n_iters) * 1000, ' ms/iter'
            print '(A)', ''
            print '(A,I0)', 'Local rows:     ', n_local
            print '(A,I0)', 'Local NNZ:      ', local_nnz
            print '(A,I0)', 'Total recv:     ', total_recv
            print '(A,I0)', 'Total send:     ', total_send
            print '(A,I0)', 'In neighbors:   ', size(in_neighbors)
            print '(A,I0)', 'Out neighbors:  ', size(out_neighbors)
            print '(A)', ''
            print '(A,F10.4,A)', 'Recv buffer:    ', real(total_recv) * 16 / 1e6, ' MB'
            print '(A,F10.4,A)', 'Send buffer:    ', real(total_send) * 16 / 1e6, ' MB'
            print '(A,F10.4,A)', 'Comm data:      ', real(total_recv) * 12 / 1e6, ' MB (indices + perm)'
        end if
        
        ! Cleanup
        deallocate(u, v, send_buf, recv_buf)
        deallocate(partition_table, row_starts, col_indexes)
        deallocate(recv_indices_sorted, sort_perm, recv_counts, recv_disps)
        deallocate(send_offsets, send_counts, send_disps, in_neighbors, out_neighbors)
        call MPI_Comm_free(graph_comm, ierr)
    end subroutine benchmark

end program test_chunked_spmv
