!   QSW_MPI -  A package for parallel Quantum Stochastic Walk simulation.


!   Copyright (C) 2019 Edric Matwiejew
!
!   This program is free software: you can redistribute it and/or modify
!   it under the terms of the GNU General Public License as published by
!   the Free Software Foundation, either version 3 of the License, or
!   (at your option) any later version.
!
!   This program is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!   GNU General Public License for more details.
!
!   You should have received a copy of the GNU General Public License
!   along with this program.  If not, see <https://www.gnu.org/licenses/>.

!
!   Module: Sparse_Operations
!
!> @brief MPI parallel sparse BLAS operations.
!
module Sparse

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, qp => real128, int64
    use :: MPI

    implicit none
    
    ! Hash table constants (Knuth's golden ratio multiplier)
    integer(int64), parameter, private :: HASH_MULT = 2654435769_int64
    integer(int64), parameter, private :: MASK32 = int(Z'FFFFFFFF', int64)

    !> @brief Compressed sparse rows (CSR) complex matrix derived type.
    !
    !> @warning *Sparse_Operations.mod* requiers that the entries for each row are
    !> stored in acessending order. This condition may not be enforced
    !> external sparse libraries.

    type, public  :: CSR

        integer :: rows
        integer :: columns
        character(len=2) :: structure
        integer(dp), dimension(:), pointer :: row_starts => null()
        integer(dp), dimension(:), pointer :: col_indexes => null()
        complex(dp), dimension(:), pointer :: values => null()

        ! Graph communicator data for O(neighbors) SpMV
        integer :: graph_comm = MPI_COMM_NULL
        integer(int64), dimension(:), allocatable :: recv_indices_sorted
        integer(int64), dimension(:), allocatable :: sort_perm
        integer, dimension(:), allocatable :: graph_recv_counts
        integer, dimension(:), allocatable :: graph_recv_disps
        integer(int64), dimension(:), allocatable :: send_offsets
        integer, dimension(:), allocatable :: graph_send_counts
        integer, dimension(:), allocatable :: graph_send_disps
        integer, dimension(:), allocatable :: in_neighbors
        integer, dimension(:), allocatable :: out_neighbors
        integer(int64) :: total_recv = 0
        integer(int64) :: total_send = 0
        integer(int64) :: lb_graph = 0
        integer(int64) :: ub_graph = 0
        ! Local copies of row_starts and col_indexes (1-based indexing for spmv)
        integer(int64), dimension(:), allocatable :: row_starts_local
        integer(int64), dimension(:), allocatable :: col_indexes_local
        ! Hash table for O(1) remote column lookup
        integer(int64), dimension(:), allocatable :: hash_keys
        integer(int64), dimension(:), allocatable :: hash_vals
        integer(int64) :: hash_size = 0
        ! Persistent communication buffers
        complex(dp), dimension(:), allocatable :: send_buf
        complex(dp), dimension(:), allocatable :: recv_buf
        ! Local values for SpMV (slice of global values)
        complex(dp), dimension(:), allocatable :: values_local
        ! Flag to indicate if graph comm is set up
        logical :: graph_comm_ready = .false.

    end type CSR

    contains

    !--------------------------------------------------------------------------
    ! Graph Communicator SpMV Helper Functions
    ! (Integrated from chunked_spmv_mod for self-contained module)
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    ! Find first position where arr(pos) >= val (lower bound)
    !--------------------------------------------------------------------------
    pure function lower_bound(arr, lo_in, hi_in, val) result(pos)
        integer(int64), intent(in) :: arr(*)
        integer(int64), intent(in) :: lo_in, hi_in, val
        integer(int64) :: pos
        integer(int64) :: lo, hi, mid
        
        lo = lo_in
        hi = hi_in + 1
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
    ! Compute hash index for a column
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
    ! Look up a column in the hash table
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
                return
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
        owner = lo - 1
    end function find_owner

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
    pure subroutine sort_with_perm(arr, perm)
        integer(int64), intent(inout) :: arr(:)
        integer(int64), intent(inout) :: perm(:)
        integer(int64) :: i, j, n, temp_p, key
        
        n = size(arr)
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
        call merge_arrays_int64(left, right, arr)
        
        deallocate(left, right)
    end subroutine merge_sort_int64
    
    !--------------------------------------------------------------------------
    ! Merge two sorted arrays into one
    !--------------------------------------------------------------------------
    pure subroutine merge_arrays_int64(left, right, arr)
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
    end subroutine merge_arrays_int64

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
                n_out, out_neighbor_list, out_weights, &
                n_in, in_neighbor_list, in_weights, &
                MPI_INFO_NULL, .false., graph_comm, ierr)
        
        deallocate(in_weights, out_weights)
        
        ! Step 4: Collect all remote columns, sort, deduplicate
        n_seen = 0
        do j = 1, size(col_indexes)
            col = col_indexes(j)
            if (col < lb .or. col > ub) then
                n_seen = n_seen + 1
            end if
        end do
        
        allocate(seen_cols(max(n_seen, 1)))
        idx = 1
        do j = 1, size(col_indexes)
            col = col_indexes(j)
            if (col < lb .or. col > ub) then
                seen_cols(idx) = col
                idx = idx + 1
            end if
        end do
        
        if (n_seen > 1) call merge_sort_int64(seen_cols(1:n_seen))
        
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
        block
            integer(int64), allocatable :: temp_indices(:)
            integer, allocatable :: neighbor_pos(:)
            
            allocate(temp_indices(max(total_recv, 1)))
            allocate(neighbor_pos(max(n_out, 1)))
            neighbor_pos = recv_disps + 1
            
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
        
        do i = 1, max(total_recv, 1)
            temp_sort_perm(i) = i
        end do
        if (total_recv > 0) then
            recv_indices_sorted(1:total_recv) = all_recv_indices(1:total_recv)
            call sort_with_perm(recv_indices_sorted(1:total_recv), temp_sort_perm(1:total_recv))
            sort_perm(1:total_recv) = temp_sort_perm(1:total_recv)
        end if
        
        deallocate(temp_sort_perm)
        
        ! Step 7: Exchange counts to set up send side
        block
            integer, allocatable :: all_recv_counts(:), all_send_counts(:)
            
            allocate(all_recv_counts(nprocs), all_send_counts(nprocs))
            all_recv_counts = 0
            
            do i = 1, n_out
                all_recv_counts(out_neighbor_list(i) + 1) = recv_counts(i)
            end do
            
            call MPI_Alltoall(all_recv_counts, 1, MPI_INTEGER, &
                              all_send_counts, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
            
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
        block
            integer, allocatable :: all_recv_counts(:), all_send_counts(:)
            integer, allocatable :: all_recv_disps(:), all_send_disps(:)
            integer(int64), allocatable :: all_send_indices(:), all_recv_requested(:)
            integer :: total_all_send, total_all_recv
            
            allocate(all_recv_counts(nprocs), all_send_counts(nprocs))
            all_recv_counts = 0
            all_send_counts = 0
            
            do i = 1, n_out
                all_recv_counts(out_neighbor_list(i) + 1) = recv_counts(i)
            end do
            do i = 1, n_in
                all_send_counts(in_neighbor_list(i) + 1) = send_counts(i)
            end do
            
            allocate(all_recv_disps(nprocs), all_send_disps(nprocs))
            all_recv_disps(1) = 0
            all_send_disps(1) = 0
            do i = 2, nprocs
                all_recv_disps(i) = all_recv_disps(i-1) + all_recv_counts(i-1)
                all_send_disps(i) = all_send_disps(i-1) + all_send_counts(i-1)
            end do
            
            total_all_send = sum(all_recv_counts)
            total_all_recv = sum(all_send_counts)
            
            allocate(all_send_indices(max(total_all_send, 1)))
            do i = 1, n_out
                do j = 1, recv_counts(i)
                    all_send_indices(all_recv_disps(out_neighbor_list(i) + 1) + j) = &
                        all_recv_indices(recv_disps(i) + j)
                end do
            end do
            
            allocate(all_recv_requested(max(total_all_recv, 1)))
            
            call MPI_Alltoallv(all_send_indices, all_recv_counts, all_recv_disps, MPI_INTEGER8, &
                               all_recv_requested, all_send_counts, all_send_disps, MPI_INTEGER8, &
                               MPI_COMM_WORLD, ierr)
            
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
        
        allocate(send_offsets(max(total_send, 1)))
        do i = 1, total_send
            send_offsets(i) = requested(i) - lb + 1
        end do
        
        deallocate(all_recv_indices, requested)
        deallocate(is_out_neighbor, is_in_neighbor)
        deallocate(in_neighbor_list, out_neighbor_list)
    end subroutine setup_graph_comm

    !--------------------------------------------------------------------------
    ! Build hash table for O(1) average remote column lookup
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
        
        hash_size = 2_int64 * total_recv + 1_int64
        if (mod(hash_size, 2_int64) == 0) hash_size = hash_size + 1_int64
        
        allocate(hash_keys(hash_size), hash_vals(hash_size))
        hash_keys = -1_int64
        hash_vals = 0_int64
        
        do i = 1, total_recv
            hash_pos = compute_hash(recv_indices_sorted(i), hash_size)
            do probe = 0, hash_size - 1
                if (hash_keys(hash_pos) < 0) then
                    hash_keys(hash_pos) = recv_indices_sorted(i)
                    hash_vals(hash_pos) = i
                    exit
                end if
                hash_pos = mod(hash_pos, hash_size) + 1_int64
            end do
        end do
    end subroutine build_hash_table

    !--------------------------------------------------------------------------
    ! SpMV using graph communicator and prebuilt hash table
    !--------------------------------------------------------------------------
    subroutine spmv(row_starts, col_indexes, u, v, scalar, &
                                 graph_comm, &
                                 recv_indices_sorted, sort_perm, &
                                 recv_counts, recv_disps, &
                                 send_offsets, send_counts, send_disps, &
                                 total_recv, total_send, lb, ub, &
                                 send_buf, recv_buf, &
                                 hash_keys, hash_vals, hash_size, &
                                 values)
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
        complex(dp), intent(in) :: values(:)
        
        integer :: ierr, request
        integer(int64) :: i, n_local, col, start_j, end_j, j
        integer(int64) :: local_start, local_end, sorted_pos
        complex(dp) :: row_sum
        integer :: status(MPI_STATUS_SIZE)
        complex(dp), allocatable :: recv_buf_sorted(:)
        
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
        !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end)
        do i = 1, n_local
            start_j = row_starts(i)
            end_j = row_starts(i + 1) - 1
            row_sum = (0.0_dp, 0.0_dp)
            
            local_start = lower_bound(col_indexes, start_j, end_j, lb)
            local_end = upper_bound(col_indexes, start_j, end_j, ub) - 1
            
            do j = local_start, local_end
                col = col_indexes(j)
                row_sum = row_sum + values(j) * u(col - lb + 1)
            end do
            
            v(i) = row_sum
        end do
        !$omp end parallel do
        
        ! Wait for communication
        call MPI_Wait(request, status, ierr)
        
        ! Reorder recv_buf to sorted order
        allocate(recv_buf_sorted(max(total_recv, 1_int64)))
        !$omp parallel do
        do i = 1, total_recv
            recv_buf_sorted(i) = recv_buf(sort_perm(i))
        end do
        !$omp end parallel do
        
        ! Add REMOTE contributions
        !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end, sorted_pos)
        do i = 1, n_local
            start_j = row_starts(i)
            end_j = row_starts(i + 1) - 1
            
            if (start_j > end_j) then
                v(i) = scalar * v(i)
                cycle
            end if
            
            if (col_indexes(start_j) >= lb .and. col_indexes(end_j) <= ub) then
                v(i) = scalar * v(i)
                cycle
            end if
            
            row_sum = v(i)
            
            local_start = lower_bound(col_indexes, start_j, end_j, lb)
            local_end = upper_bound(col_indexes, start_j, end_j, ub) - 1
            
            do j = start_j, local_start - 1
                col = col_indexes(j)
                sorted_pos = hash_lookup(col, hash_keys, hash_vals, hash_size)
                if (sorted_pos > 0) then
                    row_sum = row_sum + values(j) * recv_buf_sorted(sorted_pos)
                end if
            end do
            
            do j = local_end + 1, end_j
                col = col_indexes(j)
                sorted_pos = hash_lookup(col, hash_keys, hash_vals, hash_size)
                if (sorted_pos > 0) then
                    row_sum = row_sum + values(j) * recv_buf_sorted(sorted_pos)
                end if
            end do
            
            v(i) = scalar * row_sum
        end do
        !$omp end parallel do
        
        deallocate(recv_buf_sorted)
    end subroutine spmv

    !--------------------------------------------------------------------------
    ! Original Sparse module functions below
    !--------------------------------------------------------------------------

    !> @brief Merge sort a CSR matrix A by an array of column indexes,
    !> used to form the conjugate transpose of A.


    subroutine Merge_Dagger(column_indexes, &
                            row_indexes, &
                            values, &
                            start, &
                            mid, &
                            finish)

        integer(dp), intent(inout), dimension(:) :: column_indexes
        integer(dp), intent(inout), dimension(:) :: row_indexes
        complex(dp), intent(inout), dimension(:) :: values
        integer(dp), intent(in) :: start
        integer(dp), intent(in) :: mid
        integer(dp), intent(in) :: finish

        integer(dp), dimension(:), allocatable :: col_ind_temp
        integer(dp), dimension(:), allocatable :: row_ind_temp
        complex(dp), dimension(:), allocatable :: val_temp
        integer :: i, j, k

        allocate(col_ind_temp(finish - start + 1))
        allocate(row_ind_temp(finish - start + 1))
        allocate(val_temp(finish - start + 1))

        i = start
        j = mid + 1
        k = 1

        do while (i <= mid .and. j <= finish)

            if (column_indexes(i) <= column_indexes(j)) then
                col_ind_temp(k) = column_indexes(i)
                row_ind_temp(k) = row_indexes(i)
                val_temp(k) = values(i)
                k = k + 1
                i = i + 1
            else
                col_ind_temp(k) = column_indexes(j)
                row_ind_temp(k) = row_indexes(j)
                val_temp(k) = values(j)
                k = k + 1
                j = j+ 1
            endif

        enddo

        do while (i <= mid)
            col_ind_temp(k) = column_indexes(i)
            row_ind_temp(k) = row_indexes(i)
            val_temp(k) = values(i)
            k = k + 1
            i = i + 1
        enddo

        do while (j <= finish)
            col_ind_temp(k) = column_indexes(j)
            row_ind_temp(k) = row_indexes(j)
            val_temp(k) = values(j)
            k = k + 1
            j = j + 1
        enddo

        do i = start, finish
            column_indexes(i) = col_ind_temp(i - start + 1)
            row_indexes(i) = row_ind_temp(i - start + 1)
            values(i) = val_temp(i - start + 1)
        enddo

    end subroutine Merge_Dagger
    
    !> @brief Insertion sort a CSR matrix A by an array of column indexes,
    !> used to form the conjugate transpose of A.

    subroutine Insertion_Sort_Dagger(   column_indexes, &
                                        row_indexes, &
                                        values)

        integer(dp), intent(inout), dimension(:) :: column_indexes
        integer(dp), intent(inout), dimension(:) :: row_indexes
        complex(dp), intent(inout), dimension(:) :: values

        integer(dp) :: col_ind_temp
        integer(dp) :: row_ind_temp
        complex(dp) :: val_temp

        integer :: i, j

        do i = 2, size(column_indexes)

            col_ind_temp = column_indexes(i)
            row_ind_temp = row_indexes(i)
            val_temp = values(i)

            j = i - 1

            do while (j >= 1)

                if (column_indexes(j) <= col_ind_temp) exit
                    column_indexes(j + 1) = column_indexes(j)
                    row_indexes(j + 1) = row_indexes(j)
                    values(j + 1) = values(j)
                    j = j - 1
            enddo
            column_indexes(j + 1) = col_ind_temp
            row_indexes(j + 1) = row_ind_temp
            values(j + 1) = val_temp

        enddo

    end subroutine Insertion_Sort_Dagger

    !> @brief Merge sort a CSR matrix A by an array of column indexes,
    !> used to form the conjugate transpose of A.

    recursive subroutine Merge_Sort_Dagger( column_indexes, &
                                            row_indexes, &
                                            values, &
                                            start, &
                                            finish)

        integer(dp), intent(inout), dimension(:) :: column_indexes
        integer(dp), intent(inout), dimension(:) :: row_indexes
        complex(dp), intent(inout), dimension(:) :: values
        integer(dp), intent(in) :: start
        integer(dp), intent(in) :: finish

        integer(dp) :: mid

        if (start < finish) then
            if (finish - start >= 512) then

                mid = (start + finish) / 2

                call Merge_Sort_Dagger( column_indexes, &
                                        row_indexes, &
                                        values, &
                                        start, &
                                        mid)

                call Merge_Sort_Dagger( column_indexes, &
                                        row_indexes, &
                                        values, &
                                        mid + 1, &
                                        finish)

                call Merge_Dagger(  column_indexes, &
                                    row_indexes, &
                                    values, &
                                    start, &
                                    mid, &
                                    finish)

            else
                call insertion_sort_Dagger( column_indexes(start:finish), &
                                            row_indexes(start:finish), &
                                            values(start:finish))
            endif
        endif

    end subroutine Merge_Sort_Dagger

    !> @brief Returns the distributed conjugate transpose of CSR matrix A.

    subroutine CSR_Dagger(A, partition_table, A_T, MPI_communicator)

        type(CSR), intent(in) :: A
        integer, dimension(:), intent(in) :: partition_table
        type(CSR), intent(out) :: A_T
        integer, intent(in) :: MPI_communicator

        integer :: lb, ub
        integer :: element_lb_T, element_ub_T

        integer :: nz

        integer(dp), dimension(:), allocatable :: row_indexes, column_indexes
        integer(dp), dimension(:), allocatable :: column_indexes_in
        complex(dp), dimension(:), allocatable :: values

        integer, dimension(:), allocatable :: send_counts, rec_counts
        integer, dimension(:), allocatable :: send_disps, rec_disps

        integer, dimension(:), allocatable :: elements_per_rank
        integer, dimension(:), allocatable :: elements_per_rank_temp

        integer, dimension(:), allocatable :: mapping_disps
        integer(dp), dimension(:), allocatable :: column_indexes_out, row_indexes_out
        complex(dp), dimension(:), allocatable :: values_out

        integer, dimension(:), allocatable :: target_rank

        integer :: i, j

        !MPI_Environment
        integer :: rank
        integer :: flock
        integer :: ierr

        call MPI_comm_size(MPI_communicator, flock, ierr)
        call MPI_comm_rank(MPI_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank +2) - 1

        nz = size(A%col_indexes)

        A_T%rows = A%rows
        A_T%columns = A%columns

        allocate(column_indexes(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate(row_indexes(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate(values(A%row_starts(lb):A%row_starts(ub + 1) - 1))

        do i = lb, ub
            do j = A%row_starts(i), A%row_starts(i + 1) - 1
                row_indexes(j) = i
            enddo
        enddo

        do i = A%row_starts(lb), A%row_starts(ub + 1) - 1
            column_indexes(i) = A%col_indexes(i)
        enddo

        do i = A%row_starts(lb), A%row_starts(ub + 1) - 1
            values(i) = A%values(i)
        enddo

        allocate(send_counts(flock))

        send_counts = 0

        allocate(target_rank(A%row_starts(lb):A%row_starts(ub + 1) - 1))

        do i = lbound(column_indexes, 1), ubound(column_indexes, 1)

            do j = flock, 1, -1
                if (column_indexes(i) >= partition_table(j)) then
                    send_counts(j) = send_counts(j) + 1
                    target_rank(i) = j
                    exit
                endif
            enddo
        enddo

        allocate(send_disps(flock))

        send_disps(1) = 0

        do i = 2, flock
            send_disps(i) = send_disps(i - 1) + send_counts(i - 1)
        enddo

        allocate(mapping_disps(flock))

        mapping_disps = 0

        allocate(column_indexes_out(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate(row_indexes_out(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate(values_out(A%row_starts(lb):A%row_starts(ub + 1) - 1))

        do i = lb, ub
            do j = A%row_starts(i), A%row_starts(i + 1) - 1

                column_indexes_out(A%row_starts(lb) + send_disps(target_rank(j))  &
                    + mapping_disps(target_rank(j))) = column_indexes(j)

                values_out(A%row_starts(lb) + send_disps(target_rank(j))  &
                    + mapping_disps(target_rank(j))) = conjg(values(j))

                row_indexes_out(A%row_starts(lb) + send_disps(target_rank(j))  &
                    + mapping_disps(target_rank(j))) = row_indexes(j)

                mapping_disps(target_rank(j)) = mapping_disps(target_rank(j)) + 1

            enddo
        enddo

        allocate(rec_counts(flock))

        call MPI_alltoall(  send_counts, &
                            1, &
                            MPI_integer, &
                            rec_counts, &
                            1, &
                            MPI_integer, &
                            MPI_communicator, &
                            ierr)

        allocate(elements_per_rank_temp(flock))

        elements_per_rank_temp = 0
        elements_per_rank_temp(rank + 1) = sum(rec_counts)

        allocate(elements_per_rank(flock + 1))

        elements_per_rank(1) = 1
        elements_per_rank(2:flock + 1) = 0

        call mpi_allreduce( elements_per_rank_temp, &
                            elements_per_rank(2:flock + 1), &
                            flock, &
                            mpi_integer, &
                            mpi_sum, &
                            mpi_communicator, &
                            ierr)

        do i = 2, flock + 1
           elements_per_rank(i) = elements_per_rank(i) + elements_per_rank(i - 1)
        enddo

        element_lb_T = elements_per_rank(rank + 1)
        element_ub_T = elements_per_rank(rank + 2) - 1

        allocate(column_indexes_in(element_lb_T:element_ub_T))
        allocate(A_T%col_indexes(element_lb_T:element_ub_T))
        allocate(A_T%values(element_lb_T:element_ub_T))

        allocate(rec_disps(flock))

        rec_disps(1) = 0

        do i = 2, flock
            rec_disps(i) = rec_disps(i - 1) + rec_counts(i - 1)
        enddo

        call MPI_alltoallv( column_indexes_out, &
                            send_counts, &
                            send_disps, &
                            MPI_LONG, &
                            column_indexes_in, &
                            rec_counts, &
                            rec_disps, &
                            MPI_LONG, &
                            MPI_communicator, &
                            ierr)

        call MPI_alltoallv( row_indexes_out, &
                            send_counts, &
                            send_disps, &
                            MPI_LONG, &
                            A_T%col_indexes, &
                            rec_counts, &
                            rec_disps, &
                            MPI_LONG, &
                            MPI_communicator, &
                            ierr)

        call MPI_alltoallv( values_out, &
                            send_counts, &
                            send_disps, &
                            MPI_double_complex, &
                            A_T%values, &
                            rec_counts, &
                            rec_disps, &
                            MPI_double_complex, &
                            MPI_communicator, &
                            ierr)

        call Merge_Sort_Dagger( column_indexes_in, &
                                A_T%col_indexes, &
                                A_T%values, &
                                1_dp, &
                                size(column_indexes_in, kind = dp))

        allocate(A_T%row_starts(lb:ub+1))

        A_T%row_starts(lb) = elements_per_rank(rank + 1)
        A_T%row_starts(lb + 1:ub + 1) = 0

        do i = element_lb_T, element_ub_T
            A_T%row_starts(column_indexes_in(i) + 1) = &
                A_T%row_starts(column_indexes_in(i) + 1) + 1
        enddo

        do i = lb + 1, ub + 1
            A_T%row_starts(i) = A_T%row_starts(i) + A_T%row_starts(i - 1)
        enddo

        call MPI_barrier(MPI_communicator, ierr)

    end subroutine CSR_Dagger

    !--------------------------------------------------------------------------
    ! Setup graph communicator for efficient neighbor-based SpMV
    ! This replaces Reconcile_Communications for the new SpMV method
    !--------------------------------------------------------------------------
    subroutine Setup_Graph_Communications(A, partition_table, MPI_communicator)
        type(CSR), intent(inout) :: A
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator

        integer :: rank, flock, ierr
        integer(int64) :: lb, ub, n_local, local_nnz
        integer(int64) :: lb_elem, ub_elem
        integer(int64), allocatable :: partition_table_64(:)
        integer :: i

        call MPI_Comm_rank(MPI_communicator, rank, ierr)
        call MPI_Comm_size(MPI_communicator, flock, ierr)

        lb = int(partition_table(rank + 1), int64)
        ub = int(partition_table(rank + 2) - 1, int64)
        n_local = ub - lb + 1

        ! Convert partition table to int64
        allocate(partition_table_64(flock + 1))
        do i = 1, flock + 1
            partition_table_64(i) = int(partition_table(i), int64)
        end do

        ! Determine element bounds from the array itself
        ! A%row_starts can be indexed either locally (1:n_local+1) or globally (lb:ub+1)
        ! A%col_indexes and A%values should be consistent with row_starts
        lb_elem = int(lbound(A%col_indexes, 1), int64)
        ub_elem = int(ubound(A%col_indexes, 1), int64)
        local_nnz = ub_elem - lb_elem + 1
        
        ! Create local row_starts (1-indexed, relative values)
        allocate(A%row_starts_local(n_local + 1))
        ! Check if row_starts is locally indexed (starts at 1) or globally indexed (starts at lb)
        if (lbound(A%row_starts, 1) == 1) then
            ! Locally indexed: row_starts(1:n_local+1), values start at 1
            do i = 1, int(n_local + 1)
                A%row_starts_local(i) = A%row_starts(i) - A%row_starts(1) + 1
            end do
        else
            ! Globally indexed: row_starts(lb:ub+1), values start at row_starts(lb)
            do i = 1, int(n_local + 1)
                A%row_starts_local(i) = A%row_starts(lb + i - 1) - A%row_starts(lb) + 1
            end do
        end if
        
        ! Create local col_indexes copy (global column indices)
        allocate(A%col_indexes_local(local_nnz))
        A%col_indexes_local = A%col_indexes(lb_elem:ub_elem)
        
        ! Create local values copy
        allocate(A%values_local(local_nnz))
        A%values_local = A%values(lb_elem:ub_elem)

        ! Call chunked_spmv_mod setup
        call setup_graph_comm(A%row_starts_local, A%col_indexes_local, partition_table_64, &
                              A%graph_comm, &
                              A%recv_indices_sorted, A%sort_perm, &
                              A%graph_recv_counts, A%graph_recv_disps, &
                              A%send_offsets, A%graph_send_counts, A%graph_send_disps, &
                              A%in_neighbors, A%out_neighbors, &
                              A%total_recv, A%total_send, A%lb_graph, A%ub_graph)

        ! Build hash table for O(1) remote column lookup
        call build_hash_table(A%recv_indices_sorted, A%sort_perm, A%total_recv, &
                              A%hash_keys, A%hash_vals, A%hash_size)

        ! Allocate persistent communication buffers
        allocate(A%send_buf(max(A%total_send, 1_int64)))
        allocate(A%recv_buf(max(A%total_recv, 1_int64)))

        A%graph_comm_ready = .true.

        deallocate(partition_table_64)

    end subroutine Setup_Graph_Communications

    !--------------------------------------------------------------------------
    ! Graph-communicator-based SpMV: y = scalar * A * x
    ! Uses neighbor collectives for O(neighbors) scaling
    !--------------------------------------------------------------------------
    subroutine SpMV_Graph(A, x_local, partition_table, rank, y_local, &
                          scalar, MPI_communicator)
        type(CSR), intent(inout) :: A
        complex(dp), dimension(:), intent(in) :: x_local
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(in) :: rank
        complex(dp), dimension(:), intent(out) :: y_local
        complex(dp), intent(in), optional :: scalar
        integer, intent(in) :: MPI_communicator

        complex(dp) :: sc

        if (.not. A%graph_comm_ready) then
            call Setup_Graph_Communications(A, partition_table, MPI_communicator)
        end if

        if (present(scalar)) then
            sc = scalar
        else
            sc = (1.0_dp, 0.0_dp)
        end if

        call spmv(A%row_starts_local, A%col_indexes_local, &
                  x_local, y_local, sc, &
                  A%graph_comm, &
                  A%recv_indices_sorted, A%sort_perm, &
                  A%graph_recv_counts, A%graph_recv_disps, &
                  A%send_offsets, A%graph_send_counts, A%graph_send_disps, &
                  A%total_recv, A%total_send, A%lb_graph, A%ub_graph, &
                  A%send_buf, A%recv_buf, &
                  A%hash_keys, A%hash_vals, A%hash_size, &
                  A%values_local)

    end subroutine SpMV_Graph

    !--------------------------------------------------------------------------
    ! Graph-communicator-based SpMM: C = A^n * B
    ! Uses neighbor collectives for O(neighbors) scaling
    ! For n > 1, iterates with temporary storage
    !--------------------------------------------------------------------------
    subroutine SpMM_Graph(A, n, B_local, partition_table, rank, C_local, MPI_communicator)
        type(CSR), intent(inout) :: A
        integer, intent(in) :: n
        complex(dp), dimension(:,:), intent(in) :: B_local
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(in) :: rank
        complex(dp), dimension(:,:), intent(out) :: C_local
        integer, intent(in) :: MPI_communicator

        integer :: k, col, n_cols, n_local
        complex(dp), allocatable :: temp_in(:,:), temp_out(:,:)
        complex(dp), allocatable :: col_in(:), col_out(:)

        if (.not. A%graph_comm_ready) then
            call Setup_Graph_Communications(A, partition_table, MPI_communicator)
        end if

        n_local = int(A%ub_graph - A%lb_graph + 1)
        n_cols = size(B_local, 2)

        allocate(col_in(n_local), col_out(n_local))

        if (n == 1) then
            ! Simple case: single multiplication
            do col = 1, n_cols
                col_in = B_local(:, col)
                call spmv(A%row_starts_local, A%col_indexes_local, &
                          col_in, col_out, (1.0_dp, 0.0_dp), &
                          A%graph_comm, &
                          A%recv_indices_sorted, A%sort_perm, &
                          A%graph_recv_counts, A%graph_recv_disps, &
                          A%send_offsets, A%graph_send_counts, A%graph_send_disps, &
                          A%total_recv, A%total_send, A%lb_graph, A%ub_graph, &
                          A%send_buf, A%recv_buf, &
                          A%hash_keys, A%hash_vals, A%hash_size, &
                          A%values_local)
                C_local(:, col) = col_out
            end do
        else
            ! Multiple multiplications: A^n * B
            allocate(temp_in(n_local, n_cols), temp_out(n_local, n_cols))
            temp_in = B_local

            do k = 1, n
                do col = 1, n_cols
                    col_in = temp_in(:, col)
                    call spmv(A%row_starts_local, A%col_indexes_local, &
                              col_in, col_out, (1.0_dp, 0.0_dp), &
                              A%graph_comm, &
                              A%recv_indices_sorted, A%sort_perm, &
                              A%graph_recv_counts, A%graph_recv_disps, &
                              A%send_offsets, A%graph_send_counts, A%graph_send_disps, &
                              A%total_recv, A%total_send, A%lb_graph, A%ub_graph, &
                              A%send_buf, A%recv_buf, &
                              A%hash_keys, A%hash_vals, A%hash_size, &
                              A%values_local)
                    temp_out(:, col) = col_out
                end do
                if (k < n) then
                    temp_in = temp_out
                end if
            end do

            C_local = temp_out
            deallocate(temp_in, temp_out)
        end if

        deallocate(col_in, col_out)

    end subroutine SpMM_Graph

    !--------------------------------------------------------------------------
    ! Cleanup graph communicator resources
    !--------------------------------------------------------------------------
    subroutine Cleanup_Graph_Communications(A)
        type(CSR), intent(inout) :: A
        integer :: ierr

        if (A%graph_comm /= MPI_COMM_NULL) then
            call MPI_Comm_free(A%graph_comm, ierr)
            A%graph_comm = MPI_COMM_NULL
        end if

        if (allocated(A%recv_indices_sorted)) deallocate(A%recv_indices_sorted)
        if (allocated(A%sort_perm)) deallocate(A%sort_perm)
        if (allocated(A%graph_recv_counts)) deallocate(A%graph_recv_counts)
        if (allocated(A%graph_recv_disps)) deallocate(A%graph_recv_disps)
        if (allocated(A%send_offsets)) deallocate(A%send_offsets)
        if (allocated(A%graph_send_counts)) deallocate(A%graph_send_counts)
        if (allocated(A%graph_send_disps)) deallocate(A%graph_send_disps)
        if (allocated(A%in_neighbors)) deallocate(A%in_neighbors)
        if (allocated(A%out_neighbors)) deallocate(A%out_neighbors)
        if (allocated(A%hash_keys)) deallocate(A%hash_keys)
        if (allocated(A%hash_vals)) deallocate(A%hash_vals)
        if (allocated(A%send_buf)) deallocate(A%send_buf)
        if (allocated(A%recv_buf)) deallocate(A%recv_buf)
        if (allocated(A%row_starts_local)) deallocate(A%row_starts_local)
        if (allocated(A%col_indexes_local)) deallocate(A%col_indexes_local)
        if (allocated(A%values_local)) deallocate(A%values_local)

        A%graph_comm_ready = .false.

    end subroutine Cleanup_Graph_Communications

end module Sparse
