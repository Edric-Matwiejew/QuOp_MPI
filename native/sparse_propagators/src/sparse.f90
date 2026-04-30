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
module sparse

    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, int32, int64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
#ifdef USE_HIP
    use, intrinsic :: iso_c_binding, only: c_associated, c_size_t, c_loc
    use hipfort
    use hipfort_check
    use hipfort_types, only: dim3
    use hip_sparse_expm_kernels, only: &
        launch_complex_scale_kernel, launch_pack_send_buf_kernel, launch_reorder_recv_buf_kernel, &
        launch_spmv_local_unit_kernel, launch_spmv_local_weighted_kernel, launch_spmv_remote_unit_kernel, &
        launch_spmv_remote_weighted_kernel
#endif
    use :: MPI

    implicit none

    private
    public :: csr_dagger, spmv_graph, spmm_graph
    public :: setup_graph_communications, cleanup_graph_communications
    public :: c_ptr, c_null_ptr
#ifdef USE_HIP
    public :: csr_to_device, csr_free_device, csr_update_values_device, spmv_gpu
#endif

    ! Hash table constants (Knuth's golden ratio multiplier)
    integer(int64), parameter :: HASH_MULT = 2654435769_int64
    integer(int64), parameter :: MASK32 = int(Z'FFFFFFFF', int64)

    !> @brief Compressed sparse rows (CSR) complex matrix derived type.
    !
    !> @warning *Sparse_Operations.mod* requiers that the entries for each row are
    !> stored in acessending order. This condition may not be enforced
    !> external sparse libraries.

    type, public  :: CSR

        integer(int32) :: rows
        integer(int32) :: columns
        character(len=2) :: structure
        integer(int64), dimension(:), pointer :: row_starts => null()
        integer(int64), dimension(:), pointer :: col_indexes => null()
        complex(real64), dimension(:), pointer :: values => null()

        ! Graph communicator data for O(neighbors) SpMV
        integer(int32) :: graph_comm = MPI_COMM_NULL
        integer(int64), dimension(:), allocatable :: recv_indices_sorted
        integer(int64), dimension(:), pointer :: sort_perm => null()
        integer, dimension(:), allocatable :: graph_recv_counts
        integer, dimension(:), allocatable :: graph_recv_disps
        integer(int64), dimension(:), pointer :: send_offsets => null()
        integer, dimension(:), allocatable :: graph_send_counts
        integer, dimension(:), allocatable :: graph_send_disps
        integer, dimension(:), allocatable :: in_neighbors
        integer, dimension(:), allocatable :: out_neighbors
        integer(int64) :: total_recv = 0
        integer(int64) :: total_send = 0
        integer(int64) :: lb_graph = 0
        integer(int64) :: ub_graph = 0
        ! Local copies of row_starts and col_indexes (0-based for C/GPU interop)
        integer(int64), dimension(:), pointer :: row_starts_local => null()
        integer(int64), dimension(:), pointer :: col_indexes_local => null()
        ! Hash table for O(1) remote column lookup
        integer(int64), dimension(:), pointer :: hash_keys => null()
        integer(int64), dimension(:), pointer :: hash_vals => null()
        integer(int64) :: hash_size = 0
        ! Persistent communication buffers
        complex(real64), dimension(:), pointer :: send_buf => null()
        complex(real64), dimension(:), pointer :: recv_buf => null()
        ! Local values for SpMV (slice of global values)
        complex(real64), dimension(:), pointer :: values_local => null()
        ! True when *_local pointers own freshly allocated storage (legacy
        ! globally-indexed path); false when they alias the borrowed
        ! row_starts/col_indexes/values buffers (locally-indexed path).
        logical :: owns_local_arrays = .false.
        ! Flag to indicate if graph comm is set up
        logical :: graph_comm_ready = .false.
        ! Flag to indicate if values are explicit (false = all ones)
        logical :: has_values = .true.

#ifdef USE_HIP
        !----------------------------------------------------------------------
        ! HIP/GPU device memory (only used when GPU backend is active)
        ! These are c_ptr to device memory, initialized to c_null_ptr
        !----------------------------------------------------------------------
        ! CSR structure on device
        type(c_ptr) :: row_starts_dev = c_null_ptr
        type(c_ptr) :: col_indexes_dev = c_null_ptr
        type(c_ptr) :: values_dev = c_null_ptr
        ! Communication buffers on device
        type(c_ptr) :: send_buf_dev = c_null_ptr
        type(c_ptr) :: recv_buf_dev = c_null_ptr
        type(c_ptr) :: recv_buf_sorted_dev = c_null_ptr
        ! Hash table on device (for remote column lookup)
        type(c_ptr) :: hash_keys_dev = c_null_ptr
        type(c_ptr) :: hash_vals_dev = c_null_ptr
        ! Sort permutation on device
        type(c_ptr) :: sort_perm_dev = c_null_ptr
        ! Send offsets on device
        type(c_ptr) :: send_offsets_dev = c_null_ptr
        ! Intermediate result for Chebyshev recurrence: Aw_k from local phase
        type(c_ptr) :: Aw_k_dev = c_null_ptr
        ! HIP stream for async operations
        type(c_ptr) :: stream = c_null_ptr
        ! Flag to indicate if device memory is allocated and ready
        logical :: device_ready = .false.
#endif

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
    ! Compute hash index for a column (returns 0-based position)
    !--------------------------------------------------------------------------
    pure function compute_hash(col, hash_size) result(hash_pos)
        integer(int64), intent(in) :: col, hash_size
        integer(int64) :: hash_pos
        integer(int64) :: folded

        folded = iand(ieor(col, ishft(col, -32)), MASK32)
        hash_pos = mod(folded * HASH_MULT, hash_size)
        if (hash_pos < 0) hash_pos = hash_pos + hash_size
    end function compute_hash

    !--------------------------------------------------------------------------
    ! Look up a column in the hash table
    ! hash_keys/hash_vals are 1-based arrays, hash_keys stores 0-based columns
    ! hash_vals stores 1-based positions, returns 0 if not found
    !--------------------------------------------------------------------------
    pure function hash_lookup(col, hash_keys, hash_vals, hash_size) result(pos)
        integer(int64), intent(in) :: col ! 0-based column to look up
        integer(int64), intent(in) :: hash_keys(:) ! 1-based array, stores 0-based columns
        integer(int64), intent(in) :: hash_vals(:) ! 1-based array, stores 1-based positions
        integer(int64), intent(in) :: hash_size
        integer(int64) :: pos
        integer(int64) :: hash_pos, probe, idx

        pos = 0_int64 ! Not found sentinel (0 = invalid 1-based position)
        hash_pos = compute_hash(col, hash_size)
        do probe = 0, hash_size - 1
            idx = hash_pos + 1 ! Convert to 1-based index
            if (hash_keys(idx) == col) then
                pos = hash_vals(idx) ! 1-based position
                return
            else if (hash_keys(idx) < 0) then
                return
            end if
            hash_pos = mod(hash_pos + 1, hash_size)
        end do
    end function hash_lookup

    !--------------------------------------------------------------------------
    ! Find owner rank for a column index
    !--------------------------------------------------------------------------
    pure function find_owner(col, partition_table) result(owner)
        integer(int64), intent(in) :: col
        integer(int64), intent(in) :: partition_table(:)
        integer(int32) :: owner
        integer(int32) :: lo, hi, mid

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
        integer(int32) :: i, j, n
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
        integer(int32) :: n, mid
        integer(int64), allocatable :: left(:), right(:)

        n = size(arr)
        if (n <= 1) return

        mid = n / 2
        allocate (left(mid), right(n - mid))
        left = arr(1:mid)
        right = arr(mid + 1:n)

        call merge_sort_int64(left)
        call merge_sort_int64(right)
        call merge_arrays_int64(left, right, arr)

        deallocate (left, right)
    end subroutine merge_sort_int64

    !--------------------------------------------------------------------------
    ! Merge two sorted arrays into one
    !--------------------------------------------------------------------------
    pure subroutine merge_arrays_int64(left, right, arr)
        integer(int64), intent(in) :: left(:), right(:)
        integer(int64), intent(out) :: arr(:)
        integer(int32) :: i, j, k

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
    ! Arrays are 1-based (normal Fortran), but column values stored are 0-based
    !--------------------------------------------------------------------------
    subroutine setup_graph_comm(row_starts, col_indexes, partition_table, &
                                MPI_communicator, graph_comm, &
                                recv_indices_sorted, sort_perm, &
                                recv_counts, recv_disps, &
                                send_offsets, send_counts, send_disps, &
                                in_neighbors, out_neighbors, &
                                total_recv, total_send, lb, ub)
        integer(int64), intent(in) :: row_starts(:) ! 1-based array, 1-based offsets
        integer(int64), intent(in) :: col_indexes(:) ! 1-based array, 0-based column values
        integer(int64), intent(in) :: partition_table(:) ! 1-based array, 0-based column boundaries
        integer, intent(in) :: MPI_communicator
        integer, intent(out) :: graph_comm
        integer(int64), allocatable, intent(out) :: recv_indices_sorted(:) ! 1-based, 0-based col values
        integer(int64), pointer, intent(out) :: sort_perm(:) ! 1-based, 1-based positions
        integer, allocatable, intent(out) :: recv_counts(:), recv_disps(:)
        integer(int64), pointer, intent(out) :: send_offsets(:) ! 1-based, 1-based offsets
        integer, allocatable, intent(out) :: send_counts(:), send_disps(:)
        integer, allocatable, intent(out) :: in_neighbors(:), out_neighbors(:)
        integer(int64), intent(out) :: total_recv, total_send
        integer(int64), intent(out) :: lb, ub ! 0-based column bounds

        integer(int32) :: rank, nprocs, ierr, i, r, owner, n_out, n_in, idx, pos
        integer(int64) :: col, n_local, j, nnz_local
        integer, allocatable :: in_weights(:), out_weights(:)
        integer, allocatable :: in_neighbor_list(:), out_neighbor_list(:)
        integer(int64), allocatable :: all_recv_indices(:), requested(:)
        integer(int64), allocatable :: temp_sort_perm(:)
        logical, allocatable :: is_out_neighbor(:), is_in_neighbor(:)
        integer(int64), allocatable :: seen_cols(:)
        integer(int64) :: n_seen

        call MPI_Comm_rank(MPI_communicator, rank, ierr)
        call MPI_Comm_size(MPI_communicator, nprocs, ierr)

        ! partition_table is 1-based and contains 0-based column boundaries
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1
        nnz_local = size(col_indexes)

        ! Step 1: Identify out_neighbors (ranks we need data from)
        ! col_indexes contains 0-based column values
        allocate (is_out_neighbor(0:nprocs - 1))
        is_out_neighbor = .false.

        do j = 1, nnz_local
            col = col_indexes(j) ! 0-based column value
            if (col < lb .or. col > ub) then
                owner = find_owner(col, partition_table)
                is_out_neighbor(owner) = .true.
            end if
        end do
        is_out_neighbor(rank) = .false.

        n_out = count(is_out_neighbor)
        allocate (out_neighbor_list(max(n_out, 1)))
        allocate (out_neighbors(max(n_out, 1)))
        idx = 1
        do r = 0, nprocs - 1
            if (is_out_neighbor(r)) then
                out_neighbor_list(idx) = r
                out_neighbors(idx) = r
                idx = idx + 1
            end if
        end do

        ! Step 2: Exchange to find in_neighbors
        allocate (is_in_neighbor(0:nprocs - 1))
        is_in_neighbor = .false.

        call MPI_Alltoall(is_out_neighbor, 1, MPI_LOGICAL, &
                          is_in_neighbor, 1, MPI_LOGICAL, MPI_communicator, ierr)

        n_in = count(is_in_neighbor)
        allocate (in_neighbor_list(max(n_in, 1)))
        allocate (in_neighbors(max(n_in, 1)))
        idx = 1
        do r = 0, nprocs - 1
            if (is_in_neighbor(r)) then
                in_neighbor_list(idx) = r
                in_neighbors(idx) = r
                idx = idx + 1
            end if
        end do

        ! Step 3: Create graph communicator
        allocate (in_weights(max(n_in, 1)), out_weights(max(n_out, 1)))
        in_weights = 1
        out_weights = 1

        call MPI_Dist_graph_create_adjacent(MPI_communicator, &
                                            n_out, out_neighbor_list, out_weights, &
                                            n_in, in_neighbor_list, in_weights, &
                                            MPI_INFO_NULL, .false., graph_comm, ierr)

        deallocate (in_weights, out_weights)

        ! Step 4: Collect all remote columns, sort, deduplicate
        ! col_indexes is 1-based array with 0-based column values
        n_seen = 0
        do j = 1, nnz_local
            col = col_indexes(j) ! 0-based column value
            if (col < lb .or. col > ub) then
                n_seen = n_seen + 1
            end if
        end do

        allocate (seen_cols(max(n_seen, 1_int64)))
        idx = 1
        do j = 1, nnz_local
            col = col_indexes(j) ! 0-based column value
            if (col < lb .or. col > ub) then
                seen_cols(idx) = col
                idx = idx + 1
            end if
        end do

        if (n_seen > 1) call merge_sort_int64(seen_cols(1:n_seen))

        if (n_seen > 0) then
            total_recv = 1
            do j = 2, n_seen
                if (seen_cols(j) /= seen_cols(j - 1)) then
                    total_recv = total_recv + 1
                end if
            end do
        else
            total_recv = 0
        end if

        allocate (all_recv_indices(max(total_recv, 1_int64)))
        if (n_seen > 0) then
            all_recv_indices(1) = seen_cols(1)
            idx = 1
            do j = 2, n_seen
                if (seen_cols(j) /= seen_cols(j - 1)) then
                    idx = idx + 1
                    all_recv_indices(idx) = seen_cols(j)
                end if
            end do
        end if

        deallocate (seen_cols)

        ! Step 5: Count per neighbor and build displacements
        ! all_recv_indices contains 0-based column indices
        allocate (recv_counts(max(n_out, 1)))
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

        allocate (recv_disps(max(n_out, 1)))
        recv_disps = 0 ! Initialize to zero
        if (n_out > 0) then
            recv_disps(1) = 0
            do i = 2, n_out
                recv_disps(i) = recv_disps(i - 1) + recv_counts(i - 1)
            end do
        end if

        ! Reorder all_recv_indices to be grouped by neighbor
        block
            integer(int64), allocatable :: temp_indices(:)
            integer, allocatable :: neighbor_pos(:)

            allocate (temp_indices(max(total_recv, 1_int64)))
            allocate (neighbor_pos(max(n_out, 1)))
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
            deallocate (temp_indices, neighbor_pos)
        end block

        ! Step 6: Sort recv indices for binary search and create permutation
        ! recv_indices_sorted: 1-based array with 0-based column values
        ! sort_perm: 1-based array with 1-based positions
        allocate (recv_indices_sorted(max(total_recv, 1_int64)))
        allocate (sort_perm(max(total_recv, 1_int64)))
        allocate (temp_sort_perm(max(total_recv, 1_int64)))

        do i = 1, max(total_recv, 1_int64)
            temp_sort_perm(i) = i
        end do
        if (total_recv > 0) then
            recv_indices_sorted(1:total_recv) = all_recv_indices(1:total_recv)
            call sort_with_perm(recv_indices_sorted(1:total_recv), temp_sort_perm(1:total_recv))
            sort_perm(1:total_recv) = temp_sort_perm(1:total_recv) ! 1-based positions
        end if

        deallocate (temp_sort_perm)

        ! Step 7: Exchange counts to set up send side
        block
            integer, allocatable :: all_recv_counts(:), all_send_counts(:)

            allocate (all_recv_counts(nprocs), all_send_counts(nprocs))
            all_recv_counts = 0

            do i = 1, n_out
                all_recv_counts(out_neighbor_list(i) + 1) = recv_counts(i)
            end do

            call MPI_Alltoall(all_recv_counts, 1, MPI_INTEGER, &
                              all_send_counts, 1, MPI_INTEGER, MPI_communicator, ierr)

            allocate (send_counts(max(n_in, 1)))
            send_counts = 0 ! Initialize to zero
            do i = 1, n_in
                send_counts(i) = all_send_counts(in_neighbor_list(i) + 1)
            end do

            deallocate (all_recv_counts, all_send_counts)
        end block

        total_send = sum(send_counts)

        allocate (send_disps(max(n_in, 1)))
        send_disps = 0 ! Initialize to zero
        if (n_in > 0) then
            send_disps(1) = 0
            do i = 2, n_in
                send_disps(i) = send_disps(i - 1) + send_counts(i - 1)
            end do
        end if

        ! Step 8: Exchange indices to know what neighbors need from us
        block
            integer, allocatable :: all_recv_counts(:), all_send_counts(:)
            integer, allocatable :: all_recv_disps(:), all_send_disps(:)
            integer(int64), allocatable :: all_send_indices(:), all_recv_requested(:)
            integer(int32) :: total_all_send, total_all_recv

            allocate (all_recv_counts(nprocs), all_send_counts(nprocs))
            all_recv_counts = 0
            all_send_counts = 0

            do i = 1, n_out
                all_recv_counts(out_neighbor_list(i) + 1) = recv_counts(i)
            end do
            do i = 1, n_in
                all_send_counts(in_neighbor_list(i) + 1) = send_counts(i)
            end do

            allocate (all_recv_disps(nprocs), all_send_disps(nprocs))
            all_recv_disps(1) = 0
            all_send_disps(1) = 0
            do i = 2, nprocs
                all_recv_disps(i) = all_recv_disps(i - 1) + all_recv_counts(i - 1)
                all_send_disps(i) = all_send_disps(i - 1) + all_send_counts(i - 1)
            end do

            total_all_send = sum(all_recv_counts)
            total_all_recv = sum(all_send_counts)

            allocate (all_send_indices(max(total_all_send, 1)))
            do i = 1, n_out
                do j = 1, recv_counts(i)
                    all_send_indices(all_recv_disps(out_neighbor_list(i) + 1) + j) = &
                        all_recv_indices(recv_disps(i) + j)
                end do
            end do

            allocate (all_recv_requested(max(total_all_recv, 1)))

            call MPI_Alltoallv(all_send_indices, all_recv_counts, all_recv_disps, MPI_INTEGER8, &
                               all_recv_requested, all_send_counts, all_send_disps, MPI_INTEGER8, &
                               MPI_communicator, ierr)

            allocate (requested(max(total_send, 1_int64)))
            do i = 1, n_in
                do j = 1, send_counts(i)
                    requested(send_disps(i) + j) = &
                        all_recv_requested(all_send_disps(in_neighbor_list(i) + 1) + j)
                end do
            end do

            deallocate (all_recv_counts, all_send_counts)
            deallocate (all_recv_disps, all_send_disps)
            deallocate (all_send_indices, all_recv_requested)
        end block

        ! send_offsets: 1-based array with 1-based offsets into local vector
        ! requested contains 0-based global column indices that neighbors need
        ! lb is 0-based start of local columns
        allocate (send_offsets(max(total_send, 1_int64)))
        do i = 1, total_send
            send_offsets(i) = requested(i) - lb + 1 ! 1-based offset into local vector
        end do

        deallocate (all_recv_indices, requested)
        deallocate (is_out_neighbor, is_in_neighbor)
        deallocate (in_neighbor_list, out_neighbor_list)
    end subroutine setup_graph_comm

    !--------------------------------------------------------------------------
    ! Build hash table for O(1) average remote column lookup
    ! hash_keys stores 0-based column values (matching col_indexes_local data)
    ! hash_vals stores 1-based positions into recv_buf_sorted
    !--------------------------------------------------------------------------
    subroutine build_hash_table(recv_indices_sorted, total_recv, &
                                hash_keys, hash_vals, hash_size)
        integer(int64), intent(in) :: recv_indices_sorted(:) ! 1-based array, 0-based column values
        integer(int64), intent(in) :: total_recv
        integer(int64), pointer, intent(out) :: hash_keys(:)
        integer(int64), pointer, intent(out) :: hash_vals(:)
        integer(int64), intent(out) :: hash_size

        integer(int64) :: i, hash_pos, probe, idx

        hash_size = 2_int64 * total_recv + 1_int64
        if (mod(hash_size, 2_int64) == 0) hash_size = hash_size + 1_int64

        ! Allocate 1-based arrays (normal Fortran)
        allocate (hash_keys(hash_size), hash_vals(hash_size))
        hash_keys = -1_int64 ! -1 means empty slot
        hash_vals = 0_int64 ! 0 means not found (invalid 1-based position)

        do i = 1, total_recv
            ! recv_indices_sorted(i) contains 0-based column value
            hash_pos = compute_hash(recv_indices_sorted(i), hash_size)
            do probe = 0, hash_size - 1
                idx = hash_pos + 1 ! Convert to 1-based index
                if (hash_keys(idx) < 0) then
                    hash_keys(idx) = recv_indices_sorted(i) ! Store 0-based column
                    hash_vals(idx) = i ! Store 1-based position
                    exit
                end if
                hash_pos = mod(hash_pos + 1, hash_size)
            end do
        end do
    end subroutine build_hash_table

    !--------------------------------------------------------------------------
    ! CPU SpMV using graph communicator and prebuilt hash table (OpenMP)
    ! Uses 0-based data values in 1-based Fortran arrays
    !--------------------------------------------------------------------------
    subroutine spmv_cpu(A, x_local, y_local, scalar)
        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in) :: x_local
        complex(real64), dimension(:), intent(out) :: y_local
        complex(real64), intent(in) :: scalar

        integer(int32) :: ierr, request
        integer(int64) :: i, n_local, col, start_j, end_j, j
        integer(int64) :: local_start, local_end, sorted_pos
        complex(real64) :: row_sum
        integer(int32) :: status(MPI_STATUS_SIZE)
        complex(real64), allocatable :: recv_buf_sorted(:)

        n_local = A%ub_graph - A%lb_graph + 1

        ! Pack send buffer - send_offsets contains 1-based offsets
        !$omp parallel do
        do i = 1, A%total_send
            A%send_buf(i) = x_local(A%send_offsets(i))
        end do
        !$omp end parallel do

        ! Start non-blocking exchange
        call MPI_Ineighbor_alltoallv(A%send_buf, A%graph_send_counts, A%graph_send_disps, &
                                     MPI_DOUBLE_COMPLEX, &
                                     A%recv_buf, A%graph_recv_counts, A%graph_recv_disps, &
                                     MPI_DOUBLE_COMPLEX, &
                                     A%graph_comm, request, ierr)

        ! Compute LOCAL contributions while communication proceeds
        ! row_starts_local contains 0-based offsets, add 1 for 1-based array indexing
        ! col_indexes_local contains 0-based column values
        if (A%has_values) then
            !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end)
            do i = 1, n_local
                start_j = A%row_starts_local(i) + 1 ! Convert to 1-based
                end_j = A%row_starts_local(i + 1) ! Already the last index (1-based)
                row_sum = (0.0_real64, 0.0_real64)

                local_start = lower_bound(A%col_indexes_local, start_j, end_j, A%lb_graph)
                local_end = upper_bound(A%col_indexes_local, start_j, end_j, A%ub_graph) - 1

                do j = local_start, local_end
                    col = A%col_indexes_local(j) ! 0-based column value
                    row_sum = row_sum + A%values_local(j) * x_local(col - A%lb_graph + 1)
                end do

                y_local(i) = row_sum
            end do
            !$omp end parallel do
        else
            !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end)
            do i = 1, n_local
                start_j = A%row_starts_local(i) + 1 ! Convert to 1-based
                end_j = A%row_starts_local(i + 1) ! Already the last index (1-based)
                row_sum = (0.0_real64, 0.0_real64)

                local_start = lower_bound(A%col_indexes_local, start_j, end_j, A%lb_graph)
                local_end = upper_bound(A%col_indexes_local, start_j, end_j, A%ub_graph) - 1

                do j = local_start, local_end
                    col = A%col_indexes_local(j) ! 0-based column value
                    row_sum = row_sum + x_local(col - A%lb_graph + 1)
                end do

                y_local(i) = row_sum
            end do
            !$omp end parallel do
        end if

        ! Wait for communication
        call MPI_Wait(request, status, ierr)

        ! Reorder recv_buf to sorted order
        allocate (recv_buf_sorted(max(A%total_recv, 1_int64)))
        !$omp parallel do
        do i = 1, A%total_recv
            recv_buf_sorted(i) = A%recv_buf(A%sort_perm(i))
        end do
        !$omp end parallel do

        ! Add REMOTE contributions
        ! row_starts_local contains 0-based offsets, hash_lookup returns 1-based positions
        if (A%has_values) then
            !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end, sorted_pos)
            do i = 1, n_local
                start_j = A%row_starts_local(i) + 1 ! Convert to 1-based
                end_j = A%row_starts_local(i + 1) ! Already the last index (1-based)

                if (start_j > end_j) then
                    y_local(i) = scalar * y_local(i)
                    cycle
                end if

                if (A%col_indexes_local(start_j) >= A%lb_graph .and. &
                    A%col_indexes_local(end_j) <= A%ub_graph) then
                    y_local(i) = scalar * y_local(i)
                    cycle
                end if

                row_sum = y_local(i)

                local_start = lower_bound(A%col_indexes_local, start_j, end_j, A%lb_graph)
                local_end = upper_bound(A%col_indexes_local, start_j, end_j, A%ub_graph) - 1

                do j = start_j, local_start - 1
                    col = A%col_indexes_local(j)
                    sorted_pos = hash_lookup(col, A%hash_keys, A%hash_vals, A%hash_size)
                    if (sorted_pos > 0) then
                        row_sum = row_sum + A%values_local(j) * recv_buf_sorted(sorted_pos)
                    end if
                end do

                do j = local_end + 1, end_j
                    col = A%col_indexes_local(j)
                    sorted_pos = hash_lookup(col, A%hash_keys, A%hash_vals, A%hash_size)
                    if (sorted_pos > 0) then
                        row_sum = row_sum + A%values_local(j) * recv_buf_sorted(sorted_pos)
                    end if
                end do

                y_local(i) = scalar * row_sum
            end do
            !$omp end parallel do
        else
            !$omp parallel do private(start_j, end_j, row_sum, j, col, local_start, local_end, sorted_pos)
            do i = 1, n_local
                start_j = A%row_starts_local(i) + 1 ! Convert to 1-based
                end_j = A%row_starts_local(i + 1) ! Already the last index (1-based)

                if (start_j > end_j) then
                    y_local(i) = scalar * y_local(i)
                    cycle
                end if

                if (A%col_indexes_local(start_j) >= A%lb_graph .and. &
                    A%col_indexes_local(end_j) <= A%ub_graph) then
                    y_local(i) = scalar * y_local(i)
                    cycle
                end if

                row_sum = y_local(i)

                local_start = lower_bound(A%col_indexes_local, start_j, end_j, A%lb_graph)
                local_end = upper_bound(A%col_indexes_local, start_j, end_j, A%ub_graph) - 1

                do j = start_j, local_start - 1
                    col = A%col_indexes_local(j)
                    sorted_pos = hash_lookup(col, A%hash_keys, A%hash_vals, A%hash_size)
                    if (sorted_pos > 0) then
                        row_sum = row_sum + recv_buf_sorted(sorted_pos)
                    end if
                end do

                do j = local_end + 1, end_j
                    col = A%col_indexes_local(j)
                    sorted_pos = hash_lookup(col, A%hash_keys, A%hash_vals, A%hash_size)
                    if (sorted_pos > 0) then
                        row_sum = row_sum + recv_buf_sorted(sorted_pos)
                    end if
                end do

                y_local(i) = scalar * row_sum
            end do
            !$omp end parallel do
        end if

        deallocate (recv_buf_sorted)
    end subroutine spmv_cpu

    !--------------------------------------------------------------------------
    ! Original Sparse module functions below
    !--------------------------------------------------------------------------

    !> @brief Merge sort a CSR matrix A by an array of column indexes,
    !> used to form the conjugate transpose of A.

    subroutine merge_dagger(column_indexes, &
                            row_indexes, &
                            values, &
                            start, &
                            mid, &
                            finish)

        integer(int64), intent(inout), dimension(:) :: column_indexes
        integer(int64), intent(inout), dimension(:) :: row_indexes
        complex(real64), intent(inout), dimension(:) :: values
        integer(int64), intent(in) :: start
        integer(int64), intent(in) :: mid
        integer(int64), intent(in) :: finish

        integer(int64), dimension(:), allocatable :: col_ind_temp
        integer(int64), dimension(:), allocatable :: row_ind_temp
        complex(real64), dimension(:), allocatable :: val_temp
        integer(int32) :: i, j, k

        allocate (col_ind_temp(finish - start + 1))
        allocate (row_ind_temp(finish - start + 1))
        allocate (val_temp(finish - start + 1))

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
                j = j + 1
            end if

        end do

        do while (i <= mid)
            col_ind_temp(k) = column_indexes(i)
            row_ind_temp(k) = row_indexes(i)
            val_temp(k) = values(i)
            k = k + 1
            i = i + 1
        end do

        do while (j <= finish)
            col_ind_temp(k) = column_indexes(j)
            row_ind_temp(k) = row_indexes(j)
            val_temp(k) = values(j)
            k = k + 1
            j = j + 1
        end do

        do i = start, finish
            column_indexes(i) = col_ind_temp(i - start + 1)
            row_indexes(i) = row_ind_temp(i - start + 1)
            values(i) = val_temp(i - start + 1)
        end do

    end subroutine merge_dagger

    !> @brief Insertion sort a CSR matrix A by an array of column indexes,
    !> used to form the conjugate transpose of A.

    subroutine insertion_sort_dagger(column_indexes, &
                                     row_indexes, &
                                     values)

        integer(int64), intent(inout), dimension(:) :: column_indexes
        integer(int64), intent(inout), dimension(:) :: row_indexes
        complex(real64), intent(inout), dimension(:) :: values

        integer(int64) :: col_ind_temp
        integer(int64) :: row_ind_temp
        complex(real64) :: val_temp

        integer(int32) :: i, j

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
            end do
            column_indexes(j + 1) = col_ind_temp
            row_indexes(j + 1) = row_ind_temp
            values(j + 1) = val_temp

        end do

    end subroutine insertion_sort_dagger

    !> @brief Merge sort a CSR matrix A by an array of column indexes,
    !> used to form the conjugate transpose of A.

    recursive subroutine merge_sort_dagger(column_indexes, &
                                           row_indexes, &
                                           values, &
                                           start, &
                                           finish)

        integer(int64), intent(inout), dimension(:) :: column_indexes
        integer(int64), intent(inout), dimension(:) :: row_indexes
        complex(real64), intent(inout), dimension(:) :: values
        integer(int64), intent(in) :: start
        integer(int64), intent(in) :: finish

        integer(int64) :: mid

        if (start < finish) then
            if (finish - start >= 512) then

                mid = (start + finish) / 2

                call merge_sort_dagger(column_indexes, &
                                       row_indexes, &
                                       values, &
                                       start, &
                                       mid)

                call merge_sort_dagger(column_indexes, &
                                       row_indexes, &
                                       values, &
                                       mid + 1, &
                                       finish)

                call merge_dagger(column_indexes, &
                                  row_indexes, &
                                  values, &
                                  start, &
                                  mid, &
                                  finish)

            else
                call insertion_sort_dagger(column_indexes(start:finish), &
                                           row_indexes(start:finish), &
                                           values(start:finish))
            end if
        end if

    end subroutine merge_sort_dagger

    !> @brief Returns the distributed conjugate transpose of CSR matrix A.

    subroutine csr_dagger(A, partition_table, A_T, MPI_communicator)

        type(CSR), intent(in) :: A
        integer(int64), dimension(:), intent(in) :: partition_table
        type(CSR), intent(out) :: A_T
        integer, intent(in) :: MPI_communicator

        integer(int64) :: lb, ub
        integer(int32) :: element_lb_T, element_ub_T

        integer(int32) :: nz

        integer(int64), dimension(:), allocatable :: row_indexes, column_indexes
        integer(int64), dimension(:), allocatable :: column_indexes_in
        complex(real64), dimension(:), allocatable :: values

        integer, dimension(:), allocatable :: send_counts, rec_counts
        integer, dimension(:), allocatable :: send_disps, rec_disps

        integer, dimension(:), allocatable :: elements_per_rank
        integer, dimension(:), allocatable :: elements_per_rank_temp

        integer, dimension(:), allocatable :: mapping_disps
        integer(int64), dimension(:), allocatable :: column_indexes_out, row_indexes_out
        complex(real64), dimension(:), allocatable :: values_out

        integer, dimension(:), allocatable :: target_rank

        integer(int64) :: i, j

        !MPI_Environment
        integer(int32) :: rank
        integer(int32) :: flock
        integer(int32) :: ierr

        call MPI_comm_size(MPI_communicator, flock, ierr)
        call MPI_comm_rank(MPI_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        nz = size(A%col_indexes)

        A_T%rows = A%rows
        A_T%columns = A%columns

        allocate (column_indexes(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate (row_indexes(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate (values(A%row_starts(lb):A%row_starts(ub + 1) - 1))

        do i = lb, ub
            do j = A%row_starts(i), A%row_starts(i + 1) - 1
                row_indexes(j) = i
            end do
        end do

        do i = A%row_starts(lb), A%row_starts(ub + 1) - 1
            column_indexes(i) = A%col_indexes(i)
        end do

        do i = A%row_starts(lb), A%row_starts(ub + 1) - 1
            values(i) = A%values(i)
        end do

        allocate (send_counts(flock))

        send_counts = 0

        allocate (target_rank(A%row_starts(lb):A%row_starts(ub + 1) - 1))

        do i = lbound(column_indexes, 1), ubound(column_indexes, 1)

            do j = flock, 1, -1
                if (column_indexes(i) >= partition_table(j)) then
                    send_counts(j) = send_counts(j) + 1
                    target_rank(i) = j
                    exit
                end if
            end do
        end do

        allocate (send_disps(flock))

        send_disps(1) = 0

        do i = 2, flock
            send_disps(i) = send_disps(i - 1) + send_counts(i - 1)
        end do

        allocate (mapping_disps(flock))

        mapping_disps = 0

        allocate (column_indexes_out(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate (row_indexes_out(A%row_starts(lb):A%row_starts(ub + 1) - 1))
        allocate (values_out(A%row_starts(lb):A%row_starts(ub + 1) - 1))

        do i = lb, ub
            do j = A%row_starts(i), A%row_starts(i + 1) - 1

                column_indexes_out(A%row_starts(lb) + send_disps(target_rank(j)) &
                                   + mapping_disps(target_rank(j))) = column_indexes(j)

                values_out(A%row_starts(lb) + send_disps(target_rank(j)) &
                           + mapping_disps(target_rank(j))) = conjg(values(j))

                row_indexes_out(A%row_starts(lb) + send_disps(target_rank(j)) &
                                + mapping_disps(target_rank(j))) = row_indexes(j)

                mapping_disps(target_rank(j)) = mapping_disps(target_rank(j)) + 1

            end do
        end do

        allocate (rec_counts(flock))

        call MPI_alltoall(send_counts, &
                          1, &
                          MPI_integer, &
                          rec_counts, &
                          1, &
                          MPI_integer, &
                          MPI_communicator, &
                          ierr)

        allocate (elements_per_rank_temp(flock))

        elements_per_rank_temp = 0
        elements_per_rank_temp(rank + 1) = sum(rec_counts)

        allocate (elements_per_rank(flock + 1))

        elements_per_rank(1) = 1
        elements_per_rank(2:flock + 1) = 0

        call mpi_allreduce(elements_per_rank_temp, &
                           elements_per_rank(2:flock + 1), &
                           flock, &
                           mpi_integer, &
                           mpi_sum, &
                           mpi_communicator, &
                           ierr)

        do i = 2, flock + 1
            elements_per_rank(i) = elements_per_rank(i) + elements_per_rank(i - 1)
        end do

        element_lb_T = elements_per_rank(rank + 1)
        element_ub_T = elements_per_rank(rank + 2) - 1

        allocate (column_indexes_in(element_lb_T:element_ub_T))
        allocate (A_T%col_indexes(element_lb_T:element_ub_T))
        allocate (A_T%values(element_lb_T:element_ub_T))

        allocate (rec_disps(flock))

        rec_disps(1) = 0

        do i = 2, flock
            rec_disps(i) = rec_disps(i - 1) + rec_counts(i - 1)
        end do

        call MPI_alltoallv(column_indexes_out, &
                           send_counts, &
                           send_disps, &
                           MPI_INTEGER8, &
                           column_indexes_in, &
                           rec_counts, &
                           rec_disps, &
                           MPI_INTEGER8, &
                           MPI_communicator, &
                           ierr)

        call MPI_alltoallv(row_indexes_out, &
                           send_counts, &
                           send_disps, &
                           MPI_INTEGER8, &
                           A_T%col_indexes, &
                           rec_counts, &
                           rec_disps, &
                           MPI_INTEGER8, &
                           MPI_communicator, &
                           ierr)

        call MPI_alltoallv(values_out, &
                           send_counts, &
                           send_disps, &
                           MPI_double_complex, &
                           A_T%values, &
                           rec_counts, &
                           rec_disps, &
                           MPI_double_complex, &
                           MPI_communicator, &
                           ierr)

        call merge_sort_dagger(column_indexes_in, &
                               A_T%col_indexes, &
                               A_T%values, &
                               1_int64, &
                               size(column_indexes_in, kind=int64))

        allocate (A_T%row_starts(lb:ub + 1))

        A_T%row_starts(lb) = elements_per_rank(rank + 1)
        A_T%row_starts(lb + 1:ub + 1) = 0

        do i = element_lb_T, element_ub_T
            A_T%row_starts(column_indexes_in(i) + 1) = &
                A_T%row_starts(column_indexes_in(i) + 1) + 1
        end do

        do i = lb + 1, ub + 1
            A_T%row_starts(i) = A_T%row_starts(i) + A_T%row_starts(i - 1)
        end do

        call MPI_barrier(MPI_communicator, ierr)

    end subroutine csr_dagger

    !--------------------------------------------------------------------------
    ! Setup graph communicator for efficient neighbor-based SpMV
    ! This replaces Reconcile_Communications for the new SpMV method
    ! ASSUMES: A%row_starts and A%col_indexes already contain 0-based values
    !--------------------------------------------------------------------------
    subroutine setup_graph_communications(A, partition_table, MPI_communicator)
        type(CSR), intent(inout) :: A
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator

        integer(int32) :: rank, flock, ierr
        integer(int64) :: lb, ub, n_local, local_nnz
        integer(int64) :: lb_elem, ub_elem
        integer(int64), allocatable :: partition_table_64(:)
        integer(int32) :: i

        call MPI_Comm_rank(MPI_communicator, rank, ierr)
        call MPI_Comm_size(MPI_communicator, flock, ierr)

        ! partition_table is 1-based with 1-based column boundaries, convert to 0-based
        lb = partition_table(rank + 1) - 1 ! 0-based start
        ub = partition_table(rank + 2) - 2 ! 0-based end (inclusive)
        n_local = ub - lb + 1

        ! Convert partition table to int64 with 0-based column boundaries
        allocate (partition_table_64(flock + 1))
        do i = 1, flock + 1
            partition_table_64(i) = partition_table(i) - 1 ! Convert to 0-based
        end do

        ! Determine element bounds from the array itself
        lb_elem = int(lbound(A%col_indexes, 1), int64)
        ub_elem = int(ubound(A%col_indexes, 1), int64)
        local_nnz = ub_elem - lb_elem + 1

        ! Bind *_local to the CSR storage. The data is already 0-based, so
        ! when row_starts/col_indexes/values are locally indexed (lbound == 1)
        ! we alias the borrowed buffers directly to avoid duplicating the CSR
        ! in memory. The globally-indexed path (used only by the legacy
        ! csr_dagger/expm pipeline) still allocates owned copies.
        if (lbound(A%row_starts, 1) == 1) then
            A%owns_local_arrays = .false.
            A%row_starts_local => A%row_starts
            A%col_indexes_local => A%col_indexes
            if (A%has_values .and. associated(A%values)) then
                A%values_local => A%values
            end if
        else
            A%owns_local_arrays = .true.
            ! row_starts_local: 1-based array with 0-based offset values
            allocate (A%row_starts_local(n_local + 1))
            ! Globally indexed: row_starts(lb:ub+1) in original 1-based terms
            do i = 1, int(n_local) + 1
                A%row_starts_local(i) = A%row_starts(partition_table(rank + 1) + i - 2)
            end do

            ! col_indexes_local: 1-based array with 0-based column values (already 0-based)
            allocate (A%col_indexes_local(local_nnz))
            do i = 1, int(local_nnz)
                A%col_indexes_local(i) = A%col_indexes(lb_elem + i - 1)
            end do

            ! Create local values copy (only if has_values is true)
            if (A%has_values .and. associated(A%values)) then
                allocate (A%values_local(local_nnz))
                do i = 1, int(local_nnz)
                    A%values_local(i) = A%values(lb_elem + i - 1)
                end do
            end if
        end if

        ! Call setup_graph_comm (uses 1-based arrays with 0-based column values)
        call setup_graph_comm(A%row_starts_local, A%col_indexes_local, partition_table_64, &
                              MPI_communicator, A%graph_comm, &
                              A%recv_indices_sorted, A%sort_perm, &
                              A%graph_recv_counts, A%graph_recv_disps, &
                              A%send_offsets, A%graph_send_counts, A%graph_send_disps, &
                              A%in_neighbors, A%out_neighbors, &
                              A%total_recv, A%total_send, A%lb_graph, A%ub_graph)

        ! Build hash table for O(1) remote column lookup
        ! hash_keys stores 0-based columns, hash_vals stores 1-based positions
        call build_hash_table(A%recv_indices_sorted, A%total_recv, &
                              A%hash_keys, A%hash_vals, A%hash_size)

        ! recv_indices_sorted is only needed to populate the hash table; the
        ! SpMV path looks up remote columns via A%hash_keys/A%hash_vals, so
        ! release the sorted index array now to avoid carrying total_recv
        ! int64 entries for the propagator's lifetime.
        if (allocated(A%recv_indices_sorted)) deallocate (A%recv_indices_sorted)

        ! Allocate persistent communication buffers on host (1-based)
        ! Always needed: CPU path uses them directly, GPU path uses them for
        ! staging (unless GPU-aware MPI is enabled, but CPU fallback may still occur)
        allocate (A%send_buf(max(A%total_send, 1_int64)))
        allocate (A%recv_buf(max(A%total_recv, 1_int64)))

        A%graph_comm_ready = .true.

        deallocate (partition_table_64)

    end subroutine setup_graph_communications

    !--------------------------------------------------------------------------
    ! Graph-communicator-based SpMV: y = scalar * A * x
    ! Uses neighbor collectives for O(neighbors) scaling
    ! When A%device_ready is true, x and y are assumed to be device-allocated
    !--------------------------------------------------------------------------
    subroutine spmv_graph(A, x_local, partition_table, rank, y_local, &
                          scalar, MPI_communicator)
        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in), target :: x_local
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(in) :: rank
        complex(real64), dimension(:), intent(out), target :: y_local
        complex(real64), intent(in), optional :: scalar
        integer, intent(in) :: MPI_communicator

        complex(real64) :: sc

        if (.not. A%graph_comm_ready) then
            call setup_graph_communications(A, partition_table, MPI_communicator)
        end if

        if (present(scalar)) then
            sc = scalar
        else
            sc = (1.0_real64, 0.0_real64)
        end if

#ifdef USE_HIP
        if (A%device_ready) then
            call spmv_gpu(A, x_local, y_local, sc)
        else
#endif
            ! CPU path (OpenMP)
            call spmv_cpu(A, x_local, y_local, sc)
#ifdef USE_HIP
        end if
#endif

    end subroutine spmv_graph

    !--------------------------------------------------------------------------
    ! Graph-communicator-based SpMM: C = A^n * B
    ! Uses neighbor collectives for O(neighbors) scaling
    ! For n > 1, iterates with temporary storage
    !--------------------------------------------------------------------------
    subroutine spmm_graph(A, n, B_local, partition_table, rank, C_local, MPI_communicator)
        type(CSR), intent(inout) :: A
        integer, intent(in) :: n
        complex(real64), dimension(:, :), intent(in) :: B_local
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(in) :: rank
        complex(real64), dimension(:, :), intent(out) :: C_local
        integer, intent(in) :: MPI_communicator

        integer(int32) :: k, col, n_cols, n_local
        complex(real64), allocatable :: temp_in(:, :), temp_out(:, :)
        complex(real64), allocatable :: col_in(:), col_out(:)

        if (.not. A%graph_comm_ready) then
            call setup_graph_communications(A, partition_table, MPI_communicator)
        end if

        n_local = int(A%ub_graph - A%lb_graph + 1)
        n_cols = size(B_local, 2)

        allocate (col_in(n_local), col_out(n_local))

        if (n == 1) then
            ! Simple case: single multiplication
            do col = 1, n_cols
                col_in = B_local(:, col)
                call spmv_cpu(A, col_in, col_out, (1.0_real64, 0.0_real64))
                C_local(:, col) = col_out
            end do
        else
            ! Multiple multiplications: A^n * B
            allocate (temp_in(n_local, n_cols), temp_out(n_local, n_cols))
            temp_in = B_local

            do k = 1, n
                do col = 1, n_cols
                    col_in = temp_in(:, col)
                    call spmv_cpu(A, col_in, col_out, (1.0_real64, 0.0_real64))
                    temp_out(:, col) = col_out
                end do
                if (k < n) then
                    temp_in = temp_out
                end if
            end do

            C_local = temp_out
            deallocate (temp_in, temp_out)
        end if

        deallocate (col_in, col_out)

    end subroutine spmm_graph

    !--------------------------------------------------------------------------
    ! Cleanup graph communicator resources
    !--------------------------------------------------------------------------
    subroutine cleanup_graph_communications(A)
        type(CSR), intent(inout) :: A
        integer(int32) :: ierr

        if (A%graph_comm /= MPI_COMM_NULL) then
            call MPI_Comm_free(A%graph_comm, ierr)
            A%graph_comm = MPI_COMM_NULL
        end if

        if (allocated(A%recv_indices_sorted)) deallocate (A%recv_indices_sorted)
        if (associated(A%sort_perm)) deallocate (A%sort_perm)
        if (allocated(A%graph_recv_counts)) deallocate (A%graph_recv_counts)
        if (allocated(A%graph_recv_disps)) deallocate (A%graph_recv_disps)
        if (associated(A%send_offsets)) deallocate (A%send_offsets)
        if (allocated(A%graph_send_counts)) deallocate (A%graph_send_counts)
        if (allocated(A%graph_send_disps)) deallocate (A%graph_send_disps)
        if (allocated(A%in_neighbors)) deallocate (A%in_neighbors)
        if (allocated(A%out_neighbors)) deallocate (A%out_neighbors)
        if (associated(A%hash_keys)) deallocate (A%hash_keys)
        if (associated(A%hash_vals)) deallocate (A%hash_vals)
        if (associated(A%send_buf)) deallocate (A%send_buf)
        if (associated(A%recv_buf)) deallocate (A%recv_buf)
        if (A%owns_local_arrays) then
            if (associated(A%row_starts_local)) deallocate (A%row_starts_local)
            if (associated(A%col_indexes_local)) deallocate (A%col_indexes_local)
            if (associated(A%values_local)) deallocate (A%values_local)
        else
            nullify (A%row_starts_local)
            nullify (A%col_indexes_local)
            nullify (A%values_local)
        end if
        A%owns_local_arrays = .false.

        A%graph_comm_ready = .false.

#ifdef USE_HIP
        ! Free device memory if allocated
        if (A%device_ready) then
            call csr_free_device(A)
        end if
#endif

    end subroutine cleanup_graph_communications

#ifdef USE_HIP
    !--------------------------------------------------------------------------
    ! HIP Device Memory Management
    !--------------------------------------------------------------------------

    !--------------------------------------------------------------------------
    ! Allocate device memory for CSR structure and communication buffers
    ! Must be called after setup_graph_communications
    ! ASSUMES: row_starts_local and col_indexes_local contain 0-based values
    !--------------------------------------------------------------------------
    subroutine csr_to_device(A)
        type(CSR), intent(inout) :: A
        integer(c_size_t) :: n_local, nnz_local
        integer(int64), allocatable, target :: send_offsets_0based(:), sort_perm_0based(:)
        integer(int32) :: i

        if (.not. A%graph_comm_ready) then
            error stop "csr_to_device: graph communication not set up"
        end if

        if (A%device_ready) then
            return ! Already on device
        end if

        n_local = size(A%row_starts_local) - 1
        nnz_local = size(A%col_indexes_local)

        ! Allocate CSR structure on device
        call hipCheck(hipMalloc(A%row_starts_dev, int((n_local + 1) * 8, c_size_t)))
        call hipCheck(hipMalloc(A%col_indexes_dev, int(nnz_local * 8, c_size_t)))
        if (A%has_values) then
            call hipCheck(hipMalloc(A%values_dev, int(nnz_local * 16, c_size_t)))
        end if

        ! Allocate communication buffers on device.
        ! These buffers are exchanged via GPU-aware MPI and consumed by kernels.
        ! We keep default coarse-grained allocations and use explicit
        ! hipDeviceSynchronize points around MPI communication in spmv_gpu and
        ! chebyshev_multiply_gpu_impl.
        if (A%total_send > 0) then
            call hipCheck(hipMalloc(A%send_buf_dev, int(A%total_send * 16, c_size_t)))
            call hipCheck(hipMalloc(A%send_offsets_dev, int(A%total_send * 8, c_size_t)))
        end if
        if (A%total_recv > 0) then
            call hipCheck(hipMalloc(A%recv_buf_dev, int(A%total_recv * 16, c_size_t)))
            call hipCheck(hipMalloc(A%recv_buf_sorted_dev, int(A%total_recv * 16, c_size_t)))
            call hipCheck(hipMalloc(A%sort_perm_dev, int(A%total_recv * 8, c_size_t)))
        end if

        ! Allocate hash table on device
        if (A%hash_size > 0) then
            call hipCheck(hipMalloc(A%hash_keys_dev, int(A%hash_size * 8, c_size_t)))
            call hipCheck(hipMalloc(A%hash_vals_dev, int(A%hash_size * 8, c_size_t)))
        end if

        ! Allocate intermediate Aw_k buffer for Chebyshev
        call hipCheck(hipMalloc(A%Aw_k_dev, int(n_local * 16, c_size_t)))

        ! Copy CSR data to device - values are already 0-based, copy directly
        ! Fortran 1-based array becomes 0-based on GPU (element 1 -> index 0)
        call hipCheck(hipMemcpy(A%row_starts_dev, c_loc(A%row_starts_local(1)), &
                                int((n_local + 1) * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(A%col_indexes_dev, c_loc(A%col_indexes_local(1)), &
                                int(nnz_local * 8, c_size_t), hipMemcpyHostToDevice))
        if (A%has_values) then
            call hipCheck(hipMemcpy(A%values_dev, c_loc(A%values_local(1)), &
                                    int(nnz_local * 16, c_size_t), hipMemcpyHostToDevice))
        end if

        ! Copy communication metadata to device (convert to 0-based for GPU)
        if (A%total_send > 0) then
            allocate (send_offsets_0based(A%total_send))
            do i = 1, int(A%total_send)
                send_offsets_0based(i) = A%send_offsets(i) - 1
            end do
            call hipCheck(hipMemcpy(A%send_offsets_dev, c_loc(send_offsets_0based(1)), &
                                    int(A%total_send * 8, c_size_t), hipMemcpyHostToDevice))
            deallocate (send_offsets_0based)
        end if
        if (A%total_recv > 0) then
            allocate (sort_perm_0based(A%total_recv))
            do i = 1, int(A%total_recv)
                sort_perm_0based(i) = A%sort_perm(i) - 1
            end do
            call hipCheck(hipMemcpy(A%sort_perm_dev, c_loc(sort_perm_0based(1)), &
                                    int(A%total_recv * 8, c_size_t), hipMemcpyHostToDevice))
            deallocate (sort_perm_0based)
        end if

        ! Copy hash table to device
        ! hash_keys: 0-based column values (same as col_indexes_local), copy directly
        ! hash_vals: 1-based positions, convert to 0-based for GPU
        if (A%hash_size > 0) then
            block
                integer(int64), allocatable, target :: hash_vals_0based(:)
                allocate (hash_vals_0based(A%hash_size))
                do i = 1, int(A%hash_size)
                    if (A%hash_vals(i) > 0) then
                        hash_vals_0based(i) = A%hash_vals(i) - 1 ! Convert pos to 0-based
                    else
                        hash_vals_0based(i) = -1 ! Not found sentinel for GPU
                    end if
                end do
                call hipCheck(hipMemcpy(A%hash_keys_dev, c_loc(A%hash_keys(1)), &
                                        int(A%hash_size * 8, c_size_t), hipMemcpyHostToDevice))
                call hipCheck(hipMemcpy(A%hash_vals_dev, c_loc(hash_vals_0based(1)), &
                                        int(A%hash_size * 8, c_size_t), hipMemcpyHostToDevice))
                deallocate (hash_vals_0based)
            end block
        end if

        ! Create a HIP stream for async operations
        call hipCheck(hipStreamCreate(A%stream))

        A%device_ready = .true.

    end subroutine csr_to_device

    !--------------------------------------------------------------------------
    ! Free device memory for CSR structure
    !--------------------------------------------------------------------------
    subroutine csr_free_device(A)
        type(CSR), intent(inout) :: A

        if (.not. A%device_ready) return

        ! Free CSR structure
        if (c_associated(A%row_starts_dev)) then
            call hipCheck(hipFree(A%row_starts_dev))
            A%row_starts_dev = c_null_ptr
        end if
        if (c_associated(A%col_indexes_dev)) then
            call hipCheck(hipFree(A%col_indexes_dev))
            A%col_indexes_dev = c_null_ptr
        end if
        if (c_associated(A%values_dev)) then
            call hipCheck(hipFree(A%values_dev))
            A%values_dev = c_null_ptr
        end if

        ! Free communication buffers
        if (c_associated(A%send_buf_dev)) then
            call hipCheck(hipFree(A%send_buf_dev))
            A%send_buf_dev = c_null_ptr
        end if
        if (c_associated(A%recv_buf_dev)) then
            call hipCheck(hipFree(A%recv_buf_dev))
            A%recv_buf_dev = c_null_ptr
        end if
        if (c_associated(A%recv_buf_sorted_dev)) then
            call hipCheck(hipFree(A%recv_buf_sorted_dev))
            A%recv_buf_sorted_dev = c_null_ptr
        end if
        if (c_associated(A%sort_perm_dev)) then
            call hipCheck(hipFree(A%sort_perm_dev))
            A%sort_perm_dev = c_null_ptr
        end if
        if (c_associated(A%send_offsets_dev)) then
            call hipCheck(hipFree(A%send_offsets_dev))
            A%send_offsets_dev = c_null_ptr
        end if

        ! Free hash table
        if (c_associated(A%hash_keys_dev)) then
            call hipCheck(hipFree(A%hash_keys_dev))
            A%hash_keys_dev = c_null_ptr
        end if
        if (c_associated(A%hash_vals_dev)) then
            call hipCheck(hipFree(A%hash_vals_dev))
            A%hash_vals_dev = c_null_ptr
        end if

        ! Free Aw_k buffer
        if (c_associated(A%Aw_k_dev)) then
            call hipCheck(hipFree(A%Aw_k_dev))
            A%Aw_k_dev = c_null_ptr
        end if

        ! Destroy stream
        if (c_associated(A%stream)) then
            call hipCheck(hipStreamDestroy(A%stream))
            A%stream = c_null_ptr
        end if

        A%device_ready = .false.

    end subroutine csr_free_device

    !--------------------------------------------------------------------------
    ! Update values on device (for when values change but structure doesn't)
    !--------------------------------------------------------------------------
    subroutine csr_update_values_device(A)
        type(CSR), intent(inout) :: A
        integer(c_size_t) :: nnz_local

        if (.not. A%device_ready) then
            error stop "csr_update_values_device: device not ready"
        end if

        if (.not. A%has_values) return

        nnz_local = size(A%values_local)
        call hipCheck(hipMemcpy(A%values_dev, c_loc(A%values_local(1)), &
                                int(nnz_local * 16, c_size_t), hipMemcpyHostToDevice))

    end subroutine csr_update_values_device

    !--------------------------------------------------------------------------
    ! GPU-accelerated SpMV with MPI communication
    ! Uses HIP kernels for local/remote SpMV
    !
    ! Flow (standard MPI with host staging):
    !   1. Pack send buffer on device (pack_send_buf kernel)
    !   2. D->H transfer of send buffer
    !   3. MPI non-blocking neighbor alltoallv (host buffers)
    !   4. Local SpMV on device (while MPI proceeds)
    !   5. Wait for MPI
    !   6. H->D transfer of recv buffer
    !   7. Reorder recv buffer on device (reorder_recv_buf kernel)
    !   8. Remote SpMV on device (adds to local result)
    !
    ! Flow (GPU-aware MPI, when QUOP_GPU_AWARE_MPI is defined):
    !   1. Pack send buffer on device (pack_send_buf kernel)
    !   2. (eliminated) - MPI reads directly from device
    !   3. MPI non-blocking neighbor alltoallv (device buffers)
    !   4. Local SpMV on device (while MPI proceeds)
    !   5. Wait for MPI
    !   6. (eliminated) - MPI wrote directly to device
    !   7. Reorder recv buffer on device (reorder_recv_buf kernel)
    !   8. Remote SpMV on device (adds to local result)
    !
    ! x_local and y_local are assumed to be device-allocated arrays.
    ! Uses c_loc to get device pointers.
    !--------------------------------------------------------------------------
    subroutine spmv_gpu(A, x_local, y_local, scalar)
        use hip_sparse_expm_kernels, only: &
            launch_complex_scale_kernel, launch_pack_send_buf_kernel, launch_reorder_recv_buf_kernel, &
            launch_spmv_local_unit_kernel, launch_spmv_local_weighted_kernel, launch_spmv_remote_unit_kernel, &
            launch_spmv_remote_weighted_kernel
        use hipfort_types, only: dim3

        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in), target :: x_local ! Device array
        complex(real64), dimension(:), intent(out), target :: y_local ! Device array
        complex(real64), intent(in) :: scalar

        type(c_ptr) :: x_dev, y_dev
        integer(int32) :: ierr, request, status(MPI_STATUS_SIZE)
        integer(int64) :: n_local, lb_0, ub_0
        integer, parameter :: BLOCKSIZE = 256
        type(dim3) :: grid, block
#ifdef QUOP_GPU_AWARE_MPI
        ! Fortran pointers to device buffers for MPI (c_ptr cannot be passed
        ! directly to MPI -- Fortran would pass the address of the c_ptr variable
        ! on the host, not the device address stored within it).
        complex(real64), dimension(:), pointer :: send_buf_fptr, recv_buf_fptr
#endif

        ! Get device pointers from target arrays
        x_dev = c_loc(x_local(1))
        y_dev = c_loc(y_local(1))

        n_local = A%ub_graph - A%lb_graph + 1

        ! lb_graph and ub_graph are already 0-based (from setup_graph_comm)
        lb_0 = A%lb_graph
        ub_0 = A%ub_graph

        ! Set up kernel launch configuration
        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((n_local + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        ! Step 1: Pack send buffer on device
        if (A%total_send > 0) then
            call launch_pack_send_buf_kernel(grid, block, 0, A%stream, &
                                             x_dev, A%send_offsets_dev, A%send_buf_dev, A%total_send)

#ifdef QUOP_GPU_AWARE_MPI
            ! GPU-aware MPI: ensure packed data is visible before MPI reads
            ! send_buf_dev via RDMA.
            call hipCheck(hipDeviceSynchronize())
#else
            ! Step 2: D->H transfer of send buffer (staging for non-GPU-aware MPI)
            call hipCheck(hipMemcpyAsync(c_loc(A%send_buf(1)), A%send_buf_dev, &
                                         int(A%total_send * 16, c_size_t), hipMemcpyDeviceToHost, A%stream))
            call hipCheck(hipStreamSynchronize(A%stream))
#endif
        end if

        ! Step 3: Start non-blocking MPI exchange
#ifdef QUOP_GPU_AWARE_MPI
        ! GPU-aware MPI: communicate directly with device buffers.
        ! Convert c_ptr to Fortran pointers so MPI receives the device address,
        ! not the host address of the c_ptr variable.
        call c_f_pointer(A%send_buf_dev, send_buf_fptr, [A%total_send])
        call c_f_pointer(A%recv_buf_dev, recv_buf_fptr, [A%total_recv])
        call MPI_Ineighbor_alltoallv(send_buf_fptr, A%graph_send_counts, A%graph_send_disps, &
                                     MPI_DOUBLE_COMPLEX, &
                                     recv_buf_fptr, A%graph_recv_counts, A%graph_recv_disps, &
                                     MPI_DOUBLE_COMPLEX, &
                                     A%graph_comm, request, ierr)
#else
        ! Non-GPU-aware MPI: use host staging buffers
        call MPI_Ineighbor_alltoallv(A%send_buf, A%graph_send_counts, A%graph_send_disps, &
                                     MPI_DOUBLE_COMPLEX, &
                                     A%recv_buf, A%graph_recv_counts, A%graph_recv_disps, &
                                     MPI_DOUBLE_COMPLEX, &
                                     A%graph_comm, request, ierr)
#endif

        ! Step 4: Local SpMV on device (overlaps with MPI)
        ! Computes y = A_local * x (scalar applied in remote phase or after)
        if (A%has_values) then
            call launch_spmv_local_weighted_kernel(grid, block, 0, A%stream, &
                                                   A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                   x_dev, y_dev, lb_0, ub_0, n_local)
        else
            call launch_spmv_local_unit_kernel(grid, block, 0, A%stream, &
                                               A%row_starts_dev, A%col_indexes_dev, &
                                               x_dev, y_dev, lb_0, ub_0, n_local)
        end if

        ! Step 5: Wait for MPI to complete
        call MPI_Wait(request, status, ierr)

        ! Step 6: H->D transfer of recv buffer (only needed for non-GPU-aware MPI)
        if (A%total_recv > 0) then
#ifdef QUOP_GPU_AWARE_MPI
            ! GPU-aware MPI: ensure RDMA writes are visible before kernels read
            ! recv_buf_dev.
            call hipCheck(hipDeviceSynchronize())
#else
            ! Non-GPU-aware MPI: transfer received data from host to device
            call hipCheck(hipMemcpyAsync(A%recv_buf_dev, c_loc(A%recv_buf(1)), &
                                         int(A%total_recv * 16, c_size_t), hipMemcpyHostToDevice, A%stream))
#endif

            ! Step 7: Reorder recv buffer on device
            call launch_reorder_recv_buf_kernel(grid, block, 0, A%stream, &
                                                A%recv_buf_dev, A%sort_perm_dev, A%recv_buf_sorted_dev, A%total_recv)

            ! Step 8: Remote SpMV on device (adds remote contributions to y)
            ! The remote kernel reads from recv_buf_sorted using hash table lookup
            ! and applies: y = scalar * (y + A_remote * recv_buf)
            if (A%has_values) then
                call launch_spmv_remote_weighted_kernel(grid, block, 0, A%stream, &
                                                        A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                 A%recv_buf_sorted_dev, A%hash_keys_dev, A%hash_vals_dev, A%hash_size, &
                                                        y_dev, scalar, lb_0, ub_0, n_local)
            else
                call launch_spmv_remote_unit_kernel(grid, block, 0, A%stream, &
                                                    A%row_starts_dev, A%col_indexes_dev, &
                                                 A%recv_buf_sorted_dev, A%hash_keys_dev, A%hash_vals_dev, A%hash_size, &
                                                    y_dev, scalar, lb_0, ub_0, n_local)
            end if
        end if

        ! If no remote contributions, still need to apply scalar
        if (A%total_recv == 0 .and. abs(scalar - (1.0_real64, 0.0_real64)) > 1.0e-15_real64) then
            call launch_complex_scale_kernel(grid, block, 0, A%stream, &
                                             scalar, y_dev, n_local)
        end if

        ! Synchronize stream before returning
        call hipCheck(hipStreamSynchronize(A%stream))

    end subroutine spmv_gpu
#endif

end module sparse
