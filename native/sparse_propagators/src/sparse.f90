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
        launch_complex_scale_kernel, launch_pack_send_buf_kernel, &
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
        ! Persistent communication buffers
        complex(real64), dimension(:), pointer :: send_buf => null()
        complex(real64), dimension(:), pointer :: recv_buf => null()
        ! Local values for SpMV (slice of global values)
        complex(real64), dimension(:), pointer :: values_local => null()
        ! True when *_local pointers own freshly allocated storage (legacy
        ! globally-indexed path); false when they alias the borrowed
        ! row_starts/col_indexes/values buffers (locally-indexed path).
        logical :: owns_local_arrays = .false.
        ! Halo SpMV metadata (built once in setup_graph_communications, used
        ! by spmv_cpu in place of the hash lookup + recv_buf reorder).
        ! col_halo(j) is a 0-based index into a virtual halo'd vector of
        ! length n_local + total_recv: values in [0, n_local) reference
        ! x_local; values in [n_local, n_local+total_recv) reference recv_buf
        ! at offset (col_halo(j) - n_local). diag_lo/diag_hi delimit the
        ! contiguous run of locally-owned columns within each row (1-based
        ! inclusive indices into col_indexes_local / col_halo / values_local;
        ! diag_lo > diag_hi means the row has no diagonal-block entries).
        integer(int64), dimension(:), pointer :: col_halo => null()
        integer(int64), dimension(:), pointer :: diag_lo => null()
        integer(int64), dimension(:), pointer :: diag_hi => null()
        ! Always .false. in current design (col_halo aliases col_indexes_local);
        ! retained for forward compatibility with freshly-allocated halo storage.
        logical :: owns_col_halo = .false.
        ! Flag to indicate if graph comm is set up
        logical :: graph_comm_ready = .false.
        ! Flag to indicate if values are explicit (false = all ones)
        logical :: has_values = .true.

#ifdef USE_HIP
        !----------------------------------------------------------------------
        ! HIP/GPU device memory (only used when GPU backend is active)
        ! These are c_ptr to device memory, initialized to c_null_ptr
        !----------------------------------------------------------------------
        ! CSR structure on device.  col_indexes_dev holds the rewritten
        ! col_halo offsets uploaded by csr_to_device (matching the CPU path).
        type(c_ptr) :: row_starts_dev = c_null_ptr
        type(c_ptr) :: col_indexes_dev = c_null_ptr
        type(c_ptr) :: values_dev = c_null_ptr
        ! Per-row diagonal-block delimiters on device (0-based half-open
        ! ranges into col_indexes_dev / values_dev: diagonal entries are
        ! [diag_lo_dev[i], diag_hi_dev[i]] inclusive; off-diagonal entries
        ! occupy the rest of the row).  Allocated alongside the CSR.
        type(c_ptr) :: diag_lo_dev = c_null_ptr
        type(c_ptr) :: diag_hi_dev = c_null_ptr
        ! Communication buffers on device
        type(c_ptr) :: send_buf_dev = c_null_ptr
        type(c_ptr) :: recv_buf_dev = c_null_ptr
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
    ! Build halo metadata for spmv_cpu / spmv_gpu: col_halo + diag_lo + diag_hi.
    !
    ! Pre-conditions:
    !   - A%row_starts_local, A%col_indexes_local, A%lb_graph, A%ub_graph
    !     populated; the CSR is column-sorted within each row.
    !   - A%recv_indices_sorted and A%sort_perm populated.
    !
    ! col_halo aliases A%col_indexes_local; the borrowed CSR column array is
    ! rewritten in place to hold halo offsets.  This saves 8 * local_nnz
    ! bytes of permanent metadata per rank.  Remote-column lookup uses a
    ! binary search over A%recv_indices_sorted; no hash table is built.
    !
    ! On GPU builds the rewritten col_indexes_local is later uploaded to the
    ! device by csr_to_device, so col_indexes_dev holds the halo offsets and
    ! the device-side SpMV reuses the same metadata as the CPU path.
    !--------------------------------------------------------------------------
    subroutine build_halo_metadata(A, n_local)
        type(CSR), intent(inout) :: A
        integer(int64), intent(in) :: n_local

        integer(int64) :: i, j, row_lo, row_hi, col, sorted_pos

        ! Alias col_halo to col_indexes_local; the loop below rewrites the
        ! columns in place. col_indexes_local itself either borrows the
        ! Python CSR buffer or owns storage from the legacy globally-indexed
        ! path; cleanup_graph_communications will nullify col_halo and let
        ! the existing owns_local_arrays logic handle the underlying memory.
        A%col_halo => A%col_indexes_local
        A%owns_col_halo = .false.

        allocate (A%diag_lo(max(n_local, 1_int64)))
        allocate (A%diag_hi(max(n_local, 1_int64)))

        ! Per row: locate the contiguous diagonal-block segment using binary
        ! search on the column-sorted CSR, then translate every entry's
        ! global column into a halo offset.
        !$omp parallel do private(row_lo, row_hi, j, col, sorted_pos)
        do i = 1, n_local
            row_lo = A%row_starts_local(i) + 1            ! 1-based first
            row_hi = A%row_starts_local(i + 1)            ! 1-based last (inclusive)

            A%diag_lo(i) = lower_bound(A%col_indexes_local, row_lo, row_hi, A%lb_graph)
            A%diag_hi(i) = upper_bound(A%col_indexes_local, row_lo, row_hi, A%ub_graph) - 1

            do j = row_lo, row_hi
                col = A%col_indexes_local(j)
                if (col >= A%lb_graph .and. col <= A%ub_graph) then
                    ! Owned column: 0-based local index in [0, n_local)
                    A%col_halo(j) = col - A%lb_graph
                else
                    ! Remote column: binary search the sorted recv index
                    ! array directly, then translate via sort_perm into a
                    ! 0-based offset into recv_buf, biased by n_local so a
                    ! single index distinguishes diagonal vs halo entries.
                    sorted_pos = lower_bound(A%recv_indices_sorted, &
                                             1_int64, A%total_recv, col)
                    A%col_halo(j) = n_local + A%sort_perm(sorted_pos) - 1
                end if
            end do
        end do
        !$omp end parallel do
    end subroutine build_halo_metadata

    !--------------------------------------------------------------------------
    ! CPU SpMV using graph communicator and prebuilt halo metadata (OpenMP)
    !
    ! Each row's entries split into three contiguous segments by column:
    !   off-lower   : col < lb_graph                  (j in [row_lo, diag_lo - 1])
    !   diagonal    : lb_graph <= col <= ub_graph     (j in [diag_lo, diag_hi])
    !   off-upper   : col > ub_graph                  (j in [diag_hi + 1, row_hi])
    !
    ! col_halo(j) carries the appropriate halo offset:
    !   diagonal entries -> 0-based local index into x_local
    !   off entries      -> n_local + 0-based offset into recv_buf
    ! eliminating the per-SpMV hash lookup and recv_buf reorder.
    !--------------------------------------------------------------------------
    subroutine spmv_cpu(A, x_local, y_local, scalar)
        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in) :: x_local
        complex(real64), dimension(:), intent(out) :: y_local
        complex(real64), intent(in) :: scalar

        integer(int32) :: ierr, request
        integer(int64) :: i, n_local, j, idx
        integer(int64) :: row_lo, row_hi, diag_first, diag_last
        complex(real64) :: row_sum
        integer(int32) :: status(MPI_STATUS_SIZE)

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

        ! Diagonal-block phase: only locally-owned columns.  Runs while
        ! comms is in flight.  diag_first > diag_last skips empty rows.
        if (A%has_values) then
            !$omp parallel do private(diag_first, diag_last, row_sum, j, idx)
            do i = 1, n_local
                diag_first = A%diag_lo(i)
                diag_last = A%diag_hi(i)
                row_sum = (0.0_real64, 0.0_real64)
                do j = diag_first, diag_last
                    idx = A%col_halo(j)                ! 0-based local index
                    row_sum = row_sum + A%values_local(j) * x_local(idx + 1)
                end do
                y_local(i) = row_sum
            end do
            !$omp end parallel do
        else
            !$omp parallel do private(diag_first, diag_last, row_sum, j, idx)
            do i = 1, n_local
                diag_first = A%diag_lo(i)
                diag_last = A%diag_hi(i)
                row_sum = (0.0_real64, 0.0_real64)
                do j = diag_first, diag_last
                    idx = A%col_halo(j)
                    row_sum = row_sum + x_local(idx + 1)
                end do
                y_local(i) = row_sum
            end do
            !$omp end parallel do
        end if

        ! Wait for communication
        call MPI_Wait(request, status, ierr)

        ! Off-diagonal phase: remaining columns.  col_halo(j) - n_local is a
        ! 0-based offset into recv_buf, valid because halo offsets were
        ! populated in recv_buf order at setup time.
        if (A%has_values) then
            !$omp parallel do private(row_lo, row_hi, diag_first, diag_last, row_sum, j, idx)
            do i = 1, n_local
                row_lo = A%row_starts_local(i) + 1
                row_hi = A%row_starts_local(i + 1)
                diag_first = A%diag_lo(i)
                diag_last = A%diag_hi(i)
                row_sum = y_local(i)

                do j = row_lo, diag_first - 1
                    idx = A%col_halo(j) - n_local      ! 0-based recv_buf offset
                    row_sum = row_sum + A%values_local(j) * A%recv_buf(idx + 1)
                end do
                do j = diag_last + 1, row_hi
                    idx = A%col_halo(j) - n_local
                    row_sum = row_sum + A%values_local(j) * A%recv_buf(idx + 1)
                end do

                y_local(i) = scalar * row_sum
            end do
            !$omp end parallel do
        else
            !$omp parallel do private(row_lo, row_hi, diag_first, diag_last, row_sum, j, idx)
            do i = 1, n_local
                row_lo = A%row_starts_local(i) + 1
                row_hi = A%row_starts_local(i + 1)
                diag_first = A%diag_lo(i)
                diag_last = A%diag_hi(i)
                row_sum = y_local(i)

                do j = row_lo, diag_first - 1
                    idx = A%col_halo(j) - n_local
                    row_sum = row_sum + A%recv_buf(idx + 1)
                end do
                do j = diag_last + 1, row_hi
                    idx = A%col_halo(j) - n_local
                    row_sum = row_sum + A%recv_buf(idx + 1)
                end do

                y_local(i) = scalar * row_sum
            end do
            !$omp end parallel do
        end if

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
    !>
    !> @warning Reads `A%col_indexes` directly, NOT `A%col_indexes_local`.
    !> On CPU builds (`#ifndef USE_HIP`), `setup_graph_communications` rewrites
    !> `col_indexes_local` (which aliases `col_indexes` for the locally-indexed
    !> Python path) in place to hold halo offsets rather than global column
    !> indices.  Therefore `csr_dagger` must only be called BEFORE
    !> `setup_graph_communications`, or on an `A` whose `col_indexes` array is
    !> known to be untouched (the legacy globally-indexed `expm.f90` path).
    !> The live Chebyshev pipeline does not call this routine.

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

#ifdef USE_HIP
        ! No hash table is built any more: the GPU SpMV reuses the col_halo
        ! offsets uploaded via csr_to_device, exactly like the CPU path.
#endif

        ! Build halo metadata. col_halo aliases the borrowed col_indexes_local
        ! and is mutated in place to hold halo offsets; remote columns are
        ! resolved by binary search over recv_indices_sorted. On GPU builds
        ! csr_to_device later uploads the rewritten col_indexes_local so the
        ! device-side SpMV reuses the same metadata.
        call build_halo_metadata(A, n_local)

        ! recv_indices_sorted and sort_perm are no longer needed once
        ! col_halo is populated. Release them to recover ~2*total_recv int64
        ! entries per rank on both CPU and GPU builds.
        if (allocated(A%recv_indices_sorted)) deallocate (A%recv_indices_sorted)
        if (associated(A%sort_perm)) deallocate (A%sort_perm)

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
        if (associated(A%send_buf)) deallocate (A%send_buf)
        if (associated(A%recv_buf)) deallocate (A%recv_buf)
        if (A%owns_col_halo) then
            if (associated(A%col_halo)) deallocate (A%col_halo)
        else
            nullify (A%col_halo)
        end if
        A%owns_col_halo = .false.
        if (associated(A%diag_lo)) deallocate (A%diag_lo)
        if (associated(A%diag_hi)) deallocate (A%diag_hi)
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
        integer(int32) :: i

        if (.not. A%graph_comm_ready) then
            error stop "csr_to_device: graph communication not set up"
        end if

        if (A%device_ready) then
            return ! Already on device
        end if

        n_local = size(A%row_starts_local) - 1
        nnz_local = size(A%col_indexes_local)

        ! Allocate CSR structure on device.  col_indexes_dev holds halo
        ! offsets (col_halo) -- col_indexes_local was rewritten in place by
        ! build_halo_metadata, so a direct copy gives us the same metadata
        ! the CPU SpMV uses.
        call hipCheck(hipMalloc(A%row_starts_dev, int((n_local + 1) * 8, c_size_t)))
        call hipCheck(hipMalloc(A%col_indexes_dev, int(nnz_local * 8, c_size_t)))
        if (A%has_values) then
            call hipCheck(hipMalloc(A%values_dev, int(nnz_local * 16, c_size_t)))
        end if

        ! Per-row diagonal-block delimiters (1-based inclusive ranges into
        ! col_indexes_dev / values_dev, matching A%diag_lo / A%diag_hi on the
        ! host).  Allocated even when n_local == 0 so the device pointers
        ! are always valid for kernel launches with zero rows.
        call hipCheck(hipMalloc(A%diag_lo_dev, int(max(n_local, 1_c_size_t) * 8, c_size_t)))
        call hipCheck(hipMalloc(A%diag_hi_dev, int(max(n_local, 1_c_size_t) * 8, c_size_t)))

        ! Coarse-grained device allocations for GPU-aware MPI buffers; explicit
        ! hipDeviceSynchronize points guard MPI in spmv_gpu / chebyshev_multiply_gpu_impl.
        if (A%total_send > 0) then
            call hipCheck(hipMalloc(A%send_buf_dev, int(A%total_send * 16, c_size_t)))
            call hipCheck(hipMalloc(A%send_offsets_dev, int(A%total_send * 8, c_size_t)))
        end if
        if (A%total_recv > 0) then
            call hipCheck(hipMalloc(A%recv_buf_dev, int(A%total_recv * 16, c_size_t)))
        end if

        ! Allocate intermediate Aw_k buffer for Chebyshev
        call hipCheck(hipMalloc(A%Aw_k_dev, int(n_local * 16, c_size_t)))

        ! Copy CSR data to device.  row_starts_local stores 0-based offsets;
        ! col_indexes_local stores the halo offsets produced by
        ! build_halo_metadata (0-based indices into the virtual halo'd
        ! vector of length n_local + total_recv).
        call hipCheck(hipMemcpy(A%row_starts_dev, c_loc(A%row_starts_local(1)), &
                                int((n_local + 1) * 8, c_size_t), hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(A%col_indexes_dev, c_loc(A%col_indexes_local(1)), &
                                int(nnz_local * 8, c_size_t), hipMemcpyHostToDevice))
        if (A%has_values) then
            call hipCheck(hipMemcpy(A%values_dev, c_loc(A%values_local(1)), &
                                    int(nnz_local * 16, c_size_t), hipMemcpyHostToDevice))
        end if

        ! Copy diagonal-block delimiters.  The host arrays are 1-based
        ! inclusive Fortran indices; the kernels treat the lower bound as
        ! 1-based and convert internally so the device sees the same
        ! semantics.
        if (n_local > 0) then
            call hipCheck(hipMemcpy(A%diag_lo_dev, c_loc(A%diag_lo(1)), &
                                    int(n_local * 8, c_size_t), hipMemcpyHostToDevice))
            call hipCheck(hipMemcpy(A%diag_hi_dev, c_loc(A%diag_hi(1)), &
                                    int(n_local * 8, c_size_t), hipMemcpyHostToDevice))
        end if

        ! Copy communication metadata to device.  setup_graph_comm produced
        ! 1-based send_offsets for the legacy CPU path; mutate them in place
        ! to 0-based so the GPU upload skips a temporary buffer.  After this
        ! point the host send_offsets are owned by the GPU path only and the
        ! 0-based representation is the canonical one.
        if (A%total_send > 0) then
            do i = 1, int(A%total_send)
                A%send_offsets(i) = A%send_offsets(i) - 1
            end do
            call hipCheck(hipMemcpy(A%send_offsets_dev, c_loc(A%send_offsets(1)), &
                                    int(A%total_send * 8, c_size_t), hipMemcpyHostToDevice))
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
        if (c_associated(A%diag_lo_dev)) then
            call hipCheck(hipFree(A%diag_lo_dev))
            A%diag_lo_dev = c_null_ptr
        end if
        if (c_associated(A%diag_hi_dev)) then
            call hipCheck(hipFree(A%diag_hi_dev))
            A%diag_hi_dev = c_null_ptr
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
        if (c_associated(A%send_offsets_dev)) then
            call hipCheck(hipFree(A%send_offsets_dev))
            A%send_offsets_dev = c_null_ptr
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
    ! GPU-accelerated SpMV with MPI communication.
    ! Uses the same halo-based two-phase design as spmv_cpu: a single
    ! col_halo array drives both the diagonal phase (read x_local) and the
    ! off-diagonal phase (read recv_buf), with diag_lo / diag_hi delimiting
    ! the contiguous diagonal-block segment of each row.
    !
    ! Flow (standard MPI with host staging):
    !   1. Pack send buffer on device (pack_send_buf kernel)
    !   2. D->H transfer of send buffer
    !   3. MPI non-blocking neighbor alltoallv (host buffers)
    !   4. Diagonal SpMV on device (overlaps with MPI)
    !   5. Wait for MPI
    !   6. H->D transfer of recv buffer
    !   7. Off-diagonal SpMV on device (adds to y, applies scalar)
    !
    ! Flow (GPU-aware MPI, when QUOP_GPU_AWARE_MPI is defined):
    !   Steps 2 and 6 are eliminated; MPI reads/writes recv_buf_dev directly.
    !
    ! x_local and y_local are assumed to be device-allocated arrays.
    ! Uses c_loc to get device pointers.
    !--------------------------------------------------------------------------
    subroutine spmv_gpu(A, x_local, y_local, scalar)
        use hip_sparse_expm_kernels, only: &
            launch_complex_scale_kernel, launch_pack_send_buf_kernel, &
            launch_spmv_local_unit_kernel, launch_spmv_local_weighted_kernel, &
            launch_spmv_remote_unit_kernel, launch_spmv_remote_weighted_kernel
        use hipfort_types, only: dim3

        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in), target :: x_local ! Device array
        complex(real64), dimension(:), intent(out), target :: y_local ! Device array
        complex(real64), intent(in) :: scalar

        type(c_ptr) :: x_dev, y_dev
        integer(int32) :: ierr, request, status(MPI_STATUS_SIZE)
        integer(int64) :: n_local
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

        ! Step 4: Diagonal SpMV on device (overlaps with MPI)
        ! y = A_diag * x_local (scalar applied in remote phase or after)
        if (A%has_values) then
            call launch_spmv_local_weighted_kernel(grid, block, 0, A%stream, &
                                                   A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                   A%diag_lo_dev, A%diag_hi_dev, x_dev, y_dev, n_local)
        else
            call launch_spmv_local_unit_kernel(grid, block, 0, A%stream, &
                                               A%row_starts_dev, A%col_indexes_dev, &
                                               A%diag_lo_dev, A%diag_hi_dev, x_dev, y_dev, n_local)
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

            ! Step 7: Off-diagonal SpMV on device (adds remote contributions to y,
            ! applies scalar). Reads recv_buf via col_halo - n_local.
            if (A%has_values) then
                call launch_spmv_remote_weighted_kernel(grid, block, 0, A%stream, &
                                                        A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                        A%diag_lo_dev, A%diag_hi_dev, A%recv_buf_dev, &
                                                        y_dev, scalar, n_local, n_local)
            else
                call launch_spmv_remote_unit_kernel(grid, block, 0, A%stream, &
                                                    A%row_starts_dev, A%col_indexes_dev, &
                                                    A%diag_lo_dev, A%diag_hi_dev, A%recv_buf_dev, &
                                                    y_dev, scalar, n_local, n_local)
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
