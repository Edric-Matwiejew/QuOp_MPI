!------------------------------------------------------------------------------
!> @brief Shared transverse-field propagator logic for MPI and wavefront backends.
!>
!> @details Pure host-side routines for layout classification, partition table
!> conversion, segment computation, and segment sorting. Both mpi_transverse_field
!> and wavefront_transverse_field consume these routines; the actual data movement
!> (local pair updates, MPI exchanges, GPU kernels) lives in each backend wrapper.
!------------------------------------------------------------------------------
module transverse_field_common

    use, intrinsic :: iso_fortran_env, only: real64, int32, int64, error_unit

    implicit none

    private

    public :: TF_MODE_UNSET, TF_MODE_ALIGNED, TF_MODE_SEGMENTED
    public :: COMPLEX128_BYTES, DEFAULT_CHUNK_BYTES
    public :: tf_is_power_of_two
    public :: tf_exact_log2
    public :: tf_find_owner_0
    public :: tf_classify_layout
    public :: tf_copy_partition_table_0
    public :: tf_current_half_band_end
    public :: tf_partner_delta
    public :: tf_max_segment_end
    public :: tf_sort_remote_segments
    public :: tf_grow_remote_arrays

    public :: tf_segment_t

    integer(int64), parameter :: COMPLEX128_BYTES = 16_int64
    integer(int64), parameter :: DEFAULT_CHUNK_BYTES = 67108864_int64
    integer(int32), parameter :: TF_MODE_UNSET = 0_int32
    integer(int32), parameter :: TF_MODE_ALIGNED = 1_int32
    integer(int32), parameter :: TF_MODE_SEGMENTED = 2_int32

    type tf_segment_t
        integer(int64) :: g0 = 0_int64
        integer(int64) :: g1 = -1_int64
        integer(int64) :: delta = 0_int64
        integer(int32) :: owner = -1_int32
        integer(int64) :: exchange_key = 0_int64
    end type tf_segment_t

contains

    pure logical function tf_is_power_of_two(value)
        integer(int64), intent(in) :: value

        tf_is_power_of_two = (value > 0_int64 .and. iand(value, value - 1_int64) == 0_int64)
    end function tf_is_power_of_two

    pure subroutine tf_exact_log2(value, exponent, is_exact)
        integer(int64), intent(in) :: value
        integer(int32), intent(out) :: exponent
        logical, intent(out) :: is_exact
        integer(int64) :: tmp

        exponent = 0
        is_exact = .false.
        if (value <= 0_int64) return

        tmp = value
        do while (mod(tmp, 2_int64) == 0_int64 .and. tmp > 1_int64)
            tmp = tmp / 2_int64
            exponent = exponent + 1_int32
        end do

        is_exact = (tmp == 1_int64)
    end subroutine tf_exact_log2

    pure integer(int32) function tf_find_owner_0(col, partition_table_0) result(owner)
        integer(int64), intent(in) :: col
        integer(int64), intent(in) :: partition_table_0(:)
        integer(int32) :: lo, hi, mid

        lo = 1
        hi = size(partition_table_0) - 1
        do while (lo < hi)
            mid = (lo + hi + 1) / 2
            if (partition_table_0(mid) <= col) then
                lo = mid
            else
                hi = mid - 1
            end if
        end do

        owner = lo - 1
    end function tf_find_owner_0

    subroutine tf_classify_layout(system_size, local_i, local_i_offset, &
                                  comm_size, rank, layout_mode, n_local_qubits, error_code)
        integer(int64), intent(in) :: system_size, local_i, local_i_offset
        integer(int32), intent(in) :: comm_size, rank
        integer(int32), intent(out) :: layout_mode, n_local_qubits, error_code

        logical :: exact_power

        error_code = 0
        layout_mode = TF_MODE_SEGMENTED
        n_local_qubits = 0

        if (tf_is_power_of_two(local_i) .and. &
            tf_is_power_of_two(int(comm_size, int64)) .and. &
            local_i * int(comm_size, int64) == system_size .and. &
            local_i_offset == int(rank, int64) * local_i) then
            call tf_exact_log2(local_i, n_local_qubits, exact_power)
            if (.not. exact_power) then
                error_code = 1
                return
            end if
            layout_mode = TF_MODE_ALIGNED
        end if
    end subroutine tf_classify_layout

    subroutine tf_copy_partition_table_0(partition_table_1based, comm_size, rank, &
                                         partition_table_0, lb_global, ub_global, error_code)
        integer(int64), intent(in) :: partition_table_1based(:)
        integer(int32), intent(in) :: comm_size, rank
        integer(int64), allocatable, intent(inout) :: partition_table_0(:)
        integer(int64), intent(out) :: lb_global, ub_global
        integer(int32), intent(out) :: error_code

        integer(int32) :: idx

        error_code = 0

        if (size(partition_table_1based) /= comm_size + 1) then
            write (error_unit, '(A,I0,A,I0)') &
                'ERROR: transverse_field partition_table size mismatch: got ', &
                size(partition_table_1based), ', expected ', comm_size + 1
            error_code = 1
            return
        end if

        if (allocated(partition_table_0)) then
            deallocate (partition_table_0)
        end if
        allocate (partition_table_0(size(partition_table_1based)))

        do idx = 1, size(partition_table_1based)
            partition_table_0(idx) = partition_table_1based(idx) - 1_int64
        end do

        lb_global = partition_table_0(rank + 1)
        ub_global = partition_table_0(rank + 2) - 1_int64
    end subroutine tf_copy_partition_table_0

    pure integer(int64) function tf_current_half_band_end(g, bit_mask)
        integer(int64), intent(in) :: g, bit_mask

        tf_current_half_band_end = ((g / bit_mask) + 1_int64) * bit_mask - 1_int64
    end function tf_current_half_band_end

    pure integer(int64) function tf_partner_delta(g, bit_mask)
        integer(int64), intent(in) :: g, bit_mask

        if (iand(g, bit_mask) == 0_int64) then
            tf_partner_delta = bit_mask
        else
            tf_partner_delta = -bit_mask
        end if
    end function tf_partner_delta

    pure integer(int64) function tf_max_segment_end( &
        g, bit_mask, delta, owner, ub_global, partition_table_0)
        integer(int64), intent(in) :: g, bit_mask, delta, ub_global
        integer(int32), intent(in) :: owner
        integer(int64), intent(in) :: partition_table_0(:)

        integer(int64) :: half_end, owner_ub, owner_end

        half_end = tf_current_half_band_end(g, bit_mask)
        owner_ub = partition_table_0(owner + 2) - 1_int64
        owner_end = owner_ub - delta

        tf_max_segment_end = min(ub_global, half_end, owner_end)
    end function tf_max_segment_end

    subroutine tf_sort_remote_segments(segments, n)
        type(tf_segment_t), intent(inout) :: segments(:)
        integer(int32), intent(in) :: n

        integer(int32) :: idx, insert_at
        type(tf_segment_t) :: tmp

        do idx = 2, n
            tmp = segments(idx)
            insert_at = idx - 1
            do while (insert_at >= 1 .and. segments(insert_at)%exchange_key > tmp%exchange_key)
                segments(insert_at + 1) = segments(insert_at)
                insert_at = insert_at - 1
            end do
            segments(insert_at + 1) = tmp
        end do
    end subroutine tf_sort_remote_segments

    subroutine tf_grow_remote_arrays(segments, cap)
        type(tf_segment_t), allocatable, intent(inout) :: segments(:)
        integer(int32), intent(inout) :: cap

        integer(int32) :: new_cap
        type(tf_segment_t), allocatable :: tmp(:)

        if (cap == 0) then
            new_cap = 8
        else
            new_cap = cap * 2
        end if

        allocate (tmp(new_cap))
        if (cap > 0) tmp(1:cap) = segments(1:cap)
        call move_alloc(tmp, segments)

        cap = new_cap
    end subroutine tf_grow_remote_arrays

end module transverse_field_common
