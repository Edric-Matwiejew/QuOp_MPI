module shafft_comm_dims_utils

    use, intrinsic :: iso_fortran_env, only: int32, int64

    implicit none

    private
    public :: comm_dims_contiguous_prefer_low_nda

contains

    subroutine comm_dims_contiguous_prefer_low_nda(nranks, global_shape, &
                                                   comm_dims, nda, used_ranks, ierr)
        integer(int32), intent(in) :: nranks
        integer(int32), intent(in) :: global_shape(:)
        integer(int32), intent(out) :: comm_dims(:)
        integer(int32), intent(out) :: nda
        integer(int32), intent(out) :: used_ranks
        integer(int32), intent(out) :: ierr

        integer(int32) :: ndim, d, nca, i
        integer(int32) :: best_nda, best_used, cap_last, max_last, used32
        integer(int32), allocatable :: best_comm(:), comm(:)
        integer(int64) :: prefix, nranks64, used64
        logical :: valid

        ierr = 0
        ndim = size(global_shape)
        if (size(comm_dims) /= ndim) then
            ierr = 1
            nda = 0
            used_ranks = 0
            return
        end if

        if (nranks < 1 .or. ndim < 1) then
            ierr = 2
            nda = 0
            used_ranks = 0
            comm_dims = 1
            return
        end if
        if (any(global_shape < 1)) then
            ierr = 3
            nda = 0
            used_ranks = 0
            comm_dims = 1
            return
        end if

        allocate (best_comm(ndim), comm(ndim))

        best_comm = 1
        best_nda = 0
        best_used = 1
        nranks64 = int(nranks, int64)

        do d = 1, min(1, ndim - 1)
            nca = ndim - d
            comm = 1
            valid = .true.

            do i = 1, d - 1
                if (global_shape(i) > global_shape(i + nca)) then
                    valid = .false.
                    exit
                end if
                comm(i) = global_shape(i)
            end do
            if (.not. valid) cycle

            prefix = 1_int64
            if (d > 1) then
                do i = 1, d - 1
                    prefix = prefix * int(comm(i), int64)
                end do
            end if
            if (prefix > nranks64) cycle

            cap_last = min(global_shape(d), global_shape(d + nca))
            max_last = min(cap_last, int(nranks64 / prefix, int32))
            if (max_last < 2) cycle

            comm(d) = max_last
            used64 = prefix * int(max_last, int64)
            if (used64 > int(huge(used_ranks), int64)) cycle

            used32 = int(used64, int32)
            if (used32 > best_used .or. &
                (used32 == best_used .and. d < best_nda)) then
                best_comm = comm
                best_nda = d
                best_used = used32
            end if
        end do

        comm_dims = best_comm
        nda = best_nda
        used_ranks = best_used

    end subroutine comm_dims_contiguous_prefer_low_nda

end module shafft_comm_dims_utils
