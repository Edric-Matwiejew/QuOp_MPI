! test_hash_utils.f90
! Shared hash table utilities for kernel unit tests
! Replicates the hash function used in hip_common.hpp

module test_hash_utils
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding
    implicit none

    private
    public :: compute_hash_pos, build_hash_table

contains

    !--------------------------------------------------------------------------
    ! Compute hash position for a column index
    ! Matches the kernel's hash_lookup_dev function in hip_common.hpp
    ! Returns 1-based position in hash table
    !--------------------------------------------------------------------------
    function compute_hash_pos(col, hash_size) result(hash_pos)
        integer(c_long), intent(in) :: col, hash_size
        integer(c_long) :: hash_pos
        ! Knuth's golden ratio multiplier (must match kernel)
        integer(c_long), parameter :: HASH_MULT = 2654435769_c_long
        integer(c_long), parameter :: MASK32 = int(Z'FFFFFFFF', c_long)
        integer(c_long) :: folded

        ! Fold 64-bit to 32-bit and apply multiplicative hash
        folded = iand(ieor(col, ishft(col, -32)), MASK32)
        hash_pos = mod(folded * HASH_MULT, hash_size) + 1
        if (hash_pos < 1) hash_pos = hash_pos + hash_size
    end function compute_hash_pos

    !--------------------------------------------------------------------------
    ! Build hash table: maps global column indices to recv_buf positions
    ! hash_vals stores 1-based position (kernel subtracts 1 when accessing)
    !
    ! Arguments:
    !   cols      - array of column indices to insert
    !   num_cols  - number of columns
    !   hash_keys - output: hash keys array (must be pre-allocated)
    !   hash_vals - output: hash values array (must be pre-allocated)
    !   hash_size - size of hash table
    !--------------------------------------------------------------------------
    subroutine build_hash_table(cols, num_cols, hash_keys, hash_vals, hash_size)
        integer(c_long), intent(in) :: cols(:)
        integer, intent(in) :: num_cols
        integer(c_long), intent(out) :: hash_keys(:), hash_vals(:)
        integer(c_long), intent(in) :: hash_size
        integer(c_long) :: hash_pos, col
        integer(int32) :: i, probe

        ! Initialize with -1 (empty marker)
        hash_keys = -1_c_long
        hash_vals = -1_c_long

        do i = 1, num_cols
            col = cols(i)
            hash_pos = compute_hash_pos(col, hash_size)

            ! Linear probing for collision resolution
            do probe = 0, int(hash_size) - 1
                if (hash_keys(hash_pos) < 0) then
                    hash_keys(hash_pos) = col
                    hash_vals(hash_pos) = int(i, c_long) ! 1-based position in recv_buf
                    exit
                end if
                ! Wrap around: (pos - 1) mod size + 1 for 1-based indexing
                hash_pos = mod(hash_pos, hash_size) + 1
            end do
        end do
    end subroutine build_hash_table

end module test_hash_utils
