module csr_generators

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use iso_c_binding, only: c_f_pointer, c_ptr

    implicit none

    private

    public :: hypercube, qmoa_mixer

contains

    subroutine hypercube(N, &
                         lb, &
                         ub, &
                         row_starts, &
                         col_indexes)

        integer(int64), intent(in) :: N
        integer(int64), intent(in) :: lb
        integer(int64), intent(in) :: ub
        integer(int64), target, dimension(ub - lb + 2), intent(out) :: row_starts
        integer(int64), target, dimension(N*(lb - 1) + 1:N*ub), intent(out) :: col_indexes

        integer(int64) :: local_rows
        integer(int64) :: columns

        integer(int64), dimension(N) :: powers, binary

        integer(int64) :: dec

        integer(int64) :: i, j, k, temp

        powers = [(2**i, i=0, N - 1)]

        local_rows = ub - lb + 1
        columns = 2**N

        !$omp parallel do private(binary, dec, i, j, k, temp)
        do j = lb - 1, ub - 1
            binary = mod(j / powers, 2_int64)
            do i = 1, N
                if (binary(i) == 0) then
                    binary(i) = 1
                elseif (binary(i) == 1) then
                    binary(i) = 0
                end if
                dec = sum(binary * powers)
                if (binary(i) == 0) then
                    binary(i) = 1
                elseif (binary(i) == 1) then
                    binary(i) = 0
                end if
                if (N * j + i < 1) then
                    exit
                end if
                col_indexes(N * j + i) = dec + 1
            end do

            ! sort
            do i = j * N + 2, j * N + N
                temp = col_indexes(i)
                do k = i - 1, j * N + 1, -1
                    if (col_indexes(k) <= temp) exit
                    col_indexes(k + 1) = col_indexes(k)
                end do
                col_indexes(k + 1) = temp
            end do

        end do
        !$omp end parallel do

        row_starts(1) = (lb - 1) * N + 1
        row_starts(2:local_rows + 1) = N

        do i = 1, local_rows
            row_starts(i + 1) = row_starts(i + 1) + row_starts(i)
        end do

    end subroutine hypercube

    subroutine qmoa_mixer(local_i, &
                          local_i_offset, &
                          n_dim, &
                          Ns, &
                          ts, &
                          G_ptrs, &
                          elements_per_row, &
                          row_starts, &
                          col_indexes, &
                          values)

        integer(int64), intent(in) :: local_i, local_i_offset, n_dim, elements_per_row
        integer(int64), dimension(n_dim), intent(in) :: Ns
        !f2py integer(int64), intent(in) :: G_ptrs
        type(c_ptr), dimension(n_dim), intent(in) :: G_ptrs
        real(real64), dimension(n_dim), intent(in) :: ts
        integer(int64), dimension(local_i + 1), intent(out) :: row_starts
        integer(int64), dimension(elements_per_row*local_i), intent(out) :: col_indexes
        complex(real64), dimension(elements_per_row*local_i), intent(out) :: values

        real(real64), dimension(:, :), pointer :: G
        integer(int64), dimension(n_dim) :: inds, strides
        integer(int64) :: i, j, k, indx, bak_indx

        strides(1) = 1
        do i = 2, n_dim
            strides(i) = strides(i - 1) * Ns(i - 1)
        end do

        row_starts(1) = local_i_offset * elements_per_row + 1
        do i = 1, local_i
            row_starts(i + 1) = row_starts(i) + elements_per_row
        end do

        indx = 1
        do i = local_i_offset, local_i_offset + local_i - 1
            do j = 1, n_dim
                inds(j) = mod(i / strides(j), Ns(j))
            end do
            do j = 1, n_dim
                bak_indx = inds(j)
                call c_f_pointer(G_ptrs(j), G, [Ns(j), Ns(j)])
                do k = 0, Ns(j) - 1
                    inds(j) = k
                    if (k == bak_indx) cycle
                    if (G(bak_indx + 1, k + 1) > 0.0_real64) then
                        col_indexes(indx) = sum(inds * strides) + 1
                        values(indx) = ts(j) * G(bak_indx + 1, k + 1)
                        indx = indx + 1
                    end if
                end do
                inds(j) = bak_indx
            end do
        end do

    end subroutine qmoa_mixer

end module csr_generators

