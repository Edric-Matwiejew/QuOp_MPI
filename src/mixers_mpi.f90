subroutine hypercube(   N, &
                        lb, &
                        ub, &
                        row_starts, &
                        col_indexes, &
                        values)

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

    implicit none

    integer(sp), intent(in) :: N
    integer(sp), intent(in) :: lb
    integer(sp), intent(in) :: ub
    integer(sp), target, dimension(ub - lb + 2), intent(out) :: row_starts
    integer(sp), target, dimension((ub - lb + 1)*N), intent(out) :: col_indexes
    complex(dp), target, dimension((ub - lb + 1)*N), intent(out) :: values

    integer(sp) :: local_rows
    integer(sp) :: columns

    integer(sp), dimension(N) :: powers, binary

    integer(sp) :: dec

    integer(sp) :: i, j, k, indx, temp

    powers = [(2**i, i = 0, N - 1)]

    local_rows = ub - lb + 1
    columns = 2**N

    indx = 1
    do j = lb - 1, ub - 1
        binary = mod(j/powers,2)
        do i = 1, N
            if (binary(i) == 0) then
                binary(i) = 1
            elseif (binary(i) == 1) then
                binary(i) = 0
            endif
            dec = sum(binary*powers)
            if (binary(i) == 0) then
                binary(i) = 1
            elseif (binary(i) == 1) then
                binary(i) = 0
            endif
            col_indexes(indx) = dec + 1
            indx = indx + 1
        enddo

        do i = indx - N, indx - 1
            temp = col_indexes(i)
            do k = i -1, indx - N, -1
                if (col_indexes(k) <= temp) exit
                col_indexes(k + 1) = col_indexes(k)
            enddo
            col_indexes(k + 1) = temp
        enddo

    enddo

    row_starts(1) = (lb - 1)*N + 1
    row_starts(2:local_rows + 1) = N

    do i = 1, local_rows
        row_starts(i + 1) = row_starts(i + 1) + row_starts(i)
    enddo

    values = 1

end subroutine hypercube
