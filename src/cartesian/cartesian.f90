module cartesian

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64

    implicit none

    private

    public :: dist_vector, gen_local_grid, get_index

contains

    subroutine dist_vector(f, &
                           n_dim, &
                           Ns, &
                           strides, &
                           deltas, &
                           mins, &
                           local_i_offset, &
                           local_i, &
                           vec)

        use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64

        implicit none

        complex(real64), parameter :: cI = cmplx(0.0_real64, 1.0_real64, real64)
        real(real64), parameter :: pi = 4.0_real64 * atan(1.0_real64)
        external :: f
        complex(real64) :: f_temp
        integer(int32), intent(in) :: n_dim
        integer(int32), intent(in) :: Ns(n_dim) ! the number of grid points in each dimension
        integer(int64), intent(in) :: strides(n_dim)
        real(real64), intent(in) :: deltas(n_dim) ! the grid-size in each dimension
        real(real64), intent(in) :: mins(n_dim) ! the minimum in each dimension, maxs = mins + Ns*deltas
        integer(int64), intent(in) :: local_i_offset ! Starting index alogn n0 dimension.
        integer(int64), intent(in) :: local_i ! Number of indices alogn the n0 dimension at this rank.
        complex(real64), intent(inout) :: vec(local_i)

        real(real64), dimension(n_dim) :: grid_point
        integer(int32) :: i, j

        do i = local_i_offset + 1, local_i + local_i_offset
            call get_index(i, n_dim, Ns, strides, grid_point)
            grid_point = mins + (grid_point - 1.0_real64) * deltas
!f2py (callback) f
!f2py intent(out) :: f_temp
            call f(grid_point, f_temp, n_dim)
            vec(i - local_i_offset) = f_temp
        end do

    end subroutine dist_vector

    subroutine get_index(i, n_dim, Ns, strides, inds)

        use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64

        implicit none

        complex(real64), parameter :: cI = cmplx(0.0_real64, 1.0_real64, real64)
        real(real64), parameter :: pi = 4.0_real64 * atan(1.0_real64)

        integer(int32), intent(in) :: i
        integer(int32), intent(in) :: n_dim
        integer(int32), intent(in) :: Ns(n_dim)
        integer(int64), intent(in) :: strides(n_dim)
        real(real64), intent(out) :: inds(n_dim)

        integer(int32) :: j

        do j = 1, n_dim
            inds(j) = mod((i - 1) / strides(j), int(Ns(j), int64)) + 1
        end do

    end subroutine get_index

    subroutine gen_local_grid(N, &
                              n_dim, &
                              Ns, &
                              strides, &
                              deltas, &
                              mins, &
                              local_i_offset, &
                              local_i, &
                              local_grid)

        use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64

        implicit none

        complex(real64), parameter :: cI = cmplx(0.0_real64, 1.0_real64, real64)
        real(real64), parameter :: pi = 4.0_real64 * atan(1.0_real64)

        integer(int64), intent(in) :: N
        integer(int32), intent(in) :: n_dim
        integer(int32), intent(in) :: Ns(n_dim) ! the number of grid points in each dimension
        integer(int64), intent(in) :: strides(n_dim)
        real(real64), intent(in) :: deltas(n_dim) ! the grid-size in each dimension
        real(real64), intent(in) :: mins(n_dim) ! the minimum in each dimension, maxs = mins + Ns*deltas
        integer(int64), intent(in) :: local_i_offset ! Starting index alogn n0 dimension.
        integer(int64), intent(in) :: local_i ! Number of indices alogn the n0 dimension at this rank.
        real(real64), intent(out) :: local_grid(local_i, n_dim)

        real(real64) :: grid_point(n_dim)
        integer(int32) :: i

        do i = local_i_offset + 1, local_i + local_i_offset
            call get_index(i, n_dim, Ns, strides, grid_point)
            local_grid(i - local_i_offset, :) = mins + (grid_point - 1.0_real64) * deltas
        end do

    end subroutine gen_local_grid

end module cartesian
