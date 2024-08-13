module cartesian

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

    implicit none

    private

    public :: dist_vector, gen_local_grid, get_index

contains
  
    subroutine dist_vector( f, &
                            n_dim, &
                            Ns, &
                            strides, &
                            deltas, &
                            mins, &
                            local_i_offset, &
                            local_i, &
                            vec)
    
    
        use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

        implicit none

        complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
        real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
        external :: f
        complex(dp) :: f_temp
        integer(sp), intent(in) :: n_dim
        integer(sp), intent(in) :: Ns(n_dim) ! the number of grid points in each dimension
        integer(dp), intent(in) :: strides(n_dim)
        real(dp), intent(in) :: deltas(n_dim) ! the grid-size in each dimension
        real(dp), intent(in) :: mins(n_dim) ! the minimum in each dimension, maxs = mins + Ns*deltas
        integer(dp), intent(in) :: local_i_offset ! Starting index alogn n0 dimension.
        integer(dp), intent(in) :: local_i ! Number of indices alogn the n0 dimension at this rank.
        real(dp), intent(inout) :: vec(local_i)

        real(dp), dimension(n_dim) :: grid_point
        integer :: i, j
  
        do i = local_i_offset + 1, local_i + local_i_offset
            call get_index(i, n_dim, Ns, strides, grid_point)
            grid_point = mins + (grid_point - 1.0_dp)*deltas
!f2py (callback) f
!f2py intent(out) :: f_temp
            call f(grid_point, f_temp, n_dim)
            vec(i - local_i_offset) = f_temp
        enddo
    
    end subroutine dist_vector

    subroutine get_index(i, n_dim, Ns, strides, inds)
    
        use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

        implicit none

        complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
        real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
 

        integer(sp), intent(in) :: i
        integer(sp), intent(in) :: n_dim
        integer(sp), intent(in) :: Ns(n_dim)
        integer(dp), intent(in) :: strides(n_dim)
        real(dp), intent(out) :: inds(n_dim)

        integer(sp) :: j

        do j = 1, n_dim
            inds(j) = mod((i - 1)/strides(j), Ns(j)) + 1
        enddo

    end subroutine get_index
    
    subroutine gen_local_grid(  N, &
                                n_dim, &
                                Ns, &
                                strides, &
                                deltas, &
                                mins, &
                                local_i_offset, &
                                local_i, &
                                local_grid)
    
        use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

        implicit none

        complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
        real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
 
    
        integer(dp), intent(in) :: N
        integer(sp), intent(in) :: n_dim
        integer(sp), intent(in) :: Ns(n_dim) ! the number of grid points in each dimension
        integer(dp), intent(in) :: strides(n_dim)
        real(dp), intent(in) :: deltas(n_dim) ! the grid-size in each dimension
        real(dp), intent(in) :: mins(n_dim) ! the minimum in each dimension, maxs = mins + Ns*deltas
        integer(dp), intent(in) :: local_i_offset ! Starting index alogn n0 dimension.
        integer(dp), intent(in) :: local_i ! Number of indices alogn the n0 dimension at this rank.
        real(dp), intent(out) :: local_grid(local_i, n_dim)

        real(dp) :: grid_point(n_dim)
        integer :: i
    
        do i = local_i_offset + 1, local_i + local_i_offset
            call get_index(i, n_dim, Ns, strides, grid_point)
            local_grid(i - local_i_offset, :) = mins + (grid_point - 1.0_dp)*deltas
        enddo
    
    end subroutine gen_local_grid


end module cartesian
