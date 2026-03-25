! shafft_handler.f90
!
! Public module providing unified FFT handler interface
! Includes factory function for automatic 1D/ND selection
!
! Usage:
!   use shafft_handler
!   class(fft_handler_base), allocatable :: fft
!   call create_fft_handler(n_dims, fft)
!   call fft%configure(dims, DEVCOMM, layout, ierr)
!   call fft%init(dims, local_i, local_i_offset, DEVCOMM, ierr)
!   call fft%forward(state, work, ierr)
!   call fft%backward(state, work, ierr)
!   call fft%destroy()

module shafft_handler

    use, intrinsic :: iso_fortran_env, only: int32
    use shafft_handler_base, only: fft_handler_base, fft_layout_info
    use shafft_handler_1d, only: fft_1d_handler
    use shafft_handler_nd, only: fft_nd_handler

    implicit none

    private

    ! Export base type and layout info type
    public :: fft_handler_base
    public :: fft_layout_info

    ! Export concrete types (for type-specific operations if needed)
    public :: fft_1d_handler
    public :: fft_nd_handler

    ! Export factory function
    public :: create_fft_handler

    ! Export helper to check if handler is 1D
    public :: is_1d_handler

contains

    !---------------------------------------------------------------------------
    ! create_fft_handler: Factory function for creating appropriate handler
    !
    ! Automatically selects 1D or ND implementation based on dimensionality
    !---------------------------------------------------------------------------
    subroutine create_fft_handler(n_dims, handler)
        integer(int32), intent(in) :: n_dims
        class(fft_handler_base), allocatable, intent(out) :: handler

        if (n_dims == 1) then
            allocate (fft_1d_handler :: handler)
        else
            allocate (fft_nd_handler :: handler)
        end if

    end subroutine create_fft_handler

    !---------------------------------------------------------------------------
    ! is_1d_handler: Check if handler is 1D type
    !---------------------------------------------------------------------------
    function is_1d_handler(handler) result(is_1d)
        class(fft_handler_base), intent(in) :: handler
        logical :: is_1d

        select type (handler)
        type is (fft_1d_handler)
            is_1d = .true.
        class default
            is_1d = .false.
        end select

    end function is_1d_handler

end module shafft_handler
