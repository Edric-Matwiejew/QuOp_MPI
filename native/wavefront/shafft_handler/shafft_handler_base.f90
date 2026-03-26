! shafft_handler_base.f90
!
! Abstract base type for SHAFFT FFT handlers using Fortran 2003 OOP features.
! This module defines the common interface for both 1D and N-D distributed FFTs.
!
! Design Pattern: Template Method via Abstract Types
! - Base type defines common fields and interface
! - Concrete implementations (1D, ND) provide SHAFFT-specific logic
! - Factory function selects implementation based on dimensionality

module shafft_handler_base

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use, intrinsic :: iso_c_binding
    use MPI

    implicit none

    private

    ! Export base type and interfaces
    public :: fft_handler_base
    public :: fft_layout_info

    !---------------------------------------------------------------------------
    ! Layout information structure - holds partitioning details
    !---------------------------------------------------------------------------
    type :: fft_layout_info
        integer(c_size_t) :: local_i = 0 ! Local element count (initial layout)
        integer(c_size_t) :: local_i_offset = 0 ! Global offset (initial layout)
        integer(c_size_t) :: transformed_local_i = 0
        integer(c_size_t) :: transformed_local_i_offset = 0
        integer(c_size_t) :: alloc_size = 0 ! Required allocation size
    end type fft_layout_info

    !---------------------------------------------------------------------------
    ! Abstract base type for FFT handlers
    !---------------------------------------------------------------------------
    type, abstract :: fft_handler_base
        type(c_ptr) :: shafft_plan = c_null_ptr
        logical :: initialized = .false.
        logical :: planned = .false.
        integer(int32) :: n_dims = 0
        integer(int32), dimension(:), allocatable :: tensor_dims
        type(fft_layout_info) :: layout
    contains
        ! Abstract (deferred) procedures - MUST be implemented by extensions
        procedure(configure_interface), deferred :: configure
        procedure(init_interface), deferred :: init
        procedure(forward_interface), deferred :: forward
        procedure(backward_interface), deferred :: backward
        procedure(destroy_interface), deferred :: destroy

        ! Non-deferred (common) procedures
        procedure :: get_layout => fft_handler_get_layout
        procedure :: get_alloc_size => fft_handler_get_alloc_size
        procedure :: is_initialized => fft_handler_is_initialized
        procedure :: is_planned => fft_handler_is_planned
    end type fft_handler_base

    !---------------------------------------------------------------------------
    ! Abstract interfaces for deferred procedures
    !---------------------------------------------------------------------------
    abstract interface

        subroutine configure_interface(self, dims, DEVCOMM, layout_out, ierr)
            !-------------------------------------------------------------------
            ! Query SHAFFT for partitioning information (used in max_comm_size)
            !-------------------------------------------------------------------
            import :: fft_handler_base, fft_layout_info, int32, c_size_t
            class(fft_handler_base), intent(inout) :: self
            integer(int32), dimension(:), intent(in) :: dims
            integer(int32), intent(in) :: DEVCOMM
            type(fft_layout_info), intent(out) :: layout_out
            integer(int32), intent(out) :: ierr
        end subroutine configure_interface

        subroutine init_interface(self, dims, local_i, local_i_offset, DEVCOMM, ierr)
            !-------------------------------------------------------------------
            ! Create and initialize SHAFFT plan (used in plan phase)
            !-------------------------------------------------------------------
            import :: fft_handler_base, int32, c_size_t
            class(fft_handler_base), intent(inout) :: self
            integer(int32), dimension(:), intent(in) :: dims
            integer(c_size_t), intent(in) :: local_i
            integer(c_size_t), intent(in) :: local_i_offset
            integer(int32), intent(in) :: DEVCOMM
            integer(int32), intent(out) :: ierr
        end subroutine init_interface

        subroutine forward_interface(self, state, work, ierr)
            !-------------------------------------------------------------------
            ! Execute forward FFT with normalization
            !-------------------------------------------------------------------
            import :: fft_handler_base, c_ptr, int32, real64
            class(fft_handler_base), intent(inout) :: self
            complex(real64), pointer, intent(inout) :: state(:)
            complex(real64), pointer, intent(inout) :: work(:)
            integer(int32), intent(out) :: ierr
        end subroutine forward_interface

        subroutine backward_interface(self, state, work, ierr)
            !-------------------------------------------------------------------
            ! Execute backward FFT with normalization
            !-------------------------------------------------------------------
            import :: fft_handler_base, c_ptr, int32, real64
            class(fft_handler_base), intent(inout) :: self
            complex(real64), pointer, intent(inout) :: state(:)
            complex(real64), pointer, intent(inout) :: work(:)
            integer(int32), intent(out) :: ierr
        end subroutine backward_interface

        subroutine destroy_interface(self)
            !-------------------------------------------------------------------
            ! Clean up SHAFFT plan and allocated memory
            !-------------------------------------------------------------------
            import :: fft_handler_base
            class(fft_handler_base), intent(inout) :: self
        end subroutine destroy_interface

    end interface

contains

    !---------------------------------------------------------------------------
    ! Non-deferred (common) procedure implementations
    !---------------------------------------------------------------------------

    function fft_handler_get_layout(self) result(layout)
        class(fft_handler_base), intent(in) :: self
        type(fft_layout_info) :: layout
        layout = self%layout
    end function fft_handler_get_layout

    function fft_handler_get_alloc_size(self) result(alloc_size)
        class(fft_handler_base), intent(in) :: self
        integer(c_size_t) :: alloc_size
        alloc_size = self%layout%alloc_size
    end function fft_handler_get_alloc_size

    function fft_handler_is_initialized(self) result(is_init)
        class(fft_handler_base), intent(in) :: self
        logical :: is_init
        is_init = self%initialized
    end function fft_handler_is_initialized

    function fft_handler_is_planned(self) result(is_plan)
        class(fft_handler_base), intent(in) :: self
        logical :: is_plan
        is_plan = self%planned
    end function fft_handler_is_planned

end module shafft_handler_base
