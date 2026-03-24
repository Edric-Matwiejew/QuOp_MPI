module fftw_mpi_init_guard
    use, intrinsic :: iso_c_binding
    implicit none
    include 'fftw3-mpi.f03'
    private
    public :: ensure_fftw_mpi_init
    logical, save :: fftw_mpi_initialized = .false.
contains
    subroutine ensure_fftw_mpi_init()
        if (.not. fftw_mpi_initialized) then
            call fftw_mpi_init()
            fftw_mpi_initialized = .true.
        end if
    end subroutine ensure_fftw_mpi_init
end module fftw_mpi_init_guard
