module mpi_diagonal

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use iso_c_binding
    use MPI
    use mpi_backend

    implicit none

    private

    public :: diagonal_propagator

    type diagonal_propagator

        type(mpi_context), pointer :: context
        real(dp), dimension(:), pointer :: diagonal_operator => null()

    contains

        procedure :: max_comm_size => mpi_diagonal_max_comm_size
        procedure :: plan => mpi_diagonal_plan
        procedure :: gen_operator => mpi_diagonal_gen_operator
        procedure :: propagate => mpi_diagonal_propagate
        procedure :: destroy => mpi_diagonal_destroy

    end type diagonal_propagator


    contains

    subroutine mpi_diagonal_max_comm_size(self, system_size, available_ranks, &
        constraint_ptrs, constraint_sizes, max_size, COMM)

        class(diagonal_propagator), intent(inout) :: self
        integer(dp), intent(in) :: system_size
        integer(sp), intent(in) :: available_ranks
        integer(dp), intent(inout), dimension(:) :: constraint_ptrs
        integer(dp), intent(in), dimension(:) :: constraint_sizes
        integer(sp), intent(out) :: max_size
        integer(sp), intent(in) :: COMM

        max_size = available_ranks

    end subroutine mpi_diagonal_max_comm_size

    subroutine mpi_diagonal_plan(self, context)
        class(diagonal_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context

        self%context => context
    
    end subroutine mpi_diagonal_plan

    subroutine mpi_diagonal_gen_operator(self, array_ptrs, array_sizes)
        class(diagonal_propagator), intent(inout) :: self
        integer(dp), intent(inout), dimension(:) :: array_ptrs
        integer(dp), intent(in), dimension(:) :: array_sizes 

        type(c_ptr) :: array_ptr

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, self%diagonal_operator, [array_sizes(1)])

        if (size(self%diagonal_operator) == 1) then
            self%diagonal_operator => self%context%observables
        endif
    
    end subroutine mpi_diagonal_gen_operator

    subroutine mpi_diagonal_propagate(self, gamma)
        class(diagonal_propagator), intent(inout) :: self
        real(dp), intent(in), dimension(1) :: gamma

        self%context%initial_state(:self%context%local_i) = exp(cmplx(0.0d0, -gamma(1), dp) * &
        self%diagonal_operator) * self%context%initial_state(:self%context%local_i)

        !self%context%initial_state(:self%context%local_i) = self%context%initial_state(:self%context%local_i)

    end subroutine

    subroutine mpi_diagonal_destroy(self)

        class(diagonal_propagator), intent(inout) :: self

        self%context => null()
        self%diagonal_operator => null()

    end subroutine mpi_diagonal_destroy

end module mpi_diagonal
