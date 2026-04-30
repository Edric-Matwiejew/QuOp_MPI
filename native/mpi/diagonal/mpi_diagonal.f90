module mpi_diagonal

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use iso_c_binding, only: c_f_pointer, c_ptr
    use MPI
    use mpi_backend, only: mpi_context
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: diagonal_propagator

    type diagonal_propagator

        type(mpi_context), pointer :: context => null()
        real(real64), dimension(:), pointer :: diagonal_operator => null()

    contains

        procedure :: max_comm_size => mpi_diagonal_max_comm_size
        procedure :: store_constraints => mpi_diagonal_store_constraints
        procedure :: plan => mpi_diagonal_plan
        procedure :: gen_operator => mpi_diagonal_gen_operator
        procedure :: propagate => mpi_diagonal_propagate
        procedure :: destroy => mpi_diagonal_destroy

    end type diagonal_propagator

contains

    subroutine mpi_diagonal_max_comm_size(self, ci, error_code)
        !! The diagonal propagator is compatible with any valid configuration
        !! so nothing to do here - ci remains unchanged.
        class(diagonal_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code
        error_code = 0
    end subroutine mpi_diagonal_max_comm_size

    subroutine mpi_diagonal_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! No-op: diagonal has no constraints.
        class(diagonal_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine mpi_diagonal_store_constraints

    subroutine mpi_diagonal_plan(self, context, error_code)
        class(diagonal_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        error_code = 0

        self%context => context

    end subroutine mpi_diagonal_plan

    subroutine mpi_diagonal_gen_operator(self, array_ptrs, array_sizes, error_code)
        class(diagonal_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        error_code = 0

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, self%diagonal_operator, [array_sizes(1)])

        if (size(self%diagonal_operator) == 1) then
            self%diagonal_operator => self%context%observables
        end if

    end subroutine mpi_diagonal_gen_operator

    subroutine mpi_diagonal_propagate(self, gamma, error_code)
        class(diagonal_propagator), intent(inout) :: self
        real(real64), intent(in), dimension(1) :: gamma
        integer(int32), intent(out) :: error_code
        integer(int64) :: ci_local_i

        error_code = 0
        ci_local_i = self%context%ci%get_local_i()

        self%context%state(:ci_local_i) = exp(cmplx(0.0_real64, -gamma(1), real64) * &
                                              self%diagonal_operator) * self%context%state(:ci_local_i)

    end subroutine mpi_diagonal_propagate

    subroutine mpi_diagonal_destroy(self)

        class(diagonal_propagator), intent(inout) :: self

        self%context => null()
        self%diagonal_operator => null()

    end subroutine mpi_diagonal_destroy

end module mpi_diagonal
