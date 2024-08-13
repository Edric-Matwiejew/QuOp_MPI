module mpi_backend
    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, qp => real128
    use MPI

    implicit none

    private

    public :: mpi_context

    type mpi_context
        integer(dp) :: system_size
        logical :: initialised = .false.
        integer(dp) :: alloc_local
        integer(dp) :: local_i
        integer(dp) :: local_i_offset
        real(dp) :: expectation_value

        complex(dp), dimension(:), pointer :: initial_state => null()
        complex(dp), dimension(:), pointer :: final_state => null()
        real(dp), dimension(:), allocatable :: observables

        integer(sp) :: SUBCOMM

    contains

        procedure :: setup => context_setup
        procedure :: get_expectation_value => context_get_expectation_value
        procedure :: get_state_norm => context_get_state_norm
        procedure :: destroy => context_destroy
        procedure :: set_state => context_set_state
        procedure :: get_state => context_get_state
        !procedure :: set_final_state => context_set_final_state
        !procedure :: get_final_state => context_get_final_state
        procedure :: set_observables => context_set_observables
        procedure :: get_observables => context_get_observables

    end type mpi_context

contains

    subroutine context_setup(self, system_size, alloc_local, local_i, local_i_offset, SUBCOMM)
        class(mpi_context), intent(inout) :: self
        integer(dp), intent(in) :: system_size
        integer(dp), intent(in) :: alloc_local
        integer(dp), intent(in) :: local_i
        integer(dp), intent(in) :: local_i_offset
        integer(sp), intent(in) :: SUBCOMM
        self%system_size = system_size
        self%local_i = local_i
        self%local_i_offset = local_i_offset
        self%alloc_local = alloc_local
        self%SUBCOMM = SUBCOMM
        allocate (self%initial_state(alloc_local))
        !allocate (self%final_state(alloc_local))
        allocate (self%observables(local_i))
        self%initialised = .true.
    end subroutine context_setup

    subroutine context_destroy(self)
        class(mpi_context), intent(inout) :: self
        deallocate (self%initial_state, self%observables)
        if (associated(self%final_state)) then
            deallocate(self%final_state)
        endif
        self%initialised = .false.
    end subroutine context_destroy

    real(dp) function context_get_expectation_value(self)
        class(mpi_context), intent(inout) :: self

        real(dp) :: local_expectation_value
        integer(sp) :: ierr

        local_expectation_value = dot_product(abs(self%initial_state(:self%local_i))**2, self%observables)

        call MPI_Reduce(local_expectation_value, &
                        self%expectation_value, &
                        1, &
                        MPI_DOUBLE, &
                        MPI_SUM, &
                        0, &
                        self%SUBCOMM, &
                        ierr)

        context_get_expectation_value = self%expectation_value

    end function context_get_expectation_value

    real(dp) function context_get_state_norm(self)

        class(mpi_context), intent(in) :: self
        real(dp) :: local_probs
        integer(sp) :: ierr

        local_probs = sum(abs(self%initial_state(:self%local_i))**2)

        call MPI_Reduce(local_probs, &
                        context_get_state_norm, &
                        1, &
                        MPI_DOUBLE, &
                        MPI_SUM, &
                        0, &
                        self%SUBCOMM, &
                        ierr)

    end function context_get_state_norm

    subroutine context_set_observables(self, obs)
        class(mpi_context), intent(inout) :: self
        real(dp), intent(in) :: obs(:)
        self%observables = obs
    end subroutine context_set_observables

    subroutine context_get_observables(self, obs)
        class(mpi_context), intent(inout) :: self
        real(dp), intent(inout) :: obs(:)
        obs = self%observables
    end subroutine context_get_observables

    subroutine context_set_state(self, state)
        class(mpi_context), intent(inout) :: self
        complex(dp), intent(in) :: state(:)
        self%initial_state = state
    end subroutine context_set_state

    subroutine context_get_state(self, state)
        class(mpi_context), intent(inout) :: self
        complex(dp), intent(inout) :: state(:)
        state = self%initial_state
    end subroutine context_get_state

    !subroutine context_set_final_state(self, state)
    !    class(mpi_context), intent(inout) :: self
    !    complex(dp), intent(in) :: state(:)
    !    self%final_state = state
    !end subroutine context_set_final_state

    !subroutine context_get_final_state(self, state)
    !    class(mpi_context), intent(inout) :: self
    !    complex(dp), intent(inout) :: state(:)
    !    state = self%final_state
    !end subroutine context_get_final_state

end module mpi_backend
