module mpi_backend
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, int32, int64
    use MPI
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: mpi_context

    type mpi_context
        real(real64) :: expectation_value

        complex(real64), dimension(:), pointer :: state => null()
        ! Optional host work buffer for out-of-place propagators (e.g. sparse).
        complex(real64), dimension(:), pointer :: work => null()
        real(real64), dimension(:), allocatable :: observables

        ! Pointer to the shared quop_mpi_layout_t (owned by caller, not freed here)
        type(quop_mpi_layout_t), pointer :: ci => null()

    contains

        procedure :: setup => context_setup
        procedure :: get_expectation_value => context_get_expectation_value
        procedure :: get_state_norm => context_get_state_norm
        procedure :: destroy => context_destroy
        procedure :: set_state => context_set_state
        procedure :: get_state => context_get_state
        procedure :: set_observables => context_set_observables
        procedure :: get_observables => context_get_observables

    end type mpi_context

contains

    subroutine context_setup(self, ci, error_code)
        class(mpi_context), intent(inout) :: self
        type(quop_mpi_layout_t), target, intent(in) :: ci
        integer(int32), intent(out) :: error_code

        integer :: alloc_status
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_alloc_local, ci_local_i

        error_code = 0

        self%ci => ci
        ci_subcomm = ci%get_SUBCOMM()
        ci_alloc_local = ci%get_alloc_local()
        ci_local_i = ci%get_local_i()

        local_error = 0

        allocate (self%state(ci_alloc_local), stat=alloc_status)
        if (alloc_status /= 0) then
            local_error = 1
        end if

        if (local_error == 0) then
            self%state = cmplx(0.0_real64, 0.0_real64, real64)
        end if

        if (local_error == 0) then
            allocate (self%observables(ci_local_i), stat=alloc_status)
            if (alloc_status /= 0) then
                local_error = 2
            end if
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) then
            call self%destroy()
            return
        end if

    end subroutine context_setup

    subroutine context_destroy(self)
        class(mpi_context), intent(inout) :: self

        if (associated(self%state)) then
            deallocate (self%state)
            self%state => null()
        end if
        if (allocated(self%observables)) then
            deallocate (self%observables)
        end if
        if (associated(self%work)) then
            deallocate (self%work)
            self%work => null()
        end if

        self%ci => null()
        self%expectation_value = 0.0_real64
    end subroutine context_destroy

    real(real64) function context_get_expectation_value(self, error_code)
        !! Collective over SUBCOMM.
        !! The scalar result is defined on all active SUBCOMM ranks.
        class(mpi_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        real(real64) :: local_expectation_value
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i

        context_get_expectation_value = 0.0_real64
        self%expectation_value = 0.0_real64
        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (.not. associated(self%state)) then
            local_error = 1
        else if (.not. allocated(self%observables)) then
            local_error = 1
        else if (size(self%state) < ci_local_i) then
            local_error = 1
        else if (size(self%observables) < ci_local_i) then
            local_error = 1
        end if

        if (ci_subcomm == MPI_COMM_NULL) then
            error_code = local_error
            return
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        local_expectation_value = dot_product(abs(self%state(:ci_local_i))**2, self%observables(:ci_local_i))

        call MPI_Allreduce(local_expectation_value, &
                           self%expectation_value, &
                           1, &
                           MPI_DOUBLE, &
                           MPI_SUM, &
                           ci_subcomm, &
                           ierr)

        context_get_expectation_value = self%expectation_value

    end function context_get_expectation_value

    real(real64) function context_get_state_norm(self, error_code)
        !! Collective over SUBCOMM.
        !! The scalar result is defined on all active SUBCOMM ranks.

        class(mpi_context), intent(in) :: self
        integer(int32), intent(out) :: error_code
        real(real64) :: local_probs
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i

        context_get_state_norm = 0.0_real64
        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (.not. associated(self%state)) then
            local_error = 1
        else if (size(self%state) < ci_local_i) then
            local_error = 1
        end if

        if (ci_subcomm == MPI_COMM_NULL) then
            error_code = local_error
            return
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        local_probs = sum(abs(self%state(:ci_local_i))**2)

        call MPI_Allreduce(local_probs, &
                           context_get_state_norm, &
                           1, &
                           MPI_DOUBLE, &
                           MPI_SUM, &
                           ci_subcomm, &
                           ierr)

    end function context_get_state_norm

    subroutine context_set_observables(self, obs, error_code)
        !! Collective over SUBCOMM.
        !! MPI satisfies the shared wrapper contract trivially because the
        !! observables already reside in host memory on every active rank.
        class(mpi_context), intent(inout) :: self
        real(real64), intent(in) :: obs(:)
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (.not. allocated(self%observables)) then
            local_error = 2
        else if (size(obs) < ci_local_i) then
            local_error = 1
        else if (size(obs) > size(self%observables)) then
            local_error = 3
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        self%observables(:size(obs)) = obs
    end subroutine context_set_observables

    subroutine context_get_observables(self, obs, error_code)
        !! Collective over SUBCOMM.
        !! MPI satisfies the shared wrapper contract trivially because the
        !! observables already reside in host memory on every active rank.
        class(mpi_context), intent(inout) :: self
        real(real64), intent(inout) :: obs(:)
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (.not. allocated(self%observables)) then
            local_error = 2
        else if (size(obs) < ci_local_i) then
            local_error = 1
        else if (size(obs) > size(self%observables)) then
            local_error = 3
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        obs = self%observables(:size(obs))
    end subroutine context_get_observables

    subroutine context_set_state(self, state, error_code)
        !! Collective over SUBCOMM.
        !! MPI satisfies the shared wrapper contract trivially because the
        !! state already resides in host memory on every active rank.
        class(mpi_context), intent(inout) :: self
        complex(real64), intent(in) :: state(:)
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (.not. associated(self%state)) then
            local_error = 2
        else if (size(state) < ci_local_i) then
            local_error = 1
        else if (size(state) > size(self%state)) then
            local_error = 3
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        self%state(:size(state)) = state
        if (size(state) < size(self%state)) then
            self%state(size(state) + 1:) = cmplx(0.0_real64, 0.0_real64, real64)
        end if
    end subroutine context_set_state

    subroutine context_get_state(self, state, error_code)
        !! Collective over SUBCOMM.
        !! MPI satisfies the shared wrapper contract trivially because the
        !! state already resides in host memory on every active rank.
        class(mpi_context), intent(inout) :: self
        complex(real64), intent(inout) :: state(:)
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (.not. associated(self%state)) then
            local_error = 2
        else if (size(state) < ci_local_i) then
            local_error = 1
        else if (size(state) > size(self%state)) then
            local_error = 3
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        state = self%state(:size(state))
    end subroutine context_get_state

end module mpi_backend
