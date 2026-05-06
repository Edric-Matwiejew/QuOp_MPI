module mpi_backend
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, int32, int64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer, c_int64_t
    use MPI
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: mpi_context

    type mpi_context
        real(real64) :: expectation_value

        ! Backend-internal state buffer.  On the MPI backend this lives in
        ! host memory and the host-side mirror (host_state) aliases it; on
        ! GPU backends the two are distinct (device vs. host).
        complex(real64), dimension(:), pointer :: state => null()
        ! Optional host work buffer for out-of-place propagators (e.g. sparse).
        complex(real64), dimension(:), pointer :: work => null()
        ! Pointer (not allocatable) so it can be rebound to a Python-owned
        ! buffer via cw_attach_host_observables for zero-copy transfers.
        real(real64), dimension(:), pointer :: observables => null()

        ! ----- Host-side mirrors (owned by the CPython extension) -----
        ! On MPI these alias %state / %observables (same memory); they are
        ! attached during cw_attach_host_* and refreshed in place by the
        ! sync_host_* methods (which are no-ops on this backend because
        ! the authoritative copy already lives in host memory).
        complex(real64), dimension(:), pointer :: host_state => null()
        real(real64),    dimension(:), pointer :: host_observables => null()
        real(real64),    dimension(:), pointer :: host_local_probabilities => null()

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

        ! ----- Host/device mirror contract (shared with wavefront) ------
        procedure :: attach_host_state => context_attach_host_state
        procedure :: attach_host_observables => context_attach_host_observables
        procedure :: attach_host_local_probabilities => context_attach_host_local_probabilities
        procedure :: sync_host_state => context_sync_host_state
        procedure :: sync_device_state => context_sync_device_state
        procedure :: sync_host_observables => context_sync_host_observables
        procedure :: sync_device_observables => context_sync_device_observables
        procedure :: compute_local_probabilities => context_compute_local_probabilities
        procedure :: detach_host_buffers => context_detach_host_buffers

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

        ! Establish the host-side mirrors: on MPI these alias the
        ! backend-internal buffers (single host copy).  cw_attach_host_*
        ! may later rebind state/observables to Python-owned memory; in
        ! that case the aliases are refreshed there.
        self%host_state => self%state
        self%host_observables => self%observables

    end subroutine context_setup

    subroutine context_destroy(self)
        class(mpi_context), intent(inout) :: self

        if (associated(self%state)) then
            deallocate (self%state)
            self%state => null()
        end if
        if (associated(self%observables)) then
            deallocate (self%observables)
            self%observables => null()
        end if
        if (associated(self%work)) then
            deallocate (self%work)
            self%work => null()
        end if

        ! host_state / host_observables aliased %state / %observables, which
        ! we just deallocated (or which detach_host_buffers nullified before
        ! cw_destroy_external on the Python-owned-buffer path).  Nullify the
        ! mirror references unconditionally; never deallocate them here.
        self%host_state => null()
        self%host_observables => null()
        self%host_local_probabilities => null()

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
        else if (.not. associated(self%observables)) then
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
        else if (.not. associated(self%observables)) then
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
        else if (.not. associated(self%observables)) then
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

    ! ====================================================================
    ! Host/device mirror contract (cw_attach_host_*, cw_sync_*,
    ! cw_compute_local_probabilities).  The MPI backend has no device
    ! memory: host_state / host_observables alias the backend-internal
    ! buffers, sync_host_*/sync_device_* are no-ops, and
    ! compute_local_probabilities runs the |psi|^2 loop directly on the
    ! host buffer.
    ! ====================================================================

    subroutine context_attach_host_state(self, ptr, n)
        !! Replace the backend-internal state buffer with a caller-owned
        !! (Python-allocated) buffer of length `n` and refresh the host
        !! mirror to alias it.  The previous Fortran-allocated buffer is
        !! deallocated so it does not leak.  After this call the caller
        !! owns the memory; cw_destroy_external must be used in lieu of
        !! cw_destroy so detach_host_buffers nullifies %state before
        !! ctx%destroy() runs.
        class(mpi_context), intent(inout) :: self
        type(c_ptr), value, intent(in) :: ptr
        integer(c_int64_t), value, intent(in) :: n

        if (associated(self%state)) deallocate (self%state)
        call c_f_pointer(ptr, self%state, [n])
        self%host_state => self%state
    end subroutine context_attach_host_state

    subroutine context_attach_host_observables(self, ptr, n)
        !! Mirror of attach_host_state for the observables buffer.
        class(mpi_context), intent(inout) :: self
        type(c_ptr), value, intent(in) :: ptr
        integer(c_int64_t), value, intent(in) :: n

        if (associated(self%observables)) deallocate (self%observables)
        call c_f_pointer(ptr, self%observables, [n])
        self%host_observables => self%observables
    end subroutine context_attach_host_observables

    subroutine context_attach_host_local_probabilities(self, ptr, n)
        !! Bind the caller-owned (Python-allocated) local-probabilities
        !! buffer.  Unlike state/observables, the MPI backend has no
        !! prior allocation for this field; it is purely a Python-owned
        !! scratch buffer that compute_local_probabilities fills in
        !! place each call.
        class(mpi_context), intent(inout) :: self
        type(c_ptr), value, intent(in) :: ptr
        integer(c_int64_t), value, intent(in) :: n

        self%host_local_probabilities => null()
        call c_f_pointer(ptr, self%host_local_probabilities, [n])
    end subroutine context_attach_host_local_probabilities

    subroutine context_sync_host_state(self, error_code)
        !! No-op on MPI: host_state aliases the authoritative state.
        class(mpi_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        if (.false.) self%expectation_value = 0.0_real64  ! quiet unused-arg
        error_code = 0
    end subroutine context_sync_host_state

    subroutine context_sync_device_state(self, error_code)
        !! No-op on MPI: there is no device buffer.
        class(mpi_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        if (.false.) self%expectation_value = 0.0_real64
        error_code = 0
    end subroutine context_sync_device_state

    subroutine context_sync_host_observables(self, error_code)
        !! No-op on MPI.
        class(mpi_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        if (.false.) self%expectation_value = 0.0_real64
        error_code = 0
    end subroutine context_sync_host_observables

    subroutine context_sync_device_observables(self, error_code)
        !! No-op on MPI.
        class(mpi_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        if (.false.) self%expectation_value = 0.0_real64
        error_code = 0
    end subroutine context_sync_device_observables

    subroutine context_compute_local_probabilities(self, error_code)
        !! Fill host_local_probabilities(1:local_i) with |state(i)|^2.
        !! Performs sync_host_state first (no-op on MPI, dtoh on GPU).
        class(mpi_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        integer(int64) :: i, ci_local_i

        error_code = 0
        call self%sync_host_state(error_code)
        if (error_code /= 0) return

        if (.not. associated(self%ci)) then
            error_code = 1
            return
        end if
        ci_local_i = self%ci%get_local_i()

        if (.not. associated(self%host_local_probabilities)) then
            error_code = 2
            return
        end if
        if (.not. associated(self%host_state)) then
            error_code = 2
            return
        end if
        if (size(self%host_local_probabilities, kind=int64) < ci_local_i) then
            error_code = 1
            return
        end if
        if (size(self%host_state, kind=int64) < ci_local_i) then
            error_code = 1
            return
        end if

        do i = 1, ci_local_i
            self%host_local_probabilities(i) = real(self%host_state(i) * &
                                                    conjg(self%host_state(i)), real64)
        end do
    end subroutine context_compute_local_probabilities

    subroutine context_detach_host_buffers(self)
        !! Nullify all pointers to Python-owned buffers so the subsequent
        !! ctx%destroy() does not deallocate them.  On MPI %state and
        !! %observables alias the host mirrors, so they must also be
        !! nullified (not deallocated) here -- their backing storage is
        !! NumPy memory.
        class(mpi_context), intent(inout) :: self

        self%state => null()
        self%observables => null()
        self%host_state => null()
        self%host_observables => null()
        self%host_local_probabilities => null()
    end subroutine context_detach_host_buffers

end module mpi_backend
