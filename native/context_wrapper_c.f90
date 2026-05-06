! context_wrapper_c.f90
!
! bind(C) shim providing a stable C ABI for the CPython extension
! context_wrapper_ext.c.  Compiled per backend with preprocessor
! definitions selecting the backend module and derived type, e.g.
! `-Dcontext=mpi_backend -Dcontext_type=mpi_context` for the MPI build
! and `-Dcontext=wavefront -Dcontext_type=wavefront_context` for the
! wavefront build.
!
! Contract
! --------
! The CPython extension owns three Python-allocated NumPy arrays per
! Context: state (complex128, length alloc_local), observables (float64,
! length local_i), and (lazily) local_probabilities (float64, length
! local_i).  Each is bound into the Fortran context as a `host_*` mirror
! pointer via the cw_attach_host_* entry points.
!
! Each backend's `context_type` exposes a sync_host_* / sync_device_*
! pair that keeps the host mirror coherent with the authoritative copy:
!
!   * On the MPI backend the host mirror IS the authoritative copy, so
!     all sync_* routines are no-ops.
!
!   * On a GPU backend (e.g. wavefront) sync_host_* performs a
!     device->host gather into the attached host mirror, and
!     sync_device_* performs a host->device scatter from the attached
!     host mirror.
!
! compute_local_probabilities is implemented in Fortran (host-side
! |psi|^2 loop after sync_host_state) so the algorithm and reduction
! semantics are identical across backends.
!
! Naming convention: every entry point is prefixed cw_ to keep the
! exported symbols visibly distinct from the underlying backend
! context_type's type-bound procedures.

module context_wrapper_c

    use iso_fortran_env, only: real64, int32, int64
    use iso_c_binding
    use context, only: context_type
    use comm_info_module, only: quop_mpi_layout_t

    implicit none
    private

contains

    ! ------------------------------------------------------------------
    ! cw_setup
    !
    ! Allocate a context_type on the heap, initialise it against the
    ! supplied quop_mpi_layout_t, and return an opaque handle.
    ! ------------------------------------------------------------------
    subroutine cw_setup(ci_ptr_val, context_ptr_out, error_code) &
            bind(C, name="cw_setup")
        integer(c_int64_t), value, intent(in)  :: ci_ptr_val
        integer(c_int64_t),        intent(out) :: context_ptr_out
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                      :: ci_ptr, ctx_ptr
        type(context_type),  pointer     :: ctx
        type(quop_mpi_layout_t), pointer :: ci
        integer                          :: alloc_status

        context_ptr_out = 0_c_int64_t
        error_code      = 0_c_int32_t

        ci_ptr = transfer(ci_ptr_val, ci_ptr)

        allocate(ctx, stat=alloc_status)
        if (alloc_status /= 0) then
            error_code = 100_c_int32_t
            return
        end if

        call c_f_pointer(ci_ptr, ci)
        call ctx%setup(ci, error_code)

        if (error_code /= 0) then
            call ctx%destroy()
            deallocate(ctx)
            return
        end if

        ctx_ptr         = c_loc(ctx)
        context_ptr_out = transfer(ctx_ptr, context_ptr_out)

    end subroutine cw_setup

    ! ------------------------------------------------------------------
    ! cw_destroy
    !
    ! Destroy a context that still owns its internal buffers (i.e. no
    ! cw_attach_host_* call has rebound them to Python memory).  Used by
    ! the C extension's pre-attach failure-cleanup paths in py_setup.
    !
    ! Precondition: MUST NOT be called once the context's host mirrors
    ! have been attached to Python-owned memory; on the MPI backend
    ! host_state aliases %state, so a post-attach cw_destroy would
    ! deallocate Python memory.  After cw_attach_host_*, callers must
    ! use cw_destroy_external instead.
    ! ------------------------------------------------------------------
    subroutine cw_destroy(context_ptr_val) bind(C, name="cw_destroy")
        integer(c_int64_t), value, intent(in) :: context_ptr_val

        type(c_ptr)                  :: ctx_ptr
        type(context_type), pointer  :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%destroy()
        deallocate(ctx)

    end subroutine cw_destroy

    ! ------------------------------------------------------------------
    ! cw_destroy_external
    !
    ! Destroy a context whose host mirrors are bound to Python-owned
    ! memory.  detach_host_buffers nullifies all host_* pointers (and,
    ! on the MPI backend, the aliased %state / %observables) so that
    ! ctx%destroy() does not call deallocate() on Python memory.
    ! ------------------------------------------------------------------
    subroutine cw_destroy_external(context_ptr_val) &
            bind(C, name="cw_destroy_external")
        integer(c_int64_t), value, intent(in) :: context_ptr_val

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)

        call ctx%detach_host_buffers()
        call ctx%destroy()
        deallocate(ctx)

    end subroutine cw_destroy_external

    ! ------------------------------------------------------------------
    ! cw_attach_host_state / cw_attach_host_observables /
    ! cw_attach_host_local_probabilities
    !
    ! Bind a Python-owned buffer of length `n` as the host mirror of the
    ! corresponding context field.  Per-backend behaviour is described
    ! on each context_type%attach_host_* type-bound procedure.
    ! ------------------------------------------------------------------
    subroutine cw_attach_host_state(context_ptr_val, data, n) &
            bind(C, name="cw_attach_host_state")
        integer(c_int64_t), value, intent(in) :: context_ptr_val
        type(c_ptr),        value, intent(in) :: data
        integer(c_int64_t), value, intent(in) :: n

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%attach_host_state(data, n)
    end subroutine cw_attach_host_state

    subroutine cw_attach_host_observables(context_ptr_val, data, n) &
            bind(C, name="cw_attach_host_observables")
        integer(c_int64_t), value, intent(in) :: context_ptr_val
        type(c_ptr),        value, intent(in) :: data
        integer(c_int64_t), value, intent(in) :: n

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%attach_host_observables(data, n)
    end subroutine cw_attach_host_observables

    subroutine cw_attach_host_local_probabilities(context_ptr_val, data, n) &
            bind(C, name="cw_attach_host_local_probabilities")
        integer(c_int64_t), value, intent(in) :: context_ptr_val
        type(c_ptr),        value, intent(in) :: data
        integer(c_int64_t), value, intent(in) :: n

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%attach_host_local_probabilities(data, n)
    end subroutine cw_attach_host_local_probabilities

    ! ------------------------------------------------------------------
    ! cw_sync_host_state / cw_sync_device_state /
    ! cw_sync_host_observables / cw_sync_device_observables
    !
    ! Refresh the host or device side of the corresponding mirror.
    ! Local no-ops on the MPI backend (the host mirror aliases the
    ! authoritative buffer); collective dtoh/htod transfers on GPU
    ! backends.  Callers MUST treat each entry point as collective
    ! over the active SUBCOMM regardless of backend so the same call
    ! site is correct everywhere.
    ! ------------------------------------------------------------------
    subroutine cw_sync_host_state(context_ptr_val, error_code) &
            bind(C, name="cw_sync_host_state")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%sync_host_state(error_code)
    end subroutine cw_sync_host_state

    subroutine cw_sync_device_state(context_ptr_val, error_code) &
            bind(C, name="cw_sync_device_state")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%sync_device_state(error_code)
    end subroutine cw_sync_device_state

    subroutine cw_sync_host_observables(context_ptr_val, error_code) &
            bind(C, name="cw_sync_host_observables")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%sync_host_observables(error_code)
    end subroutine cw_sync_host_observables

    subroutine cw_sync_device_observables(context_ptr_val, error_code) &
            bind(C, name="cw_sync_device_observables")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%sync_device_observables(error_code)
    end subroutine cw_sync_device_observables

    ! ------------------------------------------------------------------
    ! cw_compute_local_probabilities
    !
    ! Fill host_local_probabilities(1:local_i) with |psi(i)|^2.  The
    ! implementation first does sync_host_state (no-op on MPI, dtoh
    ! gather on wavefront) so the same host loop runs on every backend.
    !
    ! Collective over SUBCOMM (inherits from sync_host_state).
    ! ------------------------------------------------------------------
    subroutine cw_compute_local_probabilities(context_ptr_val, error_code) &
            bind(C, name="cw_compute_local_probabilities")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call ctx%compute_local_probabilities(error_code)
    end subroutine cw_compute_local_probabilities

    ! ------------------------------------------------------------------
    ! cw_get_expectation_value / cw_get_state_norm
    !
    ! Return the collectively-reduced scalar.  Each backend reads from
    ! whichever copy is authoritative on its side (host on MPI, device
    ! on wavefront), so no explicit sync is required here.
    ! ------------------------------------------------------------------
    subroutine cw_get_expectation_value(context_ptr_val, expectation_value, error_code) &
            bind(C, name="cw_get_expectation_value")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        real(c_double),            intent(out) :: expectation_value
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        expectation_value = ctx%get_expectation_value(error_code)

    end subroutine cw_get_expectation_value

    subroutine cw_get_state_norm(context_ptr_val, state_norm, error_code) &
            bind(C, name="cw_get_state_norm")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        real(c_double),            intent(out) :: state_norm
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        state_norm = ctx%get_state_norm(error_code)

    end subroutine cw_get_state_norm

end module context_wrapper_c
