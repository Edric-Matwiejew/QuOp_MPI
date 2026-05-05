! context_wrapper_c.f90
!
! bind(C) shim providing a stable C ABI for the CPython extension
! context_wrapper_ext.c.  Compiled with the same preprocessor definitions
! as context_wrapper.f90 so that `context` and `context_type` resolve to the
! correct backend (e.g. -Dcontext=mpi_backend -Dcontext_type=mpi_context for
! the MPI build).
!
! None of the f2py directives in context_wrapper.f90 are needed here; the C
! extension calls these symbols directly via the stable names below.
!
! Naming convention: every entry point is prefixed cw_ to avoid collisions
! with the existing context_wrapper Fortran module symbols.

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
    !
    ! Arguments
    !   ci_ptr_val      [in]  int64 opaque handle for quop_mpi_layout_t
    !   context_ptr_out [out] int64 opaque handle for the new context
    !   error_code      [out] 0 on success, 100 on allocation failure,
    !                         non-zero backend status otherwise
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
    ! Destroy and deallocate the context referred to by context_ptr_val.
    ! Assumes the context still owns its state buffer (no attach was made,
    ! or cw_detach_state was called first).
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
    ! cw_attach_state
    !
    ! Replace the context's internally-allocated state buffer with a
    ! caller-owned (Python-owned) buffer of length `size_state`.
    !
    ! The currently-associated Fortran-allocated buffer is deallocated
    ! first; the new pointer is then bound via c_f_pointer.  Ownership of
    ! the new memory remains with the caller — cw_destroy must NOT be
    ! used after this call; use cw_destroy_external instead, which
    ! nullifies the pointer before invoking ctx%destroy().
    ! ------------------------------------------------------------------
    subroutine cw_attach_state(context_ptr_val, state_data, size_state) &
            bind(C, name="cw_attach_state")
        integer(c_int64_t), value, intent(in) :: context_ptr_val
        type(c_ptr),        value, intent(in) :: state_data
        integer(c_int64_t), value, intent(in) :: size_state

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)

        if (associated(ctx%state)) then
            deallocate(ctx%state)
        end if
        call c_f_pointer(state_data, ctx%state, [size_state])

    end subroutine cw_attach_state

    ! ------------------------------------------------------------------
    ! cw_destroy_external
    !
    ! Destroy a context whose state buffer is externally owned.  The
    ! state pointer is nullified first so that ctx%destroy() does not
    ! deallocate Python-managed memory.
    ! ------------------------------------------------------------------
    subroutine cw_destroy_external(context_ptr_val) &
            bind(C, name="cw_destroy_external")
        integer(c_int64_t), value, intent(in) :: context_ptr_val

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)

        ctx%state => null()
        ctx%observables => null()
        call ctx%destroy()
        deallocate(ctx)

    end subroutine cw_destroy_external

    ! ------------------------------------------------------------------
    ! cw_attach_observables
    !
    ! Replace the context's internally-allocated observables buffer with
    ! a caller-owned (Python-owned) real(real64) buffer of length
    ! `size_obs`.  Mirrors cw_attach_state: the existing pointer (if
    ! associated) is deallocated first, then rebound to the supplied
    ! address via c_f_pointer.  Ownership of the new memory remains with
    ! the caller; cw_destroy_external nullifies the pointer before
    ! invoking ctx%destroy() so Fortran does not free Python memory.
    ! ------------------------------------------------------------------
    subroutine cw_attach_observables(context_ptr_val, obs_data, size_obs) &
            bind(C, name="cw_attach_observables")
        integer(c_int64_t), value, intent(in) :: context_ptr_val
        type(c_ptr),        value, intent(in) :: obs_data
        integer(c_int64_t), value, intent(in) :: size_obs

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)

        if (associated(ctx%observables)) then
            deallocate(ctx%observables)
        end if
        call c_f_pointer(obs_data, ctx%observables, [size_obs])

    end subroutine cw_attach_observables

    ! ------------------------------------------------------------------
    ! cw_get_state
    !
    ! Collectively gather the host state buffer into the caller-supplied
    ! complex(real64) buffer of length size_state.  The buffer is owned
    ! by Python; no allocation occurs inside Fortran.
    ! ------------------------------------------------------------------
    subroutine cw_get_state(context_ptr_val, size_state, state_data, error_code) &
            bind(C, name="cw_get_state")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int64_t), value, intent(in)  :: size_state
        type(c_ptr),        value, intent(in)  :: state_data
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx
        complex(real64),    pointer :: state(:)

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call c_f_pointer(state_data, state, [size_state])
        call ctx%get_state(state, error_code)

    end subroutine cw_get_state

    ! ------------------------------------------------------------------
    ! cw_set_state
    !
    ! Collectively scatter the caller-supplied complex(real64) buffer of
    ! length size_state into the context state.
    ! ------------------------------------------------------------------
    subroutine cw_set_state(context_ptr_val, size_state, state_data, error_code) &
            bind(C, name="cw_set_state")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int64_t), value, intent(in)  :: size_state
        type(c_ptr),        value, intent(in)  :: state_data
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx
        complex(real64),    pointer :: state(:)

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call c_f_pointer(state_data, state, [size_state])
        call ctx%set_state(state, error_code)

    end subroutine cw_set_state

    ! ------------------------------------------------------------------
    ! cw_get_observables
    !
    ! Collectively gather the host observables buffer into the
    ! caller-supplied real(real64) buffer of length size_obs.
    ! ------------------------------------------------------------------
    subroutine cw_get_observables(context_ptr_val, size_obs, obs_data, error_code) &
            bind(C, name="cw_get_observables")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int64_t), value, intent(in)  :: size_obs
        type(c_ptr),        value, intent(in)  :: obs_data
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx
        real(real64),       pointer :: obs(:)

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call c_f_pointer(obs_data, obs, [size_obs])
        call ctx%get_observables(obs, error_code)

    end subroutine cw_get_observables

    ! ------------------------------------------------------------------
    ! cw_set_observables
    !
    ! Collectively scatter the caller-supplied real(real64) buffer of
    ! length size_obs into the context observables.
    ! ------------------------------------------------------------------
    subroutine cw_set_observables(context_ptr_val, size_obs, obs_data, error_code) &
            bind(C, name="cw_set_observables")
        integer(c_int64_t), value, intent(in)  :: context_ptr_val
        integer(c_int64_t), value, intent(in)  :: size_obs
        type(c_ptr),        value, intent(in)  :: obs_data
        integer(c_int32_t),        intent(out) :: error_code

        type(c_ptr)                 :: ctx_ptr
        type(context_type), pointer :: ctx
        real(real64),       pointer :: obs(:)

        ctx_ptr = transfer(context_ptr_val, ctx_ptr)
        call c_f_pointer(ctx_ptr, ctx)
        call c_f_pointer(obs_data, obs, [size_obs])
        call ctx%set_observables(obs, error_code)

    end subroutine cw_set_observables

    ! ------------------------------------------------------------------
    ! cw_get_expectation_value
    !
    ! Collectively compute the expectation value and return it as a
    ! double scalar.
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

    ! ------------------------------------------------------------------
    ! cw_get_state_norm
    !
    ! Collectively compute the state norm and return it as a double
    ! scalar.
    ! ------------------------------------------------------------------
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
