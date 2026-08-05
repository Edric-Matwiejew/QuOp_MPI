module context_wrapper

    use iso_fortran_env, only: real32, real64, int32, int64
    use iso_c_binding, only: c_loc, c_f_pointer, c_ptr, c_null_ptr
    use context, only: context_type
    use comm_info_module, only: quop_mpi_layout_t
    implicit none

    public

contains

    subroutine setup(context_ptr, ci_ptr, error_code)
        !f2py integer(int64), intent(out) :: context_ptr
        type(c_ptr), intent(out) :: context_ptr
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code

        type(context_type), pointer :: active_context
        type(quop_mpi_layout_t), pointer :: ci
        integer :: alloc_status

        context_ptr = c_null_ptr
        error_code = 0

        allocate (active_context, stat=alloc_status)
        if (alloc_status /= 0) then
            error_code = 100
            return
        end if

        call c_f_pointer(ci_ptr, ci)
        call active_context%setup(ci, error_code)

        if (error_code /= 0) then
            call active_context%destroy()
            deallocate (active_context)
            return
        end if

        context_ptr = c_loc(active_context)

    end subroutine setup

    subroutine destroy(context_ptr)
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%destroy()
        deallocate (active_context)
    end subroutine destroy

    subroutine get_state(context_ptr, size_state, state, error_code)
        !! Collectively gather the host state buffer over SUBCOMM.
        !! The host buffer length is alloc_local from the negotiated layout.
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(int64), intent(in) :: size_state
        complex(real64), dimension(size_state), intent(out) :: state
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%get_state(state, error_code)
    end subroutine get_state

    subroutine set_state(context_ptr, size_state, state, error_code)
        !! Collectively scatter the host state buffer over SUBCOMM.
        !! The host buffer length is alloc_local from the negotiated layout.
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(int64), intent(in) :: size_state
        complex(real64), dimension(size_state), intent(in) :: state
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%set_state(state, error_code)
    end subroutine set_state

    subroutine get_observables(context_ptr, size_obs, obs, error_code)
        !! Collectively gather the host observables buffer over SUBCOMM.
        !! The host buffer length is local_i from the negotiated layout.
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(int64), intent(in) :: size_obs
        real(real64), dimension(size_obs), intent(out) :: obs
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%get_observables(obs, error_code)
    end subroutine get_observables

    subroutine set_observables(context_ptr, size_obs, obs, error_code)
        !! Collectively scatter the host observables buffer over SUBCOMM.
        !! The host buffer length is local_i from the negotiated layout.
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(int64), intent(in) :: size_obs
        real(real64), dimension(size_obs), intent(in) :: obs
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%set_observables(obs, error_code)
    end subroutine set_observables

    subroutine get_expectation_value(context_ptr, expectation_value, error_code)
        !! Collectively compute the expectation value over SUBCOMM.
        !! The scalar result is defined on all active SUBCOMM ranks.
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        real(real64), intent(out) :: expectation_value
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        expectation_value = active_context%get_expectation_value(error_code)
    end subroutine get_expectation_value

    subroutine get_state_norm(context_ptr, state_norm, error_code)
        !! Collectively compute the state norm over SUBCOMM.
        !! The scalar result is defined on all active SUBCOMM ranks.
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        real(real64), intent(out) :: state_norm
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        state_norm = active_context%get_state_norm(error_code)
    end subroutine get_state_norm

end module context_wrapper
