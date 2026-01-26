module context_wrapper

    ! 'context' and 'context_type' are specified using the cpp preprocessor.

    use iso_fortran_env, only: sp => real32, dp => real64
    use iso_c_binding
    use context

    implicit none

    public

contains

    subroutine setup(context_ptr, &
                     system_size, &
                     alloc_local, &
                     local_i, &
                     local_i_offset, &
                     SUBCOMM)
        !f2py integer(dp), intent(out) :: context_ptr
        type(c_ptr), intent(out) :: context_ptr
        integer(dp), intent(in) :: system_size
        integer(dp), intent(in) :: alloc_local
        integer(dp), intent(in) :: local_i
        integer(dp), intent(in) :: local_i_offset
        integer(sp), intent(in) :: SUBCOMM
        type(context_type), pointer :: active_context

        allocate (active_context)
        call active_context%setup(system_size, &
                                  alloc_local, &
                                  local_i, &
                                  local_i_offset, &
                                  SUBCOMM)

        context_ptr = c_loc(active_context)

    end subroutine setup

    subroutine destroy(context_ptr)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%destroy()
        deallocate (active_context)
    end subroutine destroy

    subroutine get_state(context_ptr, size_state, state)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(dp), intent(in) :: size_state
        complex(dp), dimension(size_state), intent(out) :: state
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%get_state(state)
    end subroutine get_state

    subroutine set_state(context_ptr, size_state, state)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(dp), intent(in) :: size_state
        complex(dp), dimension(size_state), intent(in) :: state
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%set_state(state)
    end subroutine set_state

    !subroutine get_final_state(context_ptr, size_state, state)
    !    !f2py integer(dp), intent(in) :: context_ptr
    !    type(c_ptr), intent(in) :: context_ptr
    !    integer(dp), intent(in) :: size_state
    !    complex(dp), dimension(size_state), intent(out) :: state
    !    type(context_type), pointer :: active_context
    !    call c_f_pointer(context_ptr, active_context)
    !    call active_context%get_final_state(state)
    !end subroutine get_final_state

    !subroutine set_final_state(context_ptr, size_state, state)
    !    !f2py integer(dp), intent(in) :: context_ptr
    !    type(c_ptr), intent(in) :: context_ptr
    !    integer(dp), intent(in) :: size_state
    !    complex(dp), dimension(size_state), intent(in) :: state
    !    type(context_type), pointer :: active_context
    !    call c_f_pointer(context_ptr, active_context)
    !    call active_context%set_final_state(state)
    !end subroutine set_final_state

    subroutine get_observables(context_ptr, size_obs, obs)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(dp), intent(in) :: size_obs
        real(dp), dimension(size_obs), intent(out) :: obs
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%get_observables(obs)
    end subroutine get_observables

    subroutine set_observables(context_ptr, size_obs, obs)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        integer(dp), intent(in) :: size_obs
        real(dp), dimension(size_obs), intent(in) :: obs
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        call active_context%set_observables(obs)
    end subroutine set_observables

    subroutine get_expectation_value(context_ptr, expectation_value)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        real(dp), intent(out) :: expectation_value
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        expectation_value = active_context%get_expectation_value()
    end subroutine get_expectation_value

    subroutine get_state_norm(context_ptr, state_norm)
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(in) :: context_ptr
        real(dp), intent(out) :: state_norm
        type(context_type), pointer :: active_context
        call c_f_pointer(context_ptr, active_context)
        state_norm = active_context%get_state_norm()
    end subroutine get_state_norm

end module context_wrapper
