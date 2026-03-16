module propagator_wrapper

    use iso_fortran_env, only: real32, real64, int32, int64
    use iso_c_binding, only: c_f_pointer, c_funloc, c_funptr, c_loc, c_null_ptr, c_ptr
    use context
    use propagator
    use comm_info_module, only: quop_mpi_layout_t, negotiate_callback_iface

    implicit none

    public
    private :: negotiate_trampoline

contains

    subroutine setup(propagator_ptr, error_code)
        !f2py integer(int64), intent(out) :: propagator_ptr
        type(c_ptr), intent(out) :: propagator_ptr
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(propagator_type), pointer :: active_propagator
        integer :: alloc_status
        propagator_ptr = c_null_ptr
        error_code = 0
        allocate (active_propagator, stat=alloc_status)
        if (alloc_status /= 0) then
            error_code = 100
            return
        end if
        propagator_ptr = c_loc(active_propagator)
    end subroutine setup

    subroutine max_comm_size(propagator_ptr, ci_ptr, error_code)
        !! Call the propagator's max_comm_size with a quop_mpi_layout_t.
        !f2py integer(int64), intent(in) :: propagator_ptr
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(inout) :: propagator_ptr
        type(c_ptr), intent(inout) :: ci_ptr
        integer(int32), intent(out) :: error_code

        type(propagator_type), pointer :: active_propagator
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(propagator_ptr, active_propagator)
        call c_f_pointer(ci_ptr, ci)
        call active_propagator%max_comm_size(ci, error_code)
    end subroutine max_comm_size

    subroutine store_constraints(propagator_ptr, &
                                 n_constraint_ptrs, &
                                 constraint_ptrs, &
                                 constraint_sizes)
        !! Store constraint data on the propagator for use during negotiate.
        !f2py integer(int64), intent(in) :: propagator_ptr
        !f2py integer(int32), intent(in) :: n_constraint_ptrs
        !f2py integer(int64), intent(in) :: constraint_ptrs(n_constraint_ptrs)
        !f2py integer(int64), intent(in) :: constraint_sizes(n_constraint_ptrs)
        type(c_ptr), intent(inout) :: propagator_ptr
        integer(int32), intent(in) :: n_constraint_ptrs
        integer(int64), intent(in) :: constraint_ptrs(n_constraint_ptrs)
        integer(int64), intent(in) :: constraint_sizes(n_constraint_ptrs)

        type(propagator_type), pointer :: active_propagator

        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%store_constraints(constraint_ptrs, constraint_sizes)
    end subroutine store_constraints

    subroutine get_negotiate_callback(cb_ptr)
        !! Return the address of the negotiate_trampoline as an int64.
        !! Python passes this to wrapper_negotiate alongside the propagator_ptr.
        !f2py integer(int64), intent(out) :: cb_ptr
        integer(int64), intent(out) :: cb_ptr
        type(c_funptr) :: fp

        fp = c_funloc(negotiate_trampoline)
        cb_ptr = transfer(fp, cb_ptr)
    end subroutine get_negotiate_callback

    ! ----------------------------------------------------------------
    ! bind(C) trampoline conforming to negotiate_callback_iface.
    ! Called by comm_info_module::negotiate() via c_funptr dispatch.
    ! ----------------------------------------------------------------
    subroutine negotiate_trampoline(prop_ptr, ci_ptr, error_code) bind(C)
        type(c_ptr), value, intent(in) :: prop_ptr
        type(c_ptr), value, intent(in) :: ci_ptr
        integer(int32), intent(out) :: error_code

        type(propagator_type), pointer :: p
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(prop_ptr, p)
        call c_f_pointer(ci_ptr, ci)
        call p%max_comm_size(ci, error_code)
    end subroutine negotiate_trampoline

    subroutine plan(propagator_ptr, context_ptr, error_code)
        !f2py integer(int64), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        !f2py integer(int64), intent(in) :: context_ptr
        type(c_ptr), intent(inout) :: context_ptr
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(propagator_type), pointer :: active_propagator
        type(context_type), pointer :: active_context
        call c_f_pointer(propagator_ptr, active_propagator)
        call c_f_pointer(context_ptr, active_context)
        call active_propagator%plan(active_context, error_code)
    end subroutine plan

    subroutine destroy(propagator_ptr)
        !f2py integer(int64), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        type(propagator_type), pointer :: active_propagator
        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%destroy()
        deallocate (active_propagator)
        propagator_ptr = c_null_ptr
    end subroutine destroy

    subroutine gen_operator(propagator_ptr, &
                            n_array_ptrs, &
                            array_ptrs, &
                            array_sizes, &
                            error_code)
        !f2py integer(int64), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        integer(int32), intent(in) :: n_array_ptrs
        integer(int64), intent(inout) :: array_ptrs(n_array_ptrs)
        integer(int64), intent(in) :: array_sizes(n_array_ptrs)
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(propagator_type), pointer :: active_propagator
        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%gen_operator(array_ptrs, array_sizes, error_code)
    end subroutine gen_operator

    subroutine propagate(propagator_ptr, n_params, params, error_code)
        !f2py integer(int64), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        integer(int32), intent(in) :: n_params
        real(real64), intent(inout) :: params(n_params)
        !f2py integer(int32), intent(out) :: error_code
        integer(int32), intent(out) :: error_code
        type(propagator_type), pointer :: active_propagator
        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%propagate(params, error_code)
    end subroutine propagate

end module propagator_wrapper
