module propagator_wrapper

    use iso_fortran_env, only: sp => real32, dp => real64
    use iso_c_binding
    use context
    use propagator

    implicit none

    public

contains

    subroutine setup(propagator_ptr)
        !f2py integer(dp), intent(out) :: propagator_ptr
        type(c_ptr), intent(out) :: propagator_ptr
        type(propagator_type), pointer :: active_propagator
        allocate(active_propagator)
        propagator_ptr = c_loc(active_propagator)
    end subroutine setup

    subroutine max_comm_size(propagator_ptr, system_size, available_ranks, &
        n_constraint_ptrs, constraint_ptrs, constraint_sizes, max_size, COMM)
        !f2py integer(dp), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        integer(dp), intent(in) :: system_size
        integer(sp), intent(in) :: available_ranks
        integer(sp), intent(in) :: n_constraint_ptrs
        integer(dp), intent(inout) :: constraint_ptrs(n_constraint_ptrs)
        integer(dp), intent(in) :: constraint_sizes(n_constraint_ptrs)
        integer(sp), intent(out) :: max_size
        integer(sp), intent(in) :: COMM
        type(propagator_type), pointer :: active_propagator
        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%max_comm_size(system_size, available_ranks, &
        constraint_ptrs, constraint_sizes, max_size, COMM)
    end subroutine max_comm_size


    !subroutine comm_info(propagator_ptr, n_constraint_ptrs, constraint_ptrs, constraint_sizes, &
    !    n_comm_info, comm_info, COMM)
    !    !f2py integer(dp), intent(in) :: propagator_ptr
    !    type(c_ptr), intent(inout) :: propagator_ptr
    !    integer(sp), intent(in) :: n_constraint_ptrs
    !    integer(dp), intent(inout) :: constraint_ptrs(n_constraint_ptrs)
    !    integer(dp), intent(in) :: constraint_sizes(n_constraint_ptrs)
    !    integer(dp), intent(in) :: n_comm_info
    !    integer(dp), intent(out) :: comm_info(n_comm_info)
    !    integer(sp), intent(in) :: COMM
    !    type(propagator_type), pointer :: active_propagator
    !    call c_f_pointer(propagator_ptr, active_propagator)
    !    call active_propagator%comm_info(constraint_ptrs, constraint_sizes, comm_info, COMM)
    !end subroutine max_comm_size

    subroutine plan(propagator_ptr, context_ptr)
        !f2py integer(dp), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        !f2py integer(dp), intent(in) :: context_ptr
        type(c_ptr), intent(inout) :: context_ptr
        type(propagator_type), pointer :: active_propagator
        type(context_type), pointer :: active_context
        call c_f_pointer(propagator_ptr, active_propagator)
        call c_f_pointer(context_ptr, active_context)
        call active_propagator%plan(active_context)
    end subroutine plan

    subroutine destroy(propagator_ptr)
        !f2py integer(dp), intent(in) :: propagator_ptr
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
                             array_sizes)
        !f2py integer(dp), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        integer(sp), intent(in) :: n_array_ptrs
        integer(dp), intent(inout) :: array_ptrs(n_array_ptrs)
        integer(dp), intent(in) :: array_sizes(n_array_ptrs)
        type(propagator_type), pointer :: active_propagator
        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%gen_operator(array_ptrs, array_sizes)
    end subroutine gen_operator

    subroutine propagate(propagator_ptr, n_params, params)
        !f2py integer(dp), intent(in) :: propagator_ptr
        type(c_ptr), intent(inout) :: propagator_ptr
        integer(sp), intent(in) :: n_params
        real(dp), intent(inout) :: params(n_params)
        type(propagator_type), pointer :: active_propagator
        call c_f_pointer(propagator_ptr, active_propagator)
        call active_propagator%propagate(params)
    end subroutine propagate

end module propagator_wrapper
