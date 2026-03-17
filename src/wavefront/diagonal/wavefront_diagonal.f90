module wavefront_diagonal

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use MPI
    use iso_c_binding, only: c_double, c_f_pointer, c_int, c_loc, c_null_ptr, c_ptr
    use hipfort
    use hipfort_check
    use wavefront, only: wavefront_context
    use gpu_transfer, only: gpu_allscatterv_htod
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public diagonal_propagator

    interface
        subroutine launch_phase_shift_kernel(grid, block, shmem, stream, &
                                             gamma, diagonal_operator, state, N) bind(c)
            use hipfort_types
            implicit none
            type(c_ptr), value :: diagonal_operator, state ! input/output arrays
            real(c_double), value :: gamma ! input/output arrays
            integer(c_int), value :: N, shmem
            type(dim3) :: grid, block ! grid and block size (3D grid)
            type(c_ptr), value :: stream
        end subroutine launch_phase_shift_kernel
    end interface

    type diagonal_propagator

        type(wavefront_context), pointer :: context => null()
        real(real64), dimension(:), pointer :: diagonal_operator => null()

    contains

        procedure :: max_comm_size => wavefront_diagonal_max_comm_size
        procedure :: store_constraints => wavefront_diagonal_store_constraints
        procedure :: plan => wavefront_diagonal_plan
        procedure :: gen_operator => wavefront_diagonal_gen_operator
        procedure :: propagate => wavefront_diagonal_propagate
        procedure :: destroy => wavefront_diagonal_destroy

    end type diagonal_propagator

contains

    subroutine wavefront_diagonal_max_comm_size(self, ci, error_code)
        !! The diagonal propagator is compatible with any valid configuration
        !! so nothing to do here.
        class(diagonal_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr
        error_code = 0
        call MPI_Barrier(ci%get_SUBCOMM(), ierr)

    end subroutine wavefront_diagonal_max_comm_size

    subroutine wavefront_diagonal_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! No-op: diagonal has no constraints.
        class(diagonal_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine wavefront_diagonal_store_constraints

    subroutine wavefront_diagonal_plan(self, context, error_code)
        class(diagonal_propagator), intent(inout) :: self
        type(wavefront_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        error_code = 0

        self%context => context

    end subroutine wavefront_diagonal_plan

    subroutine wavefront_diagonal_gen_operator(self, array_ptrs, array_sizes, error_code)

        class(diagonal_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        real(real64), dimension(:), pointer :: diagonal_operator

        error_code = 0

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, diagonal_operator, [array_sizes(1)])

        if (size(diagonal_operator) == 1) then
            self%diagonal_operator => self%context%observables
        else

            if (self%context%has_device) then
                call hipCheck(hipMalloc(self%diagonal_operator, self%context%ci%get_device_local_i()))
            end if

            call gpu_allscatterv_htod(self%context%NODECOMM_counts, &
                                      self%context%DEVCOMM_NODE_counts, &
                                      self%context%NODECOMM_displs, &
                                      self%context%DEVCOMM_NODE_displs, &
                                      c_loc(diagonal_operator), &
                                      c_loc(self%diagonal_operator), &
                                      MPI_DOUBLE, &
                                      self%context%ci%get_NODECOMM())

        end if

    end subroutine wavefront_diagonal_gen_operator

    subroutine wavefront_diagonal_propagate(self, gamma, error_code)

        class(diagonal_propagator), intent(inout) :: self
        real(real64), intent(in), dimension(1) :: gamma
        integer(int32), intent(out) :: error_code
        integer(c_int) :: N
        real(c_double) :: gamma_in

        integer(int32) :: num_blocks = 1200

        error_code = 0
        gamma_in = gamma(1)
        if (self%context%has_device) then

            N = int(self%context%ci%get_device_local_i(), c_int)

            call launch_phase_shift_kernel(dim3(num_blocks), &
                                           dim3(256), &
                                           0, c_null_ptr, &
                                           gamma_in, &
                                           c_loc(self%diagonal_operator), &
                                           c_loc(self%context%state), &
                                           N)

            call hipCheck(hipDeviceSynchronize())

        end if
    end subroutine wavefront_diagonal_propagate

    subroutine wavefront_diagonal_destroy(self)
        class(diagonal_propagator), intent(inout) :: self
        self%context => null()
    end subroutine wavefront_diagonal_destroy

end module wavefront_diagonal
