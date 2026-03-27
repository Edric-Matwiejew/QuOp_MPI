module wavefront_composite

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use, intrinsic :: iso_c_binding
    use MPI
    use iso_c_binding, only: c_f_pointer, c_int, c_loc, c_null_ptr, c_ptr, c_size_t
    use hipfort
    use hipfort_check
    use shafft
    use wavefront, only: wavefront_context
    use shafft_handler, only: create_fft_handler, fft_handler_base, fft_layout_info, fft_nd_handler
    use communicators, only: create_NODECOMM, create_devcomm_with_topology
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use comm_info_module, only: quop_mpi_layout_t, sync_layout_from_device_partition

    implicit none

    private

    public :: composite_propagator

    integer(int32) :: num_blocks = 1200
    integer(int32), parameter :: MAX_KERNEL_DIMS = 20

    interface
        subroutine launch_n_dim_circulant_eigenvalues_kernel( &
            grid, block, shmem, stream, n_dim, Ns_max, Ns, graph_array, eigenvalues, N) bind(c)
            use hipfort_types
            implicit none
            type(c_ptr), value :: Ns, graph_array, eigenvalues ! input/output arrays
            integer(c_int), value :: shmem
            integer(c_int), value :: n_dim, Ns_max, N
            type(dim3) :: grid, block ! grid and block size (3D grid)
            type(c_ptr), value :: stream
        end subroutine launch_n_dim_circulant_eigenvalues_kernel
    end interface

    interface
        subroutine launch_n_dim_complete_graph_eigenvalues_kernel( &
            grid, block, shmem, stream, n_dim, Ns_max, Ns, eigenvalues, N) bind(c)
            use hipfort_types
            implicit none
            type(c_ptr), value :: Ns, eigenvalues ! input/output arrays
            integer(c_int), value :: shmem
            integer(c_int), value :: n_dim, Ns_max, N
            type(dim3) :: grid, block ! grid and block size (3D grid)
            type(c_ptr), value :: stream
        end subroutine launch_n_dim_complete_graph_eigenvalues_kernel
    end interface

    interface
        subroutine launch_constant_phase_shift_kernel(grid, block, shmem, stream, eigenvalues, state, N) bind(c)
            use hipfort_types
            implicit none
            type(c_ptr), value :: eigenvalues, state ! input/output arrays
            integer(c_size_t), value :: N
            integer(c_int), value :: shmem
            type(dim3) :: grid, block ! grid and block size (3D grid)
            type(c_ptr), value :: stream
        end subroutine launch_constant_phase_shift_kernel
    end interface

    interface
        subroutine launch_gen_composite_mixer_kernel(grid, &
                                                     block, &
                                                     shmem, &
                                                     stream, &
                                                     n_dim, &
                                                     Ns_max_, &
                                                     Ns, &
                                                     strides_, &
                                                     ts, &
                                                     eigenvalues, &
                                                     mixer, &
                                                     N, &
                                                     offset) bind(c)
            use hipfort_types
            implicit none
            type(dim3) :: grid, block ! grid and block size (3D grid)
            integer(c_int), value :: n_dim, Ns_max_, shmem
            integer(c_size_t), value :: N, offset
            type(c_ptr), value :: Ns, strides_, ts, eigenvalues, mixer, stream
        end subroutine launch_gen_composite_mixer_kernel
    end interface

    type composite_propagator

        type(wavefront_context), pointer :: context => null()
        integer(int32) :: n_dims ! dimension of the tensor
        integer(int32), dimension(:), pointer :: tensor_dims => null() ! global tensor dimensions
        integer(c_size_t) :: transformed_local_i = 0 ! local element count in transformed layout
        integer(c_size_t) :: transformed_local_i_offset = 0 ! global offset in transformed layout
        integer(int32), allocatable, dimension(:) :: strides ! strides relative to the global tensor_dims
        integer(int32) :: Ns_max
        ! Polymorphic FFT handler - supports both 1D and ND
        class(fft_handler_base), allocatable :: shafft_handler
        logical :: planned = .false.
        logical :: generated_operator = .false.
        integer(int32), dimension(:), pointer :: dev_strides => null()
        integer(int32), dimension(:), pointer :: dev_tensor_dims => null()
        real(real64), dimension(:), pointer :: dev_eigenvalues => null()
        real(real64), dimension(:), pointer :: dev_mixer => null()
        real(real64), dimension(:), pointer :: dev_t => null()

    contains

        procedure :: max_comm_size => wavefront_composite_max_comm_size
        procedure :: store_constraints => wavefront_composite_store_constraints
        procedure :: plan => wavefront_composite_plan
        procedure :: gen_operator => wavefront_composite_gen_operator
        procedure :: propagate => wavefront_composite_propagate
        procedure :: destroy => wavefront_composite_destroy

    end type composite_propagator

contains

    subroutine wavefront_composite_max_comm_size(self, ci, error_code)
        !! Query FFT handler for the device-level partitioning using
        !! communicators from ci.  Updates both device and host fields on ci.
        class(composite_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int64) :: device_local_i, device_local_i_offset
        type(fft_layout_info) :: fft_layout
        integer(int32) :: ierr, ierr_mpi
        integer(int32) :: local_error, synced_error

        error_code = 0

        device_local_i = 0
        device_local_i_offset = 0
        local_error = 0

        if (ci%get_DEVCOMM() /= MPI_COMM_NULL) then
            ! Create FFT handler using factory function (1D or ND based on n_dims)
            call create_fft_handler(self%n_dims, self%shafft_handler)

            ! Use FFT handler's configure method to query partitioning
            call self%shafft_handler%configure(self%tensor_dims, ci%get_DEVCOMM(), fft_layout, ierr)
            if (ierr /= 0) then
                write (error_unit, '(A,I0)') &
                    'ERROR: wavefront_composite: FFT handler configure failed with error code ', ierr
                local_error = 1
            else
                device_local_i = int(fft_layout%local_i, int64)
                device_local_i_offset = int(fft_layout%local_i_offset, int64)
            end if
        end if
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci%get_SUBCOMM(), ierr_mpi)
        error_code = synced_error
        if (synced_error /= 0) return

        call sync_layout_from_device_partition(ci, device_local_i, device_local_i_offset)

        ! Update the required allocation size (alloc_size from configure if
        ! available, otherwise at least device_local_i)
        if (int(fft_layout%alloc_size, int64) > ci%get_device_alloc_local()) then
            call ci%set_device_alloc_local(int(fft_layout%alloc_size, int64), error_code)
            if (error_code /= 0) return
        end if
        if (device_local_i > ci%get_device_alloc_local()) then
            call ci%set_device_alloc_local(device_local_i, error_code)
            if (error_code /= 0) return
        end if

        call ci%set_requires_device_work_buffer(.true., error_code)
        if (error_code /= 0) return

        call MPI_Barrier(ci%get_SUBCOMM(), ierr)

    end subroutine wavefront_composite_max_comm_size

    subroutine wavefront_composite_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! Store tensor dimensions from constraints for max_comm_size.
        class(composite_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes

        type(c_ptr) :: ptr
        integer(int32), dimension(:), pointer :: arr

        if (size(constraint_ptrs) > 0 .and. constraint_sizes(1) > 0) then
            ptr = transfer(constraint_ptrs(1), ptr)
            call c_f_pointer(ptr, arr, [constraint_sizes(1)])
            self%tensor_dims => arr
            self%n_dims = int(constraint_sizes(1), int32)
        end if
    end subroutine wavefront_composite_store_constraints

    subroutine wavefront_composite_plan(self, context, error_code)
        class(composite_propagator), intent(inout) :: self
        type(wavefront_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, i_stride, ierr_mpi
        integer(int32) :: ci_devcomm, ci_subcomm
        integer(int32) :: local_error, synced_error
        integer(int64) :: ci_device_local_i, ci_device_local_i_offset
        type(fft_layout_info) :: fft_layout

        error_code = 0

        self%context => context
        self%planned = .false.
        local_error = 0
        ci_devcomm = self%context%ci%get_DEVCOMM()
        ci_subcomm = self%context%ci%get_SUBCOMM()
        if (self%context%has_device) then

            ! Ensure FFT handler is allocated (should be from max_comm_size)
            if (.not. allocated(self%shafft_handler)) then
                call create_fft_handler(self%n_dims, self%shafft_handler)
            end if

            ci_device_local_i = self%context%ci%get_device_local_i()
            ci_device_local_i_offset = self%context%ci%get_device_local_i_offset()

            ! Initialize FFT handler with actual layout
            call self%shafft_handler%init(self%tensor_dims, &
                                          int(ci_device_local_i, c_size_t), &
                                          int(ci_device_local_i_offset, c_size_t), &
                                          ci_devcomm, ierr)
            if (ierr /= 0) then
                write (error_unit, '(A,I0)') &
                    'ERROR: wavefront_composite_plan: FFT handler init failed with error code ', ierr
                local_error = 1
            else
                ! Get layout info from FFT handler
                fft_layout = self%shafft_handler%get_layout()

                if (int(fft_layout%local_i, int64) /= ci_device_local_i) then
                    write (error_unit, '(A,I0,A,I0)') &
                        'ERROR: wavefront_composite_plan: negotiate/device_local_i mismatch: ci=', &
                        ci_device_local_i, ', shafft=', int(fft_layout%local_i, int64)
                    local_error = 3
                end if
                if (int(fft_layout%local_i_offset, int64) /= ci_device_local_i_offset) then
                    write (error_unit, '(A,I0,A,I0)') &
                        'ERROR: wavefront_composite_plan: negotiate/device_local_i_offset mismatch: ci=', &
                        ci_device_local_i_offset, ', shafft=', int(fft_layout%local_i_offset, int64)
                    local_error = 3
                end if

                if (local_error == 0) then
                    ! Transformed layout info for eigenvalue operations
                    self%transformed_local_i = fft_layout%transformed_local_i
                    self%transformed_local_i_offset = fft_layout%transformed_local_i_offset

                    ! Use the transformed-layout strides reported by SHAFFT.
                    if (allocated(self%strides)) then
                        deallocate (self%strides)
                    end if
                    select type (fft_impl => self%shafft_handler)
                    type is (fft_nd_handler)
                        self%strides = fft_impl%get_strides_transformed()
                    class default
                        allocate (self%strides(size(self%tensor_dims)))
                        self%strides(size(self%tensor_dims)) = 1
                        do i_stride = size(self%tensor_dims) - 1, 1, -1
                            self%strides(i_stride) = self%strides(i_stride + 1) * self%tensor_dims(i_stride + 1)
                        end do
                    end select

                    if (self%n_dims > MAX_KERNEL_DIMS) then
                        write (error_unit, '(A,I0,A,I0)') &
                            'ERROR: wavefront_composite_plan: n_dims exceeds kernel limit: ', &
                            self%n_dims, ' > ', MAX_KERNEL_DIMS
                        local_error = 2
                    end if
                end if
            end if

            if (local_error == 0) then
                self%planned = .true.
            end if

        end if
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr_mpi)
        error_code = synced_error
        if (synced_error /= 0) self%planned = .false.

    end subroutine wavefront_composite_plan

    subroutine wavefront_composite_gen_operator(self, array_ptrs, array_sizes, error_code)
        class(composite_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        integer(int32), dimension(:), pointer :: Ns
        real(real64), dimension(:), pointer :: graph_arrays

        real(real64), dimension(:), pointer :: dev_graph_arrays => null()

        integer(int32) :: i, ierr_mpi
        integer(int32) :: local_error, synced_error

        error_code = 0

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, Ns, [array_sizes(1)])
        array_ptr = transfer(array_ptrs(2), array_ptr)
        call c_f_pointer(array_ptr, graph_arrays, [array_sizes(2)])

        local_error = 0
        if (self%context%has_device) then

            self%Ns_max = maxval(self%tensor_dims)

            if (self%n_dims > MAX_KERNEL_DIMS) then
                write (error_unit, '(A,I0,A,I0)') &
                    'ERROR: wavefront_composite_gen_operator: n_dims exceeds kernel limit: ', &
                    self%n_dims, ' > ', MAX_KERNEL_DIMS
                local_error = 2
            end if
        end if
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           self%context%ci%get_SUBCOMM(), ierr_mpi)
        error_code = synced_error
        if (synced_error /= 0) return

        if (self%context%has_device) then

            call hipCheck(hipMalloc(self%dev_tensor_dims, int(size(self%tensor_dims, kind=c_size_t), c_size_t)))
            call hipCheck(hipMalloc(self%dev_strides, int(size(self%tensor_dims, kind=c_size_t), c_size_t)))
            call hipCheck(hipMemcpy(self%dev_tensor_dims, int(self%tensor_dims, int32), hipMemcpyHostToDevice))
            call hipCheck(hipMalloc(self%dev_eigenvalues, int(sum(int(self%tensor_dims, c_size_t)), c_size_t)))

            if (.not. allocated(self%strides)) then
                allocate (self%strides(size(self%tensor_dims)))
                self%strides(size(self%tensor_dims)) = 1
                do i = size(self%tensor_dims) - 1, 1, -1
                    self%strides(i) = self%strides(i + 1) * self%tensor_dims(i + 1)
                end do
            end if

            call hipCheck(hipMemcpy(self%dev_strides, self%strides, hipMemcpyHostToDevice))

            if (array_sizes(2) == size(Ns)) then

                call launch_n_dim_complete_graph_eigenvalues_kernel(dim3(num_blocks), &
                                                                    dim3(256), &
                                                                    0, c_null_ptr, &
                                                                    size(self%tensor_dims), &
                                                                    self%Ns_max, &
                                                                    c_loc(self%dev_tensor_dims), &
                                                                    c_loc(self%dev_eigenvalues), &
                                                                    sum(self%tensor_dims))
                call hipCheck(hipDeviceSynchronize())

            else

                call hipCheck(hipMalloc(dev_graph_arrays, size(graph_arrays, 1)))
                call hipCheck(hipMemcpy(dev_graph_arrays, graph_arrays, hipMemcpyHostToDevice))

                call launch_n_dim_circulant_eigenvalues_kernel(dim3(num_blocks), &
                                                               dim3(256), &
                                                               0, c_null_ptr, &
                                                               size(self%tensor_dims), &
                                                               self%Ns_max, &
                                                               c_loc(self%dev_tensor_dims), &
                                                               c_loc(dev_graph_arrays), &
                                                               c_loc(self%dev_eigenvalues), &
                                                               sum(self%tensor_dims))

                call hipCheck(hipDeviceSynchronize())
                call hipCheck(hipFree(dev_graph_arrays))

            end if

            if (self%transformed_local_i > 0_c_size_t) then
                call hipCheck(hipMalloc(self%dev_mixer, int(self%transformed_local_i, c_size_t)))
            end if
            call hipCheck(hipMalloc(self%dev_t, int(size(self%tensor_dims, kind=c_size_t), c_size_t)))

            self%generated_operator = .true.

        end if

    end subroutine wavefront_composite_gen_operator

    subroutine wavefront_composite_propagate(self, t, error_code)
        !! Collective over DEVCOMM (via SHAFFT handler).
        !! On failure, writes a diagnostic to error_unit and returns a non-zero
        !! error_code instead of aborting.
        class(composite_propagator), intent(inout) :: self
        real(real64), dimension(:), intent(in) :: t
        integer(int32), intent(out) :: error_code

        real(real64), dimension(:), pointer :: t_in

        integer(int32) :: ierr

        integer(int32) :: numblocks
        integer(int64) :: ci_device_local_i

        error_code = 0

        if (self%context%has_device) then
            ci_device_local_i = self%context%ci%get_device_local_i()

            if (self%n_dims > MAX_KERNEL_DIMS) then
                write (error_unit, '(A,I0,A,I0)') &
                    'ERROR: wavefront_composite_propagate: n_dims exceeds kernel limit: ', &
                    self%n_dims, ' > ', MAX_KERNEL_DIMS
                error_code = 1
                return
            end if
            if (ci_device_local_i <= 0) then
                return
            end if

            allocate (t_in(size(self%tensor_dims)))
            if (size(t) == 1) then
                t_in = t(1)
            else
                t_in = t
            end if
            call hipCheck(hipMemcpy(self%dev_t, t_in, hipMemcpyHostToDevice))

            ! Forward FFT using polymorphic handler
            call self%shafft_handler%forward(self%context%state, self%context%work, ierr)
            if (ierr /= 0) then
                write (error_unit, '(A,I0)') &
                    'ERROR: wavefront_composite_propagate: forward FFT failed with error code ', ierr
                error_code = ierr
                deallocate (t_in)
                return
            end if

            if (self%transformed_local_i > 0_c_size_t) then
                ! After forward FFT, data is in the redistributed Fourier-space layout.
                numblocks = int((self%transformed_local_i + 255_c_size_t) / 256_c_size_t, int32)
                call launch_gen_composite_mixer_kernel(dim3(numblocks), &
                                                       dim3(256), &
                                                       0, c_null_ptr, &
                                                       size(self%tensor_dims), &
                                                       self%Ns_max, &
                                                       c_loc(self%dev_tensor_dims), &
                                                       c_loc(self%dev_strides), &
                                                       c_loc(self%dev_t), &
                                                       c_loc(self%dev_eigenvalues), &
                                                       c_loc(self%dev_mixer), &
                                                       self%transformed_local_i, &
                                                       self%transformed_local_i_offset)
                call hipCheck(hipDeviceSynchronize())

                call launch_constant_phase_shift_kernel(dim3(numblocks), &
                                                        dim3(256), &
                                                        0, c_null_ptr, &
                                                        c_loc(self%dev_mixer), &
                                                        c_loc(self%context%state), &
                                                        self%transformed_local_i)
                call hipCheck(hipDeviceSynchronize())
            end if

            ! Backward FFT using polymorphic handler
            call self%shafft_handler%backward(self%context%state, self%context%work, ierr)
            if (ierr /= 0) then
                write (error_unit, '(A,I0)') &
                    'ERROR: wavefront_composite_propagate: backward FFT failed with error code ', ierr
                error_code = ierr
                deallocate (t_in)
                return
            end if

            deallocate (t_in)

        end if

    end subroutine wavefront_composite_propagate

    subroutine wavefront_composite_destroy(self)
        class(composite_propagator), intent(inout) :: self
        logical :: has_device

        has_device = .false.
        if (associated(self%context)) then
            has_device = self%context%has_device
        end if

        if (has_device .and. self%generated_operator) then

            if (associated(self%dev_tensor_dims)) call hipCheck(hipFree(self%dev_tensor_dims))
            if (associated(self%dev_strides)) call hipCheck(hipFree(self%dev_strides))
            if (associated(self%dev_eigenvalues)) call hipCheck(hipFree(self%dev_eigenvalues))
            if (associated(self%dev_mixer)) call hipCheck(hipFree(self%dev_mixer))
            if (associated(self%dev_t)) call hipCheck(hipFree(self%dev_t))

            self%dev_tensor_dims => null()
            self%dev_strides => null()
            self%dev_eigenvalues => null()
            self%dev_mixer => null()
            self%dev_t => null()

            self%generated_operator = .false.

        end if

        if (allocated(self%strides)) then
            deallocate (self%strides)
        end if

        if (has_device .and. self%planned) then
            ! Destroy FFT handler (polymorphic call)
            if (allocated(self%shafft_handler)) then
                call self%shafft_handler%destroy()
                deallocate (self%shafft_handler)
            end if
            self%planned = .false.
        end if
        self%tensor_dims => null()
        self%context => null()

    end subroutine wavefront_composite_destroy

end module wavefront_composite
