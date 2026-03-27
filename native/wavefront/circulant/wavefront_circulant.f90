! wavefront_circulant.f90
!
! Distributed 1D circulant propagator for the wavefront (GPU) backend
! Uses SHAFFT for multi-GPU distributed FFTs on AMD GPUs via hipFFT
!
! This module implements a circulant unitary propagator using the FFT
! diagonalization approach:
!   U = F^{-1} diag(exp(-i * t * eigenvalues)) F
!
! where F is the Discrete Fourier Transform and eigenvalues are computed
! from the circulant graph structure.

module wavefront_circulant

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use, intrinsic :: iso_c_binding
    use MPI
    use hipfort
    use hipfort_check
    use shafft
    use wavefront, only: wavefront_context
    use communicators, only: create_NODECOMM, create_devcomm_with_topology
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use comm_info_module, only: quop_mpi_layout_t, sync_layout_from_device_partition

    implicit none

    private

    public :: circulant_propagator

    integer(int32), parameter :: num_blocks = 1200

    ! HIP kernel interfaces (defined in hip_kernels.cpp)
    interface
        subroutine launch_distributed_circulant_eigenvalues_kernel(grid, block, shmem, stream, &
                                                   nnz, indexes, values, eigenvalues, local_N, global_N, offset) bind(C)
            use hipfort_types
            use, intrinsic :: iso_c_binding
            type(dim3) :: grid, block
            integer(c_int), value :: shmem, nnz
            integer(c_size_t), value :: local_N, global_N, offset
            type(c_ptr), value :: stream, indexes, values, eigenvalues
        end subroutine launch_distributed_circulant_eigenvalues_kernel

        subroutine launch_distributed_complete_graph_eigenvalues_kernel(grid, block, shmem, stream, &
                                                                        eigenvalues, local_N, global_N, offset) bind(C)
            use hipfort_types
            use, intrinsic :: iso_c_binding
            type(dim3) :: grid, block
            integer(c_int), value :: shmem
            integer(c_size_t), value :: local_N, global_N, offset
            type(c_ptr), value :: stream, eigenvalues
        end subroutine launch_distributed_complete_graph_eigenvalues_kernel

        subroutine launch_phase_shift_kernel(grid, block, shmem, stream, &
                                             gamma, diagonal_operator, state, N) bind(C)
            use hipfort_types
            use, intrinsic :: iso_c_binding
            type(dim3) :: grid, block
            integer(c_int), value :: shmem, N
            real(c_double), value :: gamma
            type(c_ptr), value :: stream, diagonal_operator, state
        end subroutine launch_phase_shift_kernel
    end interface

    type circulant_propagator

        type(wavefront_context), pointer :: context => null()
        type(c_ptr) :: shafft_plan = c_null_ptr
        logical :: planned = .false.
        logical :: generated_operator = .false.

        ! Device memory for eigenvalues
        real(real64), dimension(:), pointer :: dev_eigenvalues => null()

    contains

        procedure :: max_comm_size => wavefront_circulant_max_comm_size
        procedure :: store_constraints => wavefront_circulant_store_constraints
        procedure :: plan => wavefront_circulant_plan
        procedure :: gen_operator => wavefront_circulant_gen_operator
        procedure :: propagate => wavefront_circulant_propagate
        procedure :: destroy => wavefront_circulant_destroy

    end type circulant_propagator

contains

    subroutine wavefront_circulant_max_comm_size(self, ci, error_code)
        !! Query SHAFFT for the device-level partitioning using communicators
        !! from ci (SUBCOMM, NODECOMM, DEVCOMM, DEVCOMM_NODE, topology).
        !! Updates both device-level and host-level fields on ci.
        class(circulant_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(c_size_t) :: local_N, local_start, local_alloc_size
        integer(c_int) :: ierr_c
        integer(int32) :: ierr

        integer(int64) :: device_local_i, device_local_i_offset

        error_code = 0

        ! Initialize device partitioning values
        local_N = 0
        local_start = 0
        local_alloc_size = 0

        if (ci%get_DEVCOMM() /= MPI_COMM_NULL) then
            ! Query SHAFFT for the 1D distribution it will use over DEVCOMM
            call shafftConfiguration1D(int(ci%get_system_size(), c_size_t), &
                                       local_N, local_start, local_alloc_size, &
                                       SHAFFT_Z2Z, ci%get_DEVCOMM(), ierr_c)

            if (ierr_c /= SHAFFT_SUCCESS) then
                write (error_unit, '(A,I0)') &
                    'ERROR: wavefront_circulant: failed to query SHAFFT 1D distribution, error code ', ierr_c
                error_code = int(ierr_c, int32)
                return
            end if
        end if

        device_local_i = int(local_N, int64)
        device_local_i_offset = int(local_start, int64)

        call sync_layout_from_device_partition(ci, device_local_i, device_local_i_offset)

        if (int(local_alloc_size, int64) > ci%get_device_alloc_local()) then
            call ci%set_device_alloc_local(int(local_alloc_size, int64), error_code)
            if (error_code /= 0) return
        end if

        call ci%set_requires_device_work_buffer(.true., error_code)
        if (error_code /= 0) return

        call MPI_Barrier(ci%get_SUBCOMM(), ierr)

    end subroutine wavefront_circulant_max_comm_size

    subroutine wavefront_circulant_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! No-op: circulant has no constraints.
        class(circulant_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine wavefront_circulant_store_constraints

    subroutine wavefront_circulant_plan(self, context, error_code)
        class(circulant_propagator), intent(inout) :: self
        type(wavefront_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        integer(c_int) :: ierr_c
        integer(c_size_t) :: ci_system_size, ci_local_n, ci_local_start

        error_code = 0

        self%context => context
        ci_system_size = int(context%ci%get_system_size(), c_size_t)
        ci_local_n = int(context%ci%get_device_local_i(), c_size_t)
        ci_local_start = int(context%ci%get_device_local_i_offset(), c_size_t)

        ! Handle trivial case: system_size == 1
        if (ci_system_size <= 1) then
            return
        end if

        if (.not. context%has_device) then
            return
        end if

        ! Create and initialize SHAFFT 1D plan
        ! Note: All ranks must participate in SHAFFT collective operations.
        ! Failures are fatal -- a partial failure would cause collective hangs.
        call shafft1DCreate(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_plan: shafft1DCreate failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        call shafft1DInit(self%shafft_plan, ci_system_size, &
                          ci_local_n, ci_local_start, &
                          SHAFFT_Z2Z, context%ci%get_DEVCOMM(), ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_plan: shafft1DInit failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Create backend FFT plans
        call shafftPlan(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_plan: shafftPlan failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        self%planned = .true.

    end subroutine wavefront_circulant_plan

    subroutine wavefront_circulant_gen_operator(self, array_ptrs, array_sizes, error_code)
        class(circulant_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr
        real(real64), dimension(:), pointer :: graph_array

        ! For sparse graph representation
        integer(int32) :: nnz
        real(real64), dimension(:), allocatable :: values
        integer(int64), dimension(:), allocatable :: indexes

        ! Device memory for sparse representation
        real(real64), dimension(:), pointer :: dev_values => null()
        integer(c_long), dimension(:), pointer :: dev_indexes => null()

        integer(int32) :: i
        integer(c_size_t) :: ci_local_n, ci_local_start, ci_system_size

        error_code = 0

        ci_local_n = int(self%context%ci%get_device_local_i(), c_size_t)
        ci_local_start = int(self%context%ci%get_device_local_i_offset(), c_size_t)
        ci_system_size = int(self%context%ci%get_system_size(), c_size_t)

        ! Ranks with no local data have nothing to generate
        if (.not. self%context%has_device) return
        if (ci_local_n <= 0) return

        ! Get graph array from Python
        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, graph_array, [array_sizes(1)])

        ! Free previous eigenvalues if re-generating operator (prevents device memory leak)
        if (associated(self%dev_eigenvalues)) then
            call hipCheck(hipFree(self%dev_eigenvalues))
            self%dev_eigenvalues => null()
        end if

        ! Allocate device memory for eigenvalues (local_N elements)
        call hipCheck(hipMalloc(self%dev_eigenvalues, int(ci_local_n, c_size_t)))

        if (array_sizes(1) == 1) then
            ! Complete graph case: eigenvalue is N-1 for k=0, and -1 for k!=0
            call launch_distributed_complete_graph_eigenvalues_kernel( &
                dim3(num_blocks), &
                dim3(256), &
                0, c_null_ptr, &
                c_loc(self%dev_eigenvalues), &
                ci_local_n, &
                ci_system_size, &
                ci_local_start)
            call hipCheck(hipDeviceSynchronize())

        else
            ! General circulant case: convert to sparse and compute eigenvalues
            ! First, find non-zero elements in the graph array
            nnz = 0
            do i = 1, array_sizes(1)
                if (abs(graph_array(i)) > epsilon(1.0_real64)) then
                    nnz = nnz + 1
                end if
            end do

            allocate (indexes(nnz), values(nnz))
            nnz = 0
            do i = 1, array_sizes(1)
                if (abs(graph_array(i)) > epsilon(1.0_real64)) then
                    nnz = nnz + 1
                    indexes(nnz) = i - 1 ! 0-indexed for the kernel
                    values(nnz) = graph_array(i)
                end if
            end do

            ! Copy sparse representation to device
            call hipCheck(hipMalloc(dev_indexes, nnz))
            call hipCheck(hipMalloc(dev_values, nnz))
            call hipCheck(hipMemcpy(dev_indexes, indexes, hipMemcpyHostToDevice))
            call hipCheck(hipMemcpy(dev_values, values, hipMemcpyHostToDevice))

            ! Launch distributed eigenvalue computation kernel
            call launch_distributed_circulant_eigenvalues_kernel( &
                dim3(num_blocks), &
                dim3(256), &
                0, c_null_ptr, &
                nnz, &
                c_loc(dev_indexes), &
                c_loc(dev_values), &
                c_loc(self%dev_eigenvalues), &
                ci_local_n, &
                ci_system_size, &
                ci_local_start)
            call hipCheck(hipDeviceSynchronize())

            ! Clean up sparse representation
            call hipCheck(hipFree(dev_indexes))
            call hipCheck(hipFree(dev_values))
            deallocate (indexes, values)
        end if

        self%generated_operator = .true.

    end subroutine wavefront_circulant_gen_operator

    subroutine wavefront_circulant_propagate(self, ts, error_code)
        !! Collective over DEVCOMM (via SHAFFT).
        !! Applies U = F^{-1} diag(exp(-i t eigenvalues)) F to the device state.
        !! On failure, writes a diagnostic to error_unit and returns a non-zero
        !! error_code instead of aborting.
        class(circulant_propagator), intent(inout) :: self
        real(real64), dimension(:), intent(in) :: ts
        integer(int32), intent(out) :: error_code

        real(c_double) :: t_val
        integer(c_int) :: ierr_c
        integer(int32) :: numblocks
        integer(c_size_t) :: ci_system_size, ci_local_n, ci_local_alloc

        error_code = 0
        ci_system_size = int(self%context%ci%get_system_size(), c_size_t)
        ci_local_n = int(self%context%ci%get_device_local_i(), c_size_t)
        ci_local_alloc = int(self%context%ci%get_device_alloc_local(), c_size_t)

        ! Handle trivial case: system_size == 1
        if (ci_system_size <= 1) then
            if (ci_local_n > 0 .and. self%context%has_device) then
                t_val = ts(1)
                call launch_phase_shift_kernel( &
                    dim3(1), dim3(1), 0, c_null_ptr, &
                    t_val, &
                    c_loc(self%dev_eigenvalues), &
                    c_loc(self%context%state), &
                    1)
                call hipCheck(hipDeviceSynchronize())
            end if
            return
        end if

        if (.not. self%context%has_device) then
            return
        end if
        if (.not. self%planned) then
            return
        end if

        ! Set buffers for SHAFFT (safe to call even when local_N=0)
        call shafftSetBuffers(self%shafft_plan, self%context%state, self%context%work, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_propagate: shafftSetBuffers failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Forward FFT (collective operation - all ranks must participate)
        call shafftExecute(self%shafft_plan, SHAFFT_FORWARD, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_propagate: forward FFT failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Normalize after forward transform (safe to call even when local_N=0)
        call shafftNormalize(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_propagate: forward normalization failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Get buffers after execute - they may have been swapped (safe when local_N=0)
        call shafftGetBuffers(self%shafft_plan, ci_local_alloc, &
                              self%context%state, self%context%work, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
              'ERROR: wavefront_circulant_propagate: shafftGetBuffers after forward FFT failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Apply phase shift only for ranks with local data
        if (ci_local_n > 0) then
            t_val = ts(1)
            numblocks = (int(ci_local_n, int32) + 255) / 256
            call launch_phase_shift_kernel( &
                dim3(numblocks), dim3(256), 0, c_null_ptr, &
                t_val, &
                c_loc(self%dev_eigenvalues), &
                c_loc(self%context%state), &
                int(ci_local_n, c_int))
            call hipCheck(hipDeviceSynchronize())
        end if

        ! Re-set buffers for SHAFFT before the backward FFT.
        ! After the forward FFT, shafftGetBuffers may have swapped the state
        ! and work pointers (e.g. for non-power-of-2 sizes).  The phase shift
        ! was applied to the updated context%state, but SHAFFT's internal
        ! buffer references are stale.  We must call shafftSetBuffers again
        ! so the backward FFT reads from the correct (phase-shifted) buffer.
        call shafftSetBuffers(self%shafft_plan, self%context%state, self%context%work, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
            'ERROR: wavefront_circulant_propagate: shafftSetBuffers before backward FFT failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Backward FFT (collective operation - all ranks must participate)
        call shafftExecute(self%shafft_plan, SHAFFT_BACKWARD, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_propagate: backward FFT failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Normalize after backward transform (safe to call even when local_N=0)
        call shafftNormalize(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
                'ERROR: wavefront_circulant_propagate: backward normalization failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

        ! Get final buffers (safe when local_N=0)
        call shafftGetBuffers(self%shafft_plan, ci_local_alloc, &
                              self%context%state, self%context%work, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            write (error_unit, '(A,I0)') &
             'ERROR: wavefront_circulant_propagate: shafftGetBuffers after backward FFT failed with error code ', ierr_c
            error_code = int(ierr_c, int32)
            return
        end if

    end subroutine wavefront_circulant_propagate

    subroutine wavefront_circulant_destroy(self)
        class(circulant_propagator), intent(inout) :: self
        integer(c_int) :: ierr_c
        logical :: has_device

        ! Check if context is associated before accessing it
        has_device = .false.
        if (associated(self%context)) then
            has_device = self%context%has_device
        end if

        if (has_device .and. self%generated_operator) then
            if (associated(self%dev_eigenvalues)) then
                call hipCheck(hipFree(self%dev_eigenvalues))
                self%dev_eigenvalues => null()
            end if
            self%generated_operator = .false.
        end if

        if (has_device .and. self%planned) then
            ! Note: work buffer is freed by context (context%work_allocated)
            if (c_associated(self%shafft_plan)) then
                call shafftDestroy(self%shafft_plan, ierr_c)
                self%shafft_plan = c_null_ptr
            end if
            self%planned = .false.
        end if

        self%context => null()

    end subroutine wavefront_circulant_destroy

end module wavefront_circulant
