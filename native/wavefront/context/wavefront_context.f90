module wavefront

    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, error_unit
    use, intrinsic :: iso_c_binding, only: c_size_t, c_ptr, c_f_pointer, c_int64_t, c_int
    use MPI
    use hipfort
    use hipfort_check
    use gpu_topology, only: gpu_topology_t, init_gpu_topology

    use gpu_transfer, only: gpu_allgatherv_dtoh, gpu_allscatterv_htod
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: wavefront_context

    interface
        subroutine launch_expectation_value_kernel(grid, block, shmem, stream, out, a, b, N) bind(c)
            use hipfort_types
            implicit none
            type(c_ptr), value :: a, b, out ! input/output arrays
            integer(c_int), value :: shmem
            integer(c_long), value :: N
            type(dim3) :: grid, block ! grid and block size (3D grid)
            type(c_ptr), value :: stream
        end subroutine launch_expectation_value_kernel
    end interface

    interface
        subroutine launch_state_norm_kernel(grid, block, shmem, stream, out, a, N) bind(c)
            use hipfort_types
            implicit none
            type(c_ptr), value :: a, out ! input/output arrays
            integer(c_int), value :: shmem
            integer(c_long), value :: N
            type(dim3) :: grid, block ! grid and block size (3D grid)
            type(c_ptr), value :: stream
        end subroutine launch_state_norm_kernel
    end interface

    type wavefront_context

        logical :: has_device = .false.
        integer(int32) :: device_ID = 0

        ! GPU topology information (cached from init_gpu_topology)
        type(gpu_topology_t) :: topology

        ! Transfer counts and displacements for vector transfers between NODECOMM and DEVCOMM_NODE
        integer(int64), dimension(:), pointer :: NODECOMM_counts => null()
        integer(int64), dimension(:), pointer :: NODECOMM_displs => null()
        integer(int64), dimension(:), pointer :: DEVCOMM_NODE_counts => null()
        integer(int64), dimension(:), pointer :: DEVCOMM_NODE_displs => null()

        complex(real64), dimension(:), pointer :: state => null()
        real(real64), dimension(:), pointer :: observables => null()
        complex(real64), dimension(:), pointer :: work => null()
        logical :: work_allocated = .false.

        ! Host-side mirrors of the device state and observables.  These
        ! point at NumPy-owned buffers attached via attach_host_*; the
        ! sync_host_*/sync_device_* methods stage data between the host
        ! mirrors and the authoritative device buffers using the
        ! existing gpu_allgatherv_dtoh / gpu_allscatterv_htod paths.
        complex(real64), dimension(:), pointer :: host_state => null()
        real(real64), dimension(:), pointer :: host_observables => null()
        real(real64), dimension(:), pointer :: host_local_probabilities => null()

        ! Pre-allocated device reduction buffer for get_expectation_value / get_state_norm.
        ! Sized to reduction_num_blocks doubles; allocated once at setup, freed at destroy.
        integer(int32) :: reduction_num_blocks = 1200
        real(real64), dimension(:), pointer :: reduction_dout => null()
        real(real64), allocatable, dimension(:) :: reduction_host_out
        logical :: reduction_allocated = .false.

        ! Pointer to the shared quop_mpi_layout_t (owned by caller, not freed here)
        type(quop_mpi_layout_t), pointer :: ci => null()

    contains

        procedure :: setup => context_setup
        procedure :: get_expectation_value => context_get_expectation_value
        procedure :: get_state_norm => context_get_state_norm
        procedure :: destroy => context_destroy
        procedure :: get_state => context_get_state
        procedure :: set_state => context_set_state
        procedure :: set_observables => context_set_observables
        procedure :: get_observables => context_get_observables

        procedure :: attach_host_state => context_attach_host_state
        procedure :: attach_host_observables => context_attach_host_observables
        procedure :: attach_host_local_probabilities => context_attach_host_local_probabilities
        procedure :: sync_host_state => context_sync_host_state
        procedure :: sync_device_state => context_sync_device_state
        procedure :: sync_host_observables => context_sync_host_observables
        procedure :: sync_device_observables => context_sync_device_observables
        procedure :: compute_local_probabilities => context_compute_local_probabilities
        procedure :: detach_host_buffers => context_detach_host_buffers

    end type wavefront_context

contains

    subroutine context_setup(self, &
                             ci, &
                             error_code)

        class(wavefront_context), intent(inout) :: self
        type(quop_mpi_layout_t), target, intent(in) :: ci
        integer(int32), intent(out) :: error_code

        integer(int32) :: NODECOMM_size
        integer(int32) :: ierr, local_error, synced_error
        integer :: alloc_status
        integer(int32) :: ci_subcomm, ci_nodecomm, ci_devcomm, ci_devcomm_node
        integer(int64) :: ci_system_size, ci_local_i, ci_local_i_offset
        integer(int64) :: ci_device_local_i, ci_device_local_i_offset, ci_alloc_local
        integer(int32) :: ci_device_n_processes

        ! Debug mode variables
        logical :: debug_backend
        character(len=64) :: env_val
        integer(int32) :: expected_devcomm_node_size, actual_devcomm_node_size
        integer(int32) :: DEVCOMM_rank, global_rank
        integer(int64) :: total_elements
        logical :: env_is_valid

        error_code = 0

        call MPI_Comm_rank(MPI_COMM_WORLD, global_rank, ierr)
        call read_env_flag('QUOP_DEBUG_BACKEND', .false., debug_backend, env_is_valid, env_val)
        if (.not. env_is_valid .and. global_rank == 0) then
            write (error_unit, '(A,A,A)') &
                'WARNING: QUOP_DEBUG_BACKEND has unrecognised value "', &
                trim(env_val), '". Using 0.'
        end if

        self%ci => ci
        ci_system_size = ci%get_system_size()
        ci_subcomm = ci%get_SUBCOMM()
        ci_nodecomm = ci%get_NODECOMM()
        ci_devcomm = ci%get_DEVCOMM()
        ci_devcomm_node = ci%get_DEVCOMM_NODE()
        ci_local_i = ci%get_local_i()
        ci_local_i_offset = ci%get_local_i_offset()
        ci_device_n_processes = int(ci%get_device_n_processes(), int32)
        ci_device_local_i = ci%get_device_local_i()
        ci_device_local_i_offset = ci%get_device_local_i_offset()
        ci_alloc_local = ci%get_device_alloc_local()

        ! Cache topology from the layout
        self%topology = ci%get_topology()

        ! ===================================================================
        ! Consistency checks (enabled with QUOP_DEBUG_BACKEND=1/true)
        ! ===================================================================
        local_error = 0
        if (debug_backend) then
            ! Check 1: DEVCOMM_NODE_size from ci should match actual DEVCOMM_NODE size
            expected_devcomm_node_size = ci_device_n_processes
            if (ci_devcomm_node /= MPI_COMM_NULL) then
                call MPI_Comm_size(ci_devcomm_node, actual_devcomm_node_size, ierr)
                if (actual_devcomm_node_size /= expected_devcomm_node_size) then
                    write (error_unit, '(A,I0,A,I0,A,I0)') &
                        "ERROR [Rank ", global_rank, "]: DEVCOMM_NODE_size mismatch: actual=", &
                        actual_devcomm_node_size, ", layout=", expected_devcomm_node_size
                    local_error = max(local_error, 1)
                end if
            end if

            ! Check 2: Verify DEVCOMM membership consistency with data
            ! Ranks with data should be in DEVCOMM, ranks without should not
            if (ci_device_local_i > 0 .and. self%topology%is_gpu_rank .and. ci_devcomm == MPI_COMM_NULL) then
                write (error_unit, '(A,I0,A)') &
                    "ERROR [Rank ", global_rank, "]: has GPU and data but DEVCOMM is NULL"
                local_error = max(local_error, 2)
            end if
            if (ci_device_local_i <= 0 .and. ci_devcomm /= MPI_COMM_NULL) then
                write (error_unit, '(A,I0,A)') &
                    "ERROR [Rank ", global_rank, "]: has no data but DEVCOMM is not NULL"
                local_error = max(local_error, 2)
            end if

            ! Check 3: Total elements across DEVCOMM equals system_size
            if (ci_devcomm /= MPI_COMM_NULL) then
                call MPI_Allreduce(ci_device_local_i, total_elements, 1, &
                                   MPI_INTEGER8, MPI_SUM, ci_devcomm, ierr)
                call MPI_Comm_rank(ci_devcomm, DEVCOMM_rank, ierr)
                if (DEVCOMM_rank == 0 .and. total_elements /= ci_system_size) then
                    write (error_unit, '(A,I0,A,I0)') &
                        "ERROR: Total DEVCOMM elements=", total_elements, &
                        " does not match system_size=", ci_system_size
                    local_error = max(local_error, 3)
                end if
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

        ! Set device ownership based on DEVCOMM membership (has GPU AND data)
        if (ci_devcomm /= MPI_COMM_NULL) then
            self%device_ID = self%topology%assigned_device_id
            self%has_device = .true.
            call hipCheck(hipSetDevice(self%device_ID))
        else
            self%has_device = .false.
        end if

        ! Determine NODECOMM size for transfer metadata arrays.
        call MPI_Comm_size(ci_nodecomm, NODECOMM_size, ierr)

        ! Per-rank host/device counts + global offsets over NODECOMM, consumed
        ! by gpu_allgatherv_dtoh / gpu_allscatterv_htod overlap computation.
        ! Must be per-rank distribution (not per-partner schedule); matters when
        ! host/device partitions misalign, e.g. prime sizes where SHAFFT assigns
        ! all elements to one GPU.

        local_error = 0
        allocate (self%NODECOMM_counts(NODECOMM_size), stat=alloc_status)
        if (alloc_status /= 0) local_error = 10
        if (local_error == 0) then
            allocate (self%NODECOMM_displs(NODECOMM_size), stat=alloc_status)
            if (alloc_status /= 0) local_error = 11
        end if
        if (local_error == 0) then
            allocate (self%DEVCOMM_NODE_counts(NODECOMM_size), stat=alloc_status)
            if (alloc_status /= 0) local_error = 12
        end if
        if (local_error == 0) then
            allocate (self%DEVCOMM_NODE_displs(NODECOMM_size), stat=alloc_status)
            if (alloc_status /= 0) local_error = 13
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)
        error_code = synced_error
        if (synced_error /= 0) then
            call self%destroy()
            return
        end if

        call MPI_Allgather(ci_local_i, 1, MPI_INTEGER8, &
                           self%NODECOMM_counts, 1, MPI_INTEGER8, &
                           ci_nodecomm, ierr)
        call MPI_Allgather(ci_local_i_offset, 1, MPI_INTEGER8, &
                           self%NODECOMM_displs, 1, MPI_INTEGER8, &
                           ci_nodecomm, ierr)
        call MPI_Allgather(ci_device_local_i, 1, MPI_INTEGER8, &
                           self%DEVCOMM_NODE_counts, 1, MPI_INTEGER8, &
                           ci_nodecomm, ierr)
        call MPI_Allgather(ci_device_local_i_offset, 1, MPI_INTEGER8, &
                           self%DEVCOMM_NODE_displs, 1, MPI_INTEGER8, &
                           ci_nodecomm, ierr)

        ! Allocate device memory for the state and observables vectors
        ! Use alloc_local which may be larger than DEVCOMM_local_i if padding is needed
        ! The hipfort interface takes element counts (not bytes) for typed pointers
        if (self%has_device) then

            local_error = 0

            call hipCheck(hipMalloc(self%state, ci_alloc_local))
            if (self%ci%get_requires_device_work_buffer()) then
                call hipCheck(hipMalloc(self%work, ci_alloc_local))
                self%work_allocated = .true.
            end if

            call hipCheck(hipMalloc(self%observables, ci_device_local_i))

            ! Pre-allocate device reduction buffer for get_expectation_value / get_state_norm
            call hipCheck(hipMalloc(self%reduction_dout, self%reduction_num_blocks))
            allocate (self%reduction_host_out(self%reduction_num_blocks), stat=alloc_status)
            if (alloc_status /= 0) then
                local_error = 14
            else
                self%reduction_allocated = .true.
            end if

            ! Zero-initialize the buffers (important for padding -- padding bytes
            ! can leak into results after pointer swaps)
            if (local_error == 0) then
                call hipCheck(hipMemset(c_loc(self%state), 0, int(ci_alloc_local * 16, c_size_t)))
            end if
            if (local_error == 0 .and. self%work_allocated) then
                call hipCheck(hipMemset(c_loc(self%work), 0, int(ci_alloc_local * 16, c_size_t)))
            end if
            if (local_error == 0) then
                call hipCheck(hipMemset(c_loc(self%observables), 0, int(ci_device_local_i * 8, c_size_t)))
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

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine context_setup

    subroutine context_destroy(self)
        class(wavefront_context), intent(inout) :: self

        if (self%has_device) then

            ! Ensure correct device is active before freeing GPU memory.
            ! This is critical when destroy() is called from Python GC (__del__),
            ! where the HIP runtime's current device may have been changed.
            call hipCheck(hipSetDevice(self%device_ID))
            call hipCheck(hipDeviceSynchronize())

            if (associated(self%work)) then
                call hipCheck(hipFree(self%work))
                self%work => null()
            end if
            self%work_allocated = .false.

            if (associated(self%state)) then
                call hipCheck(hipFree(self%state))
                self%state => null()
            end if

            if (associated(self%observables)) then
                call hipCheck(hipFree(self%observables))
                self%observables => null()
            end if

            ! Free pre-allocated reduction buffer
            if (self%reduction_allocated .and. associated(self%reduction_dout)) then
                call hipCheck(hipFree(self%reduction_dout))
                self%reduction_dout => null()
            end if
            if (allocated(self%reduction_host_out)) then
                deallocate (self%reduction_host_out)
            end if
            self%reduction_allocated = .false.

        end if

        if (associated(self%NODECOMM_counts)) then
            deallocate (self%NODECOMM_counts)
        end if
        if (associated(self%NODECOMM_displs)) then
            deallocate (self%NODECOMM_displs)
        end if
        if (associated(self%DEVCOMM_NODE_counts)) then
            deallocate (self%DEVCOMM_NODE_counts)
        end if
        if (associated(self%DEVCOMM_NODE_displs)) then
            deallocate (self%DEVCOMM_NODE_displs)
        end if

        self%device_ID = 0

        ! Nullify host mirrors (Python-owned memory; nothing to free here).
        self%host_state => null()
        self%host_observables => null()
        self%host_local_probabilities => null()

        ! Nullify layout pointer (not owned)
        self%ci => null()
        self%has_device = .false.

        ! NOTE: No MPI_Barrier here. GPU resource cleanup (hipFree) is rank-local
        ! and does not require synchronisation. A barrier in a destructor that can
        ! be invoked from Python's __del__ / GC is inherently unsafe -- GC timing
        ! is non-deterministic across MPI ranks and would cause deadlocks.

    end subroutine context_destroy

    real(real64) function context_get_expectation_value(self, error_code)
        !! Collective over SUBCOMM.
        !! The scalar result is defined on all active SUBCOMM ranks.
        class(wavefront_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        real(real64) :: local_expectation_value
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_device_local_i

        context_get_expectation_value = 0.0_real64
        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_device_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_device_local_i = self%ci%get_device_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (self%has_device) then
            if (.not. associated(self%state) .or. &
                .not. associated(self%observables) .or. &
                .not. self%reduction_allocated .or. &
                .not. associated(self%reduction_dout) .or. &
                .not. allocated(self%reduction_host_out)) then
                local_error = 1
            end if
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

        local_expectation_value = 0.0_real64

        if (self%has_device) then

            call launch_expectation_value_kernel(dim3(self%reduction_num_blocks), &
                                                 dim3(256), &
                                                 0, c_null_ptr, &
                                                 c_loc(self%reduction_dout), &
                                                 c_loc(self%state), &
                                                 c_loc(self%observables), &
                                                 ci_device_local_i)

            call hipCheck(hipDeviceSynchronize())

            call hipCheck(hipMemcpy(self%reduction_host_out, self%reduction_dout, &
                                    self%reduction_num_blocks, hipMemcpyDeviceToHost))

            local_expectation_value = sum(self%reduction_host_out)

        end if

        call MPI_Allreduce(local_expectation_value, &
                           context_get_expectation_value, &
                           1, &
                           MPI_DOUBLE, &
                           MPI_SUM, &
                           ci_subcomm, &
                           ierr)

    end function context_get_expectation_value

    real(real64) function context_get_state_norm(self, error_code)
        !! Collective over SUBCOMM.
        !! The scalar result is defined on all active SUBCOMM ranks.
        class(wavefront_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        real(real64) :: local_state_norm
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_device_local_i

        context_get_state_norm = 0.0_real64
        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_device_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 1
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_device_local_i = self%ci%get_device_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL) then
            local_error = 1
        else if (self%has_device) then
            if (.not. associated(self%state) .or. &
                .not. self%reduction_allocated .or. &
                .not. associated(self%reduction_dout) .or. &
                .not. allocated(self%reduction_host_out)) then
                local_error = 1
            end if
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

        local_state_norm = 0.0_real64

        if (self%has_device) then

            call launch_state_norm_kernel(dim3(self%reduction_num_blocks), &
                                          dim3(256), &
                                          0, c_null_ptr, &
                                          c_loc(self%reduction_dout), &
                                          c_loc(self%state), &
                                          ci_device_local_i)

            call hipCheck(hipDeviceSynchronize())
            call hipCheck(hipMemcpy(self%reduction_host_out, self%reduction_dout, &
                                    self%reduction_num_blocks, hipMemcpyDeviceToHost))

            local_state_norm = sum(self%reduction_host_out)

        end if

        call MPI_Allreduce(local_state_norm, &
                           context_get_state_norm, &
                           1, &
                           MPI_DOUBLE, &
                           MPI_SUM, &
                           ci_subcomm, &
                           ierr)

    end function context_get_state_norm

    subroutine context_get_state(self, state, error_code)
        !! Collective over SUBCOMM.
        !! This gathers the host-visible state buffer from the distributed
        !! device-backed representation used by the wavefront backend.
        class(wavefront_context), intent(in) :: self
        complex(real64), target, dimension(:), intent(inout) :: state
        integer(int32), intent(out) :: error_code
        integer(c_int) :: ierr
        integer(int32) :: local_error, synced_error
        integer(int32) :: ci_subcomm, ci_nodecomm
        integer(int64) :: ci_local_i
        type(c_ptr) :: dev_ptr

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_nodecomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 2
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_nodecomm = self%ci%get_NODECOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL .or. ci_nodecomm == MPI_COMM_NULL) then
            local_error = 2
        else if (.not. associated(self%NODECOMM_counts) .or. &
                 .not. associated(self%NODECOMM_displs) .or. &
                 .not. associated(self%DEVCOMM_NODE_counts) .or. &
                 .not. associated(self%DEVCOMM_NODE_displs)) then
            local_error = 2
        else if (size(state) < ci_local_i) then
            local_error = 1
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        call MPI_Barrier(ci_subcomm, ierr)

        ! Pass C_NULL_PTR for device pointer if this rank doesn't have a device
        if (associated(self%state)) then
            dev_ptr = c_loc(self%state)
        else
            dev_ptr = C_NULL_PTR
        end if

        call gpu_allgatherv_dtoh(self%DEVCOMM_NODE_counts, &
                                 self%NODECOMM_counts, &
                                 self%DEVCOMM_NODE_displs, &
                                 self%NODECOMM_displs, &
                                 dev_ptr, &
                                 c_loc(state), &
                                 MPI_DOUBLE_COMPLEX, &
                                 ci_nodecomm)

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine context_get_state

    subroutine context_set_state(self, state, error_code)
        !! Collective over SUBCOMM.
        !! This scatters the host-visible state buffer into the distributed
        !! device-backed representation used by the wavefront backend.
        class(wavefront_context), intent(inout) :: self
        complex(real64), target, dimension(:), intent(in) :: state
        integer(int32), intent(out) :: error_code
        integer(c_int) :: ierr
        integer(int32) :: local_error, synced_error
        integer(int32) :: ci_subcomm, ci_nodecomm
        integer(int64) :: ci_local_i
        type(c_ptr) :: dev_ptr

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_nodecomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 2
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_nodecomm = self%ci%get_NODECOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL .or. ci_nodecomm == MPI_COMM_NULL) then
            local_error = 2
        else if (.not. associated(self%NODECOMM_counts) .or. &
                 .not. associated(self%NODECOMM_displs) .or. &
                 .not. associated(self%DEVCOMM_NODE_counts) .or. &
                 .not. associated(self%DEVCOMM_NODE_displs)) then
            local_error = 2
        else if (size(state) < ci_local_i) then
            local_error = 1
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        call MPI_Barrier(ci_subcomm, ierr)

        ! Pass C_NULL_PTR for device pointer if this rank doesn't have a device
        if (associated(self%state)) then
            dev_ptr = c_loc(self%state)
        else
            dev_ptr = C_NULL_PTR
        end if

        call gpu_allscatterv_htod(self%NODECOMM_counts, &
                                  self%DEVCOMM_NODE_counts, &
                                  self%NODECOMM_displs, &
                                  self%DEVCOMM_NODE_displs, &
                                  c_loc(state), &
                                  dev_ptr, &
                                  MPI_DOUBLE_COMPLEX, &
                                  ci_nodecomm)

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine context_set_state

    subroutine context_get_observables(self, obs, error_code)
        !! Collective over SUBCOMM.
        !! This gathers the host-visible observables buffer from the
        !! distributed device-backed representation used by wavefront.
        class(wavefront_context), intent(in) :: self
        real(real64), target, dimension(:), intent(inout) :: obs
        integer(int32), intent(out) :: error_code
        integer(c_int) :: ierr
        integer(int32) :: local_error, synced_error
        integer(int32) :: ci_subcomm, ci_nodecomm
        integer(int64) :: ci_local_i
        type(c_ptr) :: dev_ptr

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_nodecomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 2
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_nodecomm = self%ci%get_NODECOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL .or. ci_nodecomm == MPI_COMM_NULL) then
            local_error = 2
        else if (.not. associated(self%NODECOMM_counts) .or. &
                 .not. associated(self%NODECOMM_displs) .or. &
                 .not. associated(self%DEVCOMM_NODE_counts) .or. &
                 .not. associated(self%DEVCOMM_NODE_displs)) then
            local_error = 2
        else if (size(obs) < ci_local_i) then
            local_error = 1
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        call MPI_Barrier(ci_subcomm, ierr)

        ! Pass C_NULL_PTR for device pointer if this rank doesn't have a device
        if (associated(self%observables)) then
            dev_ptr = c_loc(self%observables)
        else
            dev_ptr = C_NULL_PTR
        end if

        call gpu_allgatherv_dtoh(self%DEVCOMM_NODE_counts, &
                                 self%NODECOMM_counts, &
                                 self%DEVCOMM_NODE_displs, &
                                 self%NODECOMM_displs, &
                                 dev_ptr, &
                                 c_loc(obs), &
                                 MPI_DOUBLE, &
                                 ci_nodecomm)

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine context_get_observables

    subroutine context_set_observables(self, obs, error_code)
        !! Collective over SUBCOMM.
        !! This scatters the host-visible observables buffer into the
        !! distributed device-backed representation used by wavefront.
        class(wavefront_context), intent(inout) :: self
        real(real64), target, dimension(:), intent(in) :: obs
        integer(int32), intent(out) :: error_code
        integer(c_int) :: ierr
        integer(int32) :: local_error, synced_error
        integer(int32) :: ci_subcomm, ci_nodecomm
        integer(int64) :: ci_local_i
        type(c_ptr) :: dev_ptr

        error_code = 0

        ci_subcomm = MPI_COMM_NULL
        ci_nodecomm = MPI_COMM_NULL
        ci_local_i = 0_int64
        local_error = 0
        if (.not. associated(self%ci)) then
            local_error = 2
        else
            ci_subcomm = self%ci%get_SUBCOMM()
            ci_nodecomm = self%ci%get_NODECOMM()
            ci_local_i = self%ci%get_local_i()
        end if
        if (ci_subcomm == MPI_COMM_NULL .or. ci_nodecomm == MPI_COMM_NULL) then
            local_error = 2
        else if (.not. associated(self%NODECOMM_counts) .or. &
                 .not. associated(self%NODECOMM_displs) .or. &
                 .not. associated(self%DEVCOMM_NODE_counts) .or. &
                 .not. associated(self%DEVCOMM_NODE_displs)) then
            local_error = 2
        else if (size(obs) < ci_local_i) then
            local_error = 1
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) return

        call MPI_Barrier(ci_subcomm, ierr)

        ! Pass C_NULL_PTR for device pointer if this rank doesn't have a device
        ! The transfer will still work - only ranks with devices need valid device pointers
        if (associated(self%observables)) then
            dev_ptr = c_loc(self%observables)
        else
            dev_ptr = C_NULL_PTR
        end if

        call gpu_allscatterv_htod(self%NODECOMM_counts, &
                                  self%DEVCOMM_NODE_counts, &
                                  self%NODECOMM_displs, &
                                  self%DEVCOMM_NODE_displs, &
                                  c_loc(obs), &
                                  dev_ptr, &
                                  MPI_DOUBLE, &
                                  ci_nodecomm)

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine context_set_observables

    ! ====================================================================
    ! Host mirror / sync contract.
    !
    ! Wavefront keeps the authoritative state and observables in device
    ! memory.  Python provides separately-allocated host buffers via
    ! attach_host_*; sync_host_* refresh them with a device->host gather
    ! using the same path as get_state / get_observables, and
    ! sync_device_* push them back with the existing scatter path.
    ! Local probabilities are computed host-side after a sync_host_state
    ! to keep device memory free for the simulation buffers.
    ! ====================================================================

    subroutine context_attach_host_state(self, ptr, n)
        !! Bind a Python-owned host buffer of length `n` as the host
        !! mirror for the state.  Does NOT touch %state -- that is the
        !! device pointer owned by hipMalloc/hipFree.
        class(wavefront_context), intent(inout) :: self
        type(c_ptr), value, intent(in) :: ptr
        integer(c_int64_t), value, intent(in) :: n

        self%host_state => null()
        call c_f_pointer(ptr, self%host_state, [n])
    end subroutine context_attach_host_state

    subroutine context_attach_host_observables(self, ptr, n)
        !! Mirror of attach_host_state for the observables buffer.
        class(wavefront_context), intent(inout) :: self
        type(c_ptr), value, intent(in) :: ptr
        integer(c_int64_t), value, intent(in) :: n

        self%host_observables => null()
        call c_f_pointer(ptr, self%host_observables, [n])
    end subroutine context_attach_host_observables

    subroutine context_attach_host_local_probabilities(self, ptr, n)
        !! Bind a Python-owned host buffer for local probabilities.
        class(wavefront_context), intent(inout) :: self
        type(c_ptr), value, intent(in) :: ptr
        integer(c_int64_t), value, intent(in) :: n

        self%host_local_probabilities => null()
        call c_f_pointer(ptr, self%host_local_probabilities, [n])
    end subroutine context_attach_host_local_probabilities

    subroutine context_sync_host_state(self, error_code)
        !! Refresh host_state from the device buffer (device->host gather).
        !!
        !! Collective over SUBCOMM.  The actual data movement is a
        !! NODECOMM-scoped gpu_allgatherv_dtoh; SUBCOMM-scoped
        !! MPI_Allreduce / MPI_Barrier in self%get_state synchronise
        !! error_code and ordering across nodes.
        class(wavefront_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (.not. associated(self%host_state)) then
            error_code = 2
            return
        end if
        call self%get_state(self%host_state, error_code)
    end subroutine context_sync_host_state

    subroutine context_sync_device_state(self, error_code)
        !! Push host_state to the device buffer (host->device scatter).
        !! Collective over SUBCOMM (see sync_host_state).
        class(wavefront_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (.not. associated(self%host_state)) then
            error_code = 2
            return
        end if
        call self%set_state(self%host_state, error_code)
    end subroutine context_sync_device_state

    subroutine context_sync_host_observables(self, error_code)
        !! Refresh host_observables from the device buffer.
        !! Collective over SUBCOMM (see sync_host_state).
        class(wavefront_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (.not. associated(self%host_observables)) then
            error_code = 2
            return
        end if
        call self%get_observables(self%host_observables, error_code)
    end subroutine context_sync_host_observables

    subroutine context_sync_device_observables(self, error_code)
        !! Push host_observables to the device buffer.
        !! Collective over SUBCOMM (see sync_host_state).
        class(wavefront_context), intent(inout) :: self
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (.not. associated(self%host_observables)) then
            error_code = 2
            return
        end if
        call self%set_observables(self%host_observables, error_code)
    end subroutine context_sync_device_observables

    subroutine context_compute_local_probabilities(self, error_code)
        !! Fill host_local_probabilities(1:local_i) with |state(i)|^2.
        !! Performs sync_host_state first to bring the device-owned
        !! state into the host mirror, then runs the host-side
        !! magnitude-squared loop.  Keeping the |psi|^2 reduction on
        !! host avoids tying up device memory for an extra real64
        !! buffer of length local_i.
        !!
        !! Collective over SUBCOMM (inherits from sync_host_state).
        class(wavefront_context), intent(inout) :: self
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
        !! Nullify host-mirror pointers to Python-owned memory before
        !! destroy() runs.  The device %state and %observables are
        !! left intact -- they will be hipFree'd by context_destroy.
        class(wavefront_context), intent(inout) :: self

        self%host_state => null()
        self%host_observables => null()
        self%host_local_probabilities => null()
    end subroutine context_detach_host_buffers

    subroutine read_env_flag(name, default_value, value, env_is_valid, raw_value)
        character(len=*), intent(in) :: name
        logical, intent(in) :: default_value
        logical, intent(out) :: value
        logical, intent(out) :: env_is_valid
        character(len=*), intent(out) :: raw_value

        raw_value = ''
        call get_environment_variable(name, raw_value)
        raw_value = trim(adjustl(raw_value))

        if (len_trim(raw_value) == 0) then
            value = default_value
            env_is_valid = .true.
            return
        end if

        call lowercase_inplace(raw_value)
        select case (trim(raw_value))
        case ('1', 'true', 'yes', 'on')
            value = .true.
            env_is_valid = .true.
        case ('0', 'false', 'no', 'off')
            value = .false.
            env_is_valid = .true.
        case default
            value = default_value
            env_is_valid = .false.
        end select
    end subroutine read_env_flag

    subroutine lowercase_inplace(str)
        character(len=*), intent(inout) :: str

        integer :: i, code

        do i = 1, len_trim(str)
            code = iachar(str(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                str(i:i) = achar(code + 32)
            end if
        end do
    end subroutine lowercase_inplace

end module wavefront
