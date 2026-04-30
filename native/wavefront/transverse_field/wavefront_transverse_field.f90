!------------------------------------------------------------------------------
!> @brief Wavefront (HIP/GPU) transverse-field propagator.
!>
!> @details Applies the uniform transverse-field mixer
!>   U(theta) = exp(-i theta/2 sum_j X_j)
!> on GPU state vectors distributed over DEVCOMM. Uses the shared
!> transverse_field_common module for layout classification and segment
!> computation, with HIP kernels for local pair updates and remote exchanges.
!>
!> Communication for remote qubits uses MPI_Sendrecv on DEVCOMM, with
!> GPU-aware or host-staged transfer depending on QUOP_GPU_AWARE_MPI.
!------------------------------------------------------------------------------
module wavefront_transverse_field

    use, intrinsic :: iso_fortran_env, only: real64, int32, int64, error_unit
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_f_pointer, c_loc, c_size_t, c_associated
    use mpi
    use hipfort
    use hipfort_check
    use wavefront, only: wavefront_context
    use comm_info_module, only: quop_mpi_layout_t
    use transverse_field_common
    use hip_transverse_field_kernels

    implicit none

    private

    public :: transverse_field_propagator

    type transverse_field_propagator

        type(wavefront_context), pointer :: context => null()

        ! Device buffers for MPI exchange
        type(c_ptr) :: sendbuf_dev = c_null_ptr
        type(c_ptr) :: recvbuf_dev = c_null_ptr

        ! Host staging buffers (non-GPU-aware MPI only)
        complex(real64), pointer, dimension(:) :: sendbuf_host => null()
        complex(real64), pointer, dimension(:) :: recvbuf_host => null()

        integer(int64), allocatable, dimension(:) :: partition_table_0

        integer(int32) :: rank = 0
        integer(int32) :: comm_size = 0
        integer(int32) :: n_qubits = 0
        integer(int32) :: n_local_qubits = 0
        integer(int32) :: layout_mode = TF_MODE_UNSET
        integer(int64) :: local_i = 0_int64
        integer(int64) :: chunk_elems = 0_int64
        integer(int64) :: lb_global = 0_int64
        integer(int64) :: ub_global = -1_int64

    contains

        procedure :: max_comm_size => wf_transverse_field_max_comm_size
        procedure :: store_constraints => wf_transverse_field_store_constraints
        procedure :: plan => wf_transverse_field_plan
        procedure :: gen_operator => wf_transverse_field_gen_operator
        procedure :: propagate => wf_transverse_field_propagate
        procedure :: destroy => wf_transverse_field_destroy

    end type transverse_field_propagator

contains

    subroutine wf_transverse_field_reset_state(self)
        class(transverse_field_propagator), intent(inout) :: self

        if (c_associated(self%sendbuf_dev)) then
            call hipCheck(hipFree(self%sendbuf_dev))
            self%sendbuf_dev = c_null_ptr
        end if
        if (c_associated(self%recvbuf_dev)) then
            call hipCheck(hipFree(self%recvbuf_dev))
            self%recvbuf_dev = c_null_ptr
        end if
        if (associated(self%sendbuf_host)) then
            deallocate (self%sendbuf_host)
            nullify (self%sendbuf_host)
        end if
        if (associated(self%recvbuf_host)) then
            deallocate (self%recvbuf_host)
            nullify (self%recvbuf_host)
        end if
        if (allocated(self%partition_table_0)) deallocate (self%partition_table_0)

        self%context => null()
        self%rank = 0
        self%comm_size = 0
        self%n_qubits = 0
        self%n_local_qubits = 0
        self%layout_mode = TF_MODE_UNSET
        self%local_i = 0_int64
        self%chunk_elems = 0_int64
        self%lb_global = 0_int64
        self%ub_global = -1_int64
    end subroutine wf_transverse_field_reset_state

    subroutine wf_transverse_field_sync_error(self, local_error, error_code)
        class(transverse_field_propagator), intent(in) :: self
        integer(int32), intent(in) :: local_error
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, ci_subcomm

        error_code = local_error
        ci_subcomm = self%context%ci%get_SUBCOMM()
        call MPI_Allreduce(local_error, error_code, 1, MPI_INTEGER4, MPI_MAX, ci_subcomm, ierr)
    end subroutine wf_transverse_field_sync_error

    subroutine wf_transverse_field_build_device_partition_table( &
        self, device_local_i, device_local_i_offset, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        integer(int64), intent(in) :: device_local_i, device_local_i_offset
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, idx, ci_devcomm
        integer(int64) :: expected_offset
        integer(int64), allocatable :: all_device_local_i(:), all_device_offsets(:)

        error_code = 0

        if (self%comm_size <= 0) then
            write (error_unit, '(A,I0)') &
                'ERROR: transverse_field requires a valid DEVCOMM size, got ', self%comm_size
            error_code = 1
            return
        end if

        ci_devcomm = self%context%ci%get_DEVCOMM()

        allocate (all_device_local_i(self%comm_size))
        allocate (all_device_offsets(self%comm_size))

        call MPI_Allgather(device_local_i, 1, MPI_INTEGER8, all_device_local_i, 1, MPI_INTEGER8, ci_devcomm, ierr)
        call MPI_Allgather(device_local_i_offset, 1, MPI_INTEGER8, all_device_offsets, 1, MPI_INTEGER8, ci_devcomm, ierr)

        expected_offset = 0_int64
        do idx = 1, self%comm_size
            if (all_device_offsets(idx) /= expected_offset) then
                write (error_unit, '(A,I0,A,I0,A,I0)') &
                    'ERROR: transverse_field requires contiguous DEVCOMM offsets: rank=', idx - 1, &
                    ', got ', all_device_offsets(idx), ', expected ', expected_offset
                error_code = 1
                deallocate (all_device_local_i, all_device_offsets)
                return
            end if
            expected_offset = expected_offset + all_device_local_i(idx)
        end do

        if (allocated(self%partition_table_0)) then
            deallocate (self%partition_table_0)
        end if
        allocate (self%partition_table_0(self%comm_size + 1))

        self%partition_table_0(1) = 0_int64
        do idx = 1, self%comm_size
            self%partition_table_0(idx + 1) = self%partition_table_0(idx) + all_device_local_i(idx)
        end do

        self%lb_global = device_local_i_offset
        self%ub_global = device_local_i_offset + device_local_i - 1_int64

        deallocate (all_device_local_i, all_device_offsets)
    end subroutine wf_transverse_field_build_device_partition_table

    subroutine wf_transverse_field_max_comm_size(self, ci, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int64) :: system_size, available_ranks, target_ranks
        integer(int32) :: ierr

        error_code = 0

        system_size = ci%get_system_size()
        available_ranks = ci%get_n_processes()

        if (.not. tf_is_power_of_two(system_size)) then
            write (error_unit, '(A,I0)') &
                "ERROR: transverse_field requires power-of-two system_size, got ", system_size
            error_code = 1
            return
        end if

        if (available_ranks <= 0_int64) then
            write (error_unit, '(A,I0)') &
                "ERROR: transverse_field received invalid communicator size ", available_ranks
            error_code = 1
            return
        end if

        target_ranks = min(system_size, available_ranks)
        call ci%set_n_processes(target_ranks, error_code)
        if (error_code /= 0) return

        call MPI_Barrier(ci%get_SUBCOMM(), ierr)
    end subroutine wf_transverse_field_max_comm_size

    subroutine wf_transverse_field_store_constraints(self, constraint_ptrs, constraint_sizes)
        class(transverse_field_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine wf_transverse_field_store_constraints

    subroutine wf_transverse_field_plan(self, context, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(wavefront_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_devcomm
        integer(int64) :: desired_chunk, buf_bytes, device_local_i

        error_code = 0
        local_error = 0

        call wf_transverse_field_reset_state(self)
        self%context => context

        if (self%context%has_device) then
            ci_devcomm = self%context%ci%get_DEVCOMM()
            call MPI_Comm_rank(ci_devcomm, self%rank, ierr)
            call MPI_Comm_size(ci_devcomm, self%comm_size, ierr)

            device_local_i = self%context%ci%get_device_local_i()
            self%local_i = device_local_i

            if (device_local_i <= 0_int64) then
                write (error_unit, '(A,I0)') &
                    'ERROR: transverse_field requires positive device_local_i, got ', device_local_i
                local_error = 1
            end if
        end if

        if (local_error == 0 .and. self%context%has_device) then
            desired_chunk = DEFAULT_CHUNK_BYTES / COMPLEX128_BYTES
            self%chunk_elems = min(self%local_i, desired_chunk)
            if (self%chunk_elems < 1_int64) self%chunk_elems = 1_int64

            ! Allocate device send/recv buffers
            buf_bytes = self%chunk_elems * COMPLEX128_BYTES
            call hipCheck(hipMalloc(self%sendbuf_dev, buf_bytes))
            call hipCheck(hipMalloc(self%recvbuf_dev, buf_bytes))

#ifndef QUOP_GPU_AWARE_MPI
            ! Allocate host staging buffers for non-GPU-aware MPI
            allocate (self%sendbuf_host(self%chunk_elems))
            allocate (self%recvbuf_host(self%chunk_elems))
#endif
        end if

        call wf_transverse_field_sync_error(self, local_error, synced_error)
        error_code = synced_error
        if (synced_error /= 0) then
            call wf_transverse_field_reset_state(self)
        end if
    end subroutine wf_transverse_field_plan

    subroutine wf_transverse_field_gen_operator(self, array_ptrs, array_sizes, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, local_error, synced_error
        integer(int64) :: system_size, device_local_i, device_local_i_offset, alloc_local
        integer(int32) :: ci_devcomm
        logical :: exact_power

        error_code = 0
        local_error = 0

        if (self%context%has_device) then
            ci_devcomm = self%context%ci%get_DEVCOMM()
            call MPI_Comm_rank(ci_devcomm, self%rank, ierr)
            call MPI_Comm_size(ci_devcomm, self%comm_size, ierr)
        end if

        system_size = self%context%ci%get_system_size()

        call tf_exact_log2(system_size, self%n_qubits, exact_power)
        if (.not. exact_power) then
            write (error_unit, '(A,I0)') &
                "ERROR: transverse_field requires power-of-two system_size, got ", system_size
            local_error = 1
        end if

        if (local_error == 0 .and. self%context%has_device) then
            device_local_i = self%context%ci%get_device_local_i()
            device_local_i_offset = self%context%ci%get_device_local_i_offset()
            alloc_local = self%context%ci%get_device_alloc_local()

            self%local_i = device_local_i

            call wf_transverse_field_build_device_partition_table( &
                self, device_local_i, device_local_i_offset, local_error)
        end if

        if (local_error == 0 .and. self%context%has_device .and. alloc_local < device_local_i) then
            write (error_unit, '(A,I0,A,I0)') &
                "ERROR: transverse_field requires device_alloc_local >= device_local_i, got device_alloc_local=", &
                alloc_local, ", device_local_i=", device_local_i
            local_error = 1
        end if

        if (local_error == 0 .and. self%context%has_device) then
            call tf_classify_layout(system_size, device_local_i, device_local_i_offset, &
                                   self%comm_size, self%rank, self%layout_mode, &
                                   self%n_local_qubits, local_error)
        end if

        call wf_transverse_field_sync_error(self, local_error, synced_error)
        error_code = synced_error

        if (synced_error /= 0) then
            call wf_transverse_field_reset_state(self)
        end if
    end subroutine wf_transverse_field_gen_operator

    subroutine wf_transverse_field_exchange_remote_segment( &
        self, psi_dev, q, seg, coeff_diag, coeff_offdiag, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(c_ptr), intent(in) :: psi_dev
        integer(int32), intent(in) :: q
        type(tf_segment_t), intent(in) :: seg
        complex(real64), intent(in) :: coeff_diag, coeff_offdiag
        integer(int32), intent(out) :: error_code

        integer(int64) :: chunk_g0, chunk_g1, m, local0, buf_bytes
        integer(int32) :: count, ierr
        integer(int32) :: ci_devcomm
        integer(int32) :: status(MPI_STATUS_SIZE)
#ifdef QUOP_GPU_AWARE_MPI
        complex(real64), pointer :: sendbuf_fptr(:), recvbuf_fptr(:)
#endif

        error_code = 0
        ci_devcomm = self%context%ci%get_DEVCOMM()

        chunk_g0 = seg%g0
        do while (chunk_g0 <= seg%g1)
            chunk_g1 = min(seg%g1, chunk_g0 + self%chunk_elems - 1_int64)
            m = chunk_g1 - chunk_g0 + 1_int64
            count = int(m, int32)
            buf_bytes = m * COMPLEX128_BYTES

            local0 = chunk_g0 - self%lb_global

            ! Pack local data into send buffer on device
            call launch_tf_pack_send_kernel(self%sendbuf_dev, psi_dev, local0, m, c_null_ptr)

#ifdef QUOP_GPU_AWARE_MPI
            ! GPU-aware: synchronize then MPI directly on device buffers
            call hipCheck(hipDeviceSynchronize())

            call c_f_pointer(self%sendbuf_dev, sendbuf_fptr, [count])
            call c_f_pointer(self%recvbuf_dev, recvbuf_fptr, [count])

            call MPI_Sendrecv(sendbuf_fptr, count, MPI_DOUBLE_COMPLEX, seg%owner, q, &
                              recvbuf_fptr, count, MPI_DOUBLE_COMPLEX, seg%owner, q, &
                              ci_devcomm, status, ierr)

            call hipCheck(hipDeviceSynchronize())
#else
            ! Host-staged: D→H, MPI on host, H→D
            call hipCheck(hipMemcpy(c_loc(self%sendbuf_host), self%sendbuf_dev, &
                                    buf_bytes, hipMemcpyDeviceToHost))

            call MPI_Sendrecv(self%sendbuf_host, count, MPI_DOUBLE_COMPLEX, seg%owner, q, &
                              self%recvbuf_host, count, MPI_DOUBLE_COMPLEX, seg%owner, q, &
                              ci_devcomm, status, ierr)

            call hipCheck(hipMemcpy(self%recvbuf_dev, c_loc(self%recvbuf_host), &
                                    buf_bytes, hipMemcpyHostToDevice))
#endif

            if (ierr /= MPI_SUCCESS) then
                error_code = 1
                return
            end if

            ! Apply: psi[j] = c*psi[j] + a*recv[j]
            call launch_tf_remote_update_kernel(psi_dev, self%recvbuf_dev, local0, m, &
                                               coeff_diag, coeff_offdiag, c_null_ptr)

            chunk_g0 = chunk_g1 + 1_int64
        end do
    end subroutine wf_transverse_field_exchange_remote_segment

    subroutine wf_transverse_field_propagate_segmented(self, psi_dev, coeff_diag, coeff_offdiag, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(c_ptr), intent(in) :: psi_dev
        complex(real64), intent(in) :: coeff_diag, coeff_offdiag
        integer(int32), intent(out) :: error_code

        integer(int32) :: q, owner, n_remote, remote_idx, remote_cap
        integer(int64) :: bit_mask, g, gp, g_end, delta, seg_count
        logical :: is_lower_half
        type(tf_segment_t), allocatable :: remote_segs(:)

        error_code = 0

        if (.not. allocated(self%partition_table_0)) then
            error_code = 1
            return
        end if

        remote_cap = 0

        do q = 0, self%n_qubits - 1
            bit_mask = ishft(1_int64, q)

            n_remote = 0
            g = self%lb_global

            do while (g <= self%ub_global)
                gp = ieor(g, bit_mask)
                owner = tf_find_owner_0(gp, self%partition_table_0)
                is_lower_half = (iand(g, bit_mask) == 0_int64)
                delta = tf_partner_delta(g, bit_mask)

                if (owner == self%rank .and. .not. is_lower_half) then
                    g_end = min(self%ub_global, tf_current_half_band_end(g, bit_mask))
                    g = g_end + 1_int64
                    cycle
                end if

                g_end = tf_max_segment_end( &
                    g, bit_mask, delta, owner, self%ub_global, self%partition_table_0)

                if (owner == self%rank) then
                    seg_count = g_end - g + 1_int64
                    call launch_tf_local_pair_kernel(psi_dev, self%lb_global, g, seg_count, &
                                                    delta, coeff_diag, coeff_offdiag, c_null_ptr)
                else
                    n_remote = n_remote + 1
                    if (n_remote > remote_cap) then
                        call tf_grow_remote_arrays(remote_segs, remote_cap)
                    end if
                    remote_segs(n_remote)%g0 = g
                    remote_segs(n_remote)%g1 = g_end
                    remote_segs(n_remote)%delta = delta
                    remote_segs(n_remote)%owner = owner
                    remote_segs(n_remote)%exchange_key = min(g, gp)
                end if

                g = g_end + 1_int64
            end do

            ! Synchronize after all local kernels before MPI
            call hipCheck(hipDeviceSynchronize())

            if (n_remote > 1) then
                call tf_sort_remote_segments(remote_segs, n_remote)
            end if

            do remote_idx = 1, n_remote
                call wf_transverse_field_exchange_remote_segment( &
                    self, psi_dev, q, remote_segs(remote_idx), &
                    coeff_diag, coeff_offdiag, error_code)
                if (error_code /= 0) then
                    if (allocated(remote_segs)) deallocate (remote_segs)
                    return
                end if
            end do
        end do

        if (allocated(remote_segs)) deallocate (remote_segs)
    end subroutine wf_transverse_field_propagate_segmented

    subroutine wf_transverse_field_propagate(self, theta, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        real(real64), intent(in), dimension(1) :: theta
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: psi_dev
        complex(real64) :: coeff_diag, coeff_offdiag
        integer(int32) :: local_error

        error_code = 0
        local_error = 0

        if (.not. associated(self%context)) then
            error_code = 1
            return
        end if

        if (self%context%has_device) then
            psi_dev = c_loc(self%context%state(1))

            coeff_diag = cmplx(cos(theta(1) / 2.0_real64), 0.0_real64, real64)
            coeff_offdiag = cmplx(0.0_real64, -sin(theta(1) / 2.0_real64), real64)

            select case (self%layout_mode)
            case (TF_MODE_ALIGNED, TF_MODE_SEGMENTED)
                call wf_transverse_field_propagate_segmented(self, psi_dev, coeff_diag, coeff_offdiag, local_error)
            case default
                write (error_unit, '(A,I0)') &
                    'ERROR: transverse_field layout mode is unset: ', self%layout_mode
                local_error = 1
            end select

            call hipCheck(hipDeviceSynchronize())
        end if

        call wf_transverse_field_sync_error(self, local_error, error_code)
    end subroutine wf_transverse_field_propagate

    subroutine wf_transverse_field_destroy(self)
        class(transverse_field_propagator), intent(inout) :: self

        if (associated(self%context) .and. self%context%has_device) then
            call hipCheck(hipSetDevice(self%context%device_ID))
            call hipCheck(hipDeviceSynchronize())
        end if

        call wf_transverse_field_reset_state(self)
    end subroutine wf_transverse_field_destroy

end module wavefront_transverse_field
