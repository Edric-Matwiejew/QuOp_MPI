module mpi_transverse_field

    use, intrinsic :: iso_fortran_env, only: real64, int32, int64, error_unit
    use mpi
    use mpi_backend, only: mpi_context
    use comm_info_module, only: quop_mpi_layout_t
    use transverse_field_common

    implicit none

    private

    public :: transverse_field_propagator

    type transverse_field_propagator

        type(mpi_context), pointer :: context => null()
        complex(real64), allocatable, dimension(:) :: recvbuf
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

        procedure :: max_comm_size => mpi_transverse_field_max_comm_size
        procedure :: store_constraints => mpi_transverse_field_store_constraints
        procedure :: plan => mpi_transverse_field_plan
        procedure :: gen_operator => mpi_transverse_field_gen_operator
        procedure :: propagate => mpi_transverse_field_propagate
        procedure :: destroy => mpi_transverse_field_destroy

    end type transverse_field_propagator

contains

    subroutine mpi_transverse_field_apply_local_pair_segment( &
        psi, lb_global, g0, g1, delta, coeff_diag, coeff_offdiag)
        complex(real64), intent(inout) :: psi(:)
        integer(int64), intent(in) :: lb_global, g0, g1, delta
        complex(real64), intent(in) :: coeff_diag, coeff_offdiag

        integer(int64) :: g, local_u, local_v
        complex(real64) :: u, v

        do g = g0, g1
            local_u = g - lb_global + 1_int64
            local_v = g + delta - lb_global + 1_int64

            u = psi(local_u)
            v = psi(local_v)

            psi(local_u) = coeff_diag * u + coeff_offdiag * v
            psi(local_v) = coeff_offdiag * u + coeff_diag * v
        end do
    end subroutine mpi_transverse_field_apply_local_pair_segment

    subroutine mpi_transverse_field_exchange_remote_segment( &
        self, psi, q, seg, coeff_diag, coeff_offdiag, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        complex(real64), intent(inout) :: psi(:)
        integer(int32), intent(in) :: q
        type(tf_segment_t), intent(in) :: seg
        complex(real64), intent(in) :: coeff_diag, coeff_offdiag
        integer(int32), intent(out) :: error_code

        integer(int64) :: chunk_g0, chunk_g1, m, local0, j
        integer(int32) :: count, ierr
        integer(int32) :: ci_subcomm
        integer(int32) :: status(MPI_STATUS_SIZE)

        error_code = 0
        ci_subcomm = self%context%ci%get_SUBCOMM()

        chunk_g0 = seg%g0
        do while (chunk_g0 <= seg%g1)
            chunk_g1 = min(seg%g1, chunk_g0 + self%chunk_elems - 1_int64)
            m = chunk_g1 - chunk_g0 + 1_int64
            count = int(m, int32)

            local0 = chunk_g0 - self%lb_global + 1_int64

            call MPI_Sendrecv(psi(local0:local0 + m - 1_int64), count, MPI_DOUBLE_COMPLEX, seg%owner, q, &
                              self%recvbuf(1:count), count, MPI_DOUBLE_COMPLEX, seg%owner, q, &
                              ci_subcomm, status, ierr)

            if (ierr /= MPI_SUCCESS) then
                error_code = 1
                return
            end if

            do j = 1_int64, m
                psi(local0 + j - 1_int64) = coeff_diag * psi(local0 + j - 1_int64) + &
                                            coeff_offdiag * self%recvbuf(j)
            end do

            chunk_g0 = chunk_g1 + 1_int64
        end do
    end subroutine mpi_transverse_field_exchange_remote_segment

    subroutine mpi_transverse_field_propagate_segmented(self, psi, coeff_diag, coeff_offdiag, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        complex(real64), intent(inout) :: psi(:)
        complex(real64), intent(in) :: coeff_diag, coeff_offdiag
        integer(int32), intent(out) :: error_code

        integer(int32) :: q, owner, n_remote, remote_idx, remote_cap
        integer(int64) :: bit_mask, g, gp, g_end, delta
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
                    call mpi_transverse_field_apply_local_pair_segment( &
                        psi, self%lb_global, g, g_end, delta, coeff_diag, coeff_offdiag)
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

            if (n_remote > 1) then
                call tf_sort_remote_segments(remote_segs, n_remote)
            end if

            do remote_idx = 1, n_remote
                call mpi_transverse_field_exchange_remote_segment( &
                    self, psi, q, remote_segs(remote_idx), &
                    coeff_diag, coeff_offdiag, error_code)
                if (error_code /= 0) then
                    if (allocated(remote_segs)) deallocate (remote_segs)
                    return
                end if
            end do
        end do

        if (allocated(remote_segs)) deallocate (remote_segs)
    end subroutine mpi_transverse_field_propagate_segmented

    subroutine mpi_transverse_field_reset_state(self)
        class(transverse_field_propagator), intent(inout) :: self

        if (allocated(self%recvbuf)) then
            deallocate (self%recvbuf)
        end if
        if (allocated(self%partition_table_0)) then
            deallocate (self%partition_table_0)
        end if

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
    end subroutine mpi_transverse_field_reset_state

    subroutine mpi_transverse_field_max_comm_size(self, ci, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int64) :: system_size, available_ranks, target_ranks

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
    end subroutine mpi_transverse_field_max_comm_size

    subroutine mpi_transverse_field_store_constraints(self, constraint_ptrs, constraint_sizes)
        class(transverse_field_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine mpi_transverse_field_store_constraints

    subroutine mpi_transverse_field_plan(self, context, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, alloc_status, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: desired_chunk

        error_code = 0
        local_error = 0

        call mpi_transverse_field_reset_state(self)
        self%context => context

        ci_subcomm = self%context%ci%get_SUBCOMM()
        self%local_i = self%context%ci%get_local_i()

        if (self%local_i <= 0_int64) then
            local_error = 1
        end if

        if (local_error == 0) then
            call MPI_Comm_rank(ci_subcomm, self%rank, ierr)
            call MPI_Comm_size(ci_subcomm, self%comm_size, ierr)

            desired_chunk = DEFAULT_CHUNK_BYTES / COMPLEX128_BYTES
            self%chunk_elems = min(self%local_i, desired_chunk)
            if (self%chunk_elems < 1_int64) self%chunk_elems = 1_int64

            allocate (self%recvbuf(self%chunk_elems), stat=alloc_status)
            if (alloc_status /= 0) then
                local_error = 1
            end if
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, ci_subcomm, ierr)

        error_code = synced_error
        if (synced_error /= 0) then
            call mpi_transverse_field_reset_state(self)
        end if
    end subroutine mpi_transverse_field_plan

    subroutine mpi_transverse_field_gen_operator(self, array_ptrs, array_sizes, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, local_error, synced_error
        integer(int64) :: system_size, local_i, local_i_offset, alloc_local
        integer(int64), pointer :: pt(:)
        logical :: exact_power

        error_code = 0
        local_error = 0

        if (.not. associated(self%context)) then
            local_error = 1
        else
            system_size = self%context%ci%get_system_size()
            local_i = self%context%ci%get_local_i()
            local_i_offset = self%context%ci%get_local_i_offset()
            alloc_local = self%context%ci%get_alloc_local()

            call tf_exact_log2(system_size, self%n_qubits, exact_power)
            if (.not. exact_power) then
                write (error_unit, '(A,I0)') &
                    "ERROR: transverse_field requires power-of-two system_size, got ", system_size
                local_error = 1
            end if

            if (local_error == 0) then
                self%local_i = local_i

                pt => self%context%ci%get_partition_table()
                if (.not. associated(pt)) then
                    write (error_unit, '(A)') &
                        'ERROR: transverse_field requires an allocated partition_table.'
                    local_error = 1
                else
                    call tf_copy_partition_table_0(pt, self%comm_size, self%rank, &
                                                  self%partition_table_0, self%lb_global, &
                                                  self%ub_global, local_error)
                end if
            end if

            if (local_error == 0 .and. alloc_local < local_i) then
                write (error_unit, '(A,I0,A,I0)') &
                    "ERROR: transverse_field requires alloc_local >= local_i, got alloc_local=", &
                    alloc_local, ", local_i=", local_i
                local_error = 1
            end if

            if (local_error == 0) then
                call tf_classify_layout(system_size, local_i, local_i_offset, &
                                       self%comm_size, self%rank, self%layout_mode, &
                                       self%n_local_qubits, local_error)
            end if
        end if

        synced_error = local_error
        if (associated(self%context)) then
            call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                               self%context%ci%get_SUBCOMM(), ierr)
        end if

        error_code = synced_error
    end subroutine mpi_transverse_field_gen_operator

    subroutine mpi_transverse_field_propagate(self, theta, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        real(real64), intent(in), dimension(1) :: theta
        integer(int32), intent(out) :: error_code

        complex(real64), dimension(:), pointer :: psi
        complex(real64) :: coeff_diag, coeff_offdiag
        integer(int64) :: local_i

        error_code = 0

        if (.not. associated(self%context)) then
            error_code = 1
            return
        end if

        if (.not. allocated(self%recvbuf)) then
            error_code = 1
            return
        end if

        if (self%n_qubits < 0 .or. self%n_local_qubits < 0) then
            error_code = 1
            return
        end if

        psi => self%context%state
        if (.not. associated(psi)) then
            error_code = 1
            return
        end if

        local_i = self%local_i
        if (size(psi) < local_i) then
            error_code = 1
            return
        end if

        coeff_diag = cmplx(cos(theta(1) / 2.0_real64), 0.0_real64, real64)
        coeff_offdiag = cmplx(0.0_real64, -sin(theta(1) / 2.0_real64), real64)

        select case (self%layout_mode)
        case (TF_MODE_ALIGNED)
            call mpi_transverse_field_propagate_aligned(self, psi, local_i, coeff_diag, coeff_offdiag, error_code)
        case (TF_MODE_SEGMENTED)
            call mpi_transverse_field_propagate_segmented(self, psi, coeff_diag, coeff_offdiag, error_code)
        case default
            write (error_unit, '(A,I0)') &
                'ERROR: transverse_field layout mode is not yet implemented: ', self%layout_mode
            error_code = 1
        end select
    end subroutine mpi_transverse_field_propagate

    subroutine mpi_transverse_field_propagate_aligned(self, psi, local_i, coeff_diag, coeff_offdiag, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        complex(real64), intent(inout), dimension(:) :: psi
        integer(int64), intent(in) :: local_i
        complex(real64), intent(in) :: coeff_diag, coeff_offdiag
        integer(int32), intent(out) :: error_code

        complex(real64) :: u, v
        integer(int64) :: stride, base, j, off, m
        integer(int32) :: q, peer_mask, peer, count, ierr
        integer(int32) :: ci_subcomm
        integer(int32) :: status(MPI_STATUS_SIZE)

        ci_subcomm = self%context%ci%get_SUBCOMM()

        do q = 0, self%n_qubits - 1
            if (q < self%n_local_qubits) then
                stride = ishft(1_int64, q)
                do base = 0_int64, local_i - 1_int64, 2_int64 * stride
                    do j = 0_int64, stride - 1_int64
                        u = psi(base + j + 1_int64)
                        v = psi(base + j + stride + 1_int64)
                        psi(base + j + 1_int64) = coeff_diag * u + coeff_offdiag * v
                        psi(base + j + stride + 1_int64) = coeff_offdiag * u + coeff_diag * v
                    end do
                end do
            else
                peer_mask = ishft(1_int32, q - self%n_local_qubits)
                peer = ieor(self%rank, peer_mask)

                do off = 0_int64, local_i - 1_int64, self%chunk_elems
                    m = min(self%chunk_elems, local_i - off)
                    count = int(m, int32)

                    call MPI_Sendrecv(psi(off + 1_int64:off + m), count, MPI_DOUBLE_COMPLEX, peer, q, &
                                      self%recvbuf(1:count), count, MPI_DOUBLE_COMPLEX, peer, q, &
                                      ci_subcomm, status, ierr)

                    if (ierr /= MPI_SUCCESS) then
                        error_code = 1
                        return
                    end if

                    do j = 1_int64, m
                        psi(off + j) = coeff_diag * psi(off + j) + coeff_offdiag * self%recvbuf(j)
                    end do
                end do
            end if
        end do
    end subroutine mpi_transverse_field_propagate_aligned

    subroutine mpi_transverse_field_destroy(self)
        class(transverse_field_propagator), intent(inout) :: self

        call mpi_transverse_field_reset_state(self)
    end subroutine mpi_transverse_field_destroy

end module mpi_transverse_field
