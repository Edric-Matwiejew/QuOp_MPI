module mpi_transverse_field

    use, intrinsic :: iso_fortran_env, only: real64, int32, int64, error_unit
    use mpi
    use mpi_backend, only: mpi_context
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: transverse_field_propagator

    integer(int64), parameter :: COMPLEX128_BYTES = 16_int64
    integer(int64), parameter :: DEFAULT_CHUNK_BYTES = 67108864_int64

    type transverse_field_propagator

        type(mpi_context), pointer :: context => null()
        complex(real64), allocatable, dimension(:) :: recvbuf

        integer(int32) :: rank = 0
        integer(int32) :: comm_size = 0
        integer(int32) :: n_qubits = 0
        integer(int32) :: n_local_qubits = 0
        integer(int64) :: local_i = 0_int64
        integer(int64) :: chunk_elems = 0_int64

    contains

        procedure :: max_comm_size => mpi_transverse_field_max_comm_size
        procedure :: store_constraints => mpi_transverse_field_store_constraints
        procedure :: plan => mpi_transverse_field_plan
        procedure :: gen_operator => mpi_transverse_field_gen_operator
        procedure :: propagate => mpi_transverse_field_propagate
        procedure :: destroy => mpi_transverse_field_destroy

    end type transverse_field_propagator

contains

    pure logical function is_power_of_two_int64(value)
        integer(int64), intent(in) :: value
        integer(int64) :: tmp

        is_power_of_two_int64 = .false.
        if (value <= 0_int64) return

        tmp = value
        do while (mod(tmp, 2_int64) == 0_int64 .and. tmp > 1_int64)
            tmp = tmp / 2_int64
        end do

        is_power_of_two_int64 = (tmp == 1_int64)
    end function is_power_of_two_int64

    pure integer(int64) function largest_power_of_two_leq(limit)
        integer(int64), intent(in) :: limit

        largest_power_of_two_leq = 1_int64
        if (limit <= 1_int64) return

        do while (largest_power_of_two_leq <= limit / 2_int64)
            largest_power_of_two_leq = largest_power_of_two_leq * 2_int64
        end do
    end function largest_power_of_two_leq

    subroutine exact_log2_int64(value, exponent, is_exact)
        integer(int64), intent(in) :: value
        integer(int32), intent(out) :: exponent
        logical, intent(out) :: is_exact
        integer(int64) :: tmp

        exponent = 0
        is_exact = .false.
        if (value <= 0_int64) return

        tmp = value
        do while (mod(tmp, 2_int64) == 0_int64 .and. tmp > 1_int64)
            tmp = tmp / 2_int64
            exponent = exponent + 1_int32
        end do

        is_exact = (tmp == 1_int64)
    end subroutine exact_log2_int64

    subroutine mpi_transverse_field_reset_state(self)
        class(transverse_field_propagator), intent(inout) :: self

        if (allocated(self%recvbuf)) then
            deallocate (self%recvbuf)
        end if

        self%context => null()
        self%rank = 0
        self%comm_size = 0
        self%n_qubits = 0
        self%n_local_qubits = 0
        self%local_i = 0_int64
        self%chunk_elems = 0_int64
    end subroutine mpi_transverse_field_reset_state

    subroutine mpi_transverse_field_max_comm_size(self, ci, error_code)
        class(transverse_field_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int64) :: system_size, available_ranks, target_ranks

        error_code = 0

        system_size = ci%get_system_size()
        available_ranks = ci%get_n_processes()

        if (.not. is_power_of_two_int64(system_size)) then
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

        target_ranks = largest_power_of_two_leq(min(system_size, available_ranks))

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

            call exact_log2_int64(system_size, self%n_qubits, exact_power)
            if (.not. exact_power) then
                write (error_unit, '(A,I0)') &
                    "ERROR: transverse_field requires power-of-two system_size, got ", system_size
                local_error = 1
            end if

            if (local_error == 0) then
                call exact_log2_int64(local_i, self%n_local_qubits, exact_power)
                if (.not. exact_power) then
                    write (error_unit, '(A,I0)') &
                        "ERROR: transverse_field requires power-of-two local_i, got ", local_i
                    local_error = 1
                end if
            end if

            if (local_error == 0 .and. .not. is_power_of_two_int64(int(self%comm_size, int64))) then
                write (error_unit, '(A,I0)') &
                    "ERROR: transverse_field requires power-of-two communicator size, got ", self%comm_size
                local_error = 1
            end if

            if (local_error == 0 .and. local_i * int(self%comm_size, int64) /= system_size) then
                write (error_unit, '(A,I0,A,I0,A,I0)') &
                    "ERROR: transverse_field requires equal block distribution: local_i=", local_i, &
                    ", comm_size=", self%comm_size, ", system_size=", system_size
                local_error = 1
            end if

            if (local_error == 0 .and. local_i_offset /= int(self%rank, int64) * local_i) then
                write (error_unit, '(A,I0,A,I0,A,I0)') &
                    "ERROR: transverse_field requires aligned local offsets: rank=", self%rank, &
                    ", local_i_offset=", local_i_offset, ", local_i=", local_i
                local_error = 1
            end if

            if (local_error == 0 .and. alloc_local < local_i) then
                write (error_unit, '(A,I0,A,I0)') &
                    "ERROR: transverse_field requires alloc_local >= local_i, got alloc_local=", &
                    alloc_local, ", local_i=", local_i
                local_error = 1
            end if

            if (local_error == 0) then
                self%local_i = local_i
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
        complex(real64) :: u, v, coeff_diag, coeff_offdiag
        integer(int64) :: stride, base, j, off, m, local_i
        integer(int32) :: q, peer_mask, peer, count, ierr
        integer(int32) :: ci_subcomm
        integer(int32) :: status(MPI_STATUS_SIZE)

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
    end subroutine mpi_transverse_field_propagate

    subroutine mpi_transverse_field_destroy(self)
        class(transverse_field_propagator), intent(inout) :: self

        call mpi_transverse_field_reset_state(self)
    end subroutine mpi_transverse_field_destroy

end module mpi_transverse_field
