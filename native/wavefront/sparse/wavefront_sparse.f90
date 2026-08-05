!------------------------------------------------------------------------------
!> @brief Wavefront (HIP/GPU) sparse propagator using Chebyshev time evolution.
!>
!> @details This module implements a sparse matrix exponential propagator for
!> GPU-accelerated quantum simulation. It uses the shared sparse Chebyshev
!> implementation from sparse_propagators with HIP enabled.
!>
!> Key differences from mpi_sparse:
!> - Uses wavefront_context (GPU state distributed over DEVCOMM)
!> - CSR data is redistributed from NODECOMM to DEVCOMM distribution
!> - Graph communicator is built on DEVCOMM (multi-node GPU communicator)
!> - Transfers CSR to device after redistribution
!> - State lives on GPU, handled by wavefront_context
!------------------------------------------------------------------------------
module wavefront_sparse

    use iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use iso_c_binding, only: c_f_pointer, c_ptr
    use mpi
    use hipfort
    use hipfort_check
    use wavefront, only: wavefront_context
    use sparse, only: cleanup_graph_communications, csr, csr_to_device, setup_graph_communications
    use chebyshev, only: chebyshev_multiply, estimate_spectral_radius
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: sparse_propagator

    type sparse_propagator

        type(wavefront_context), pointer :: context => null()
        integer(int64), dimension(:), allocatable :: partition_table
        type(CSR) :: generator
        real(real64) :: spectral_radius

        ! Local 0-based CSR arrays (after redistribution to DEVCOMM layout)
        integer(int64), dimension(:), pointer :: row_starts_0based => null()
        integer(int64), dimension(:), pointer :: col_indexes_0based => null()
        complex(real64), dimension(:), pointer :: values_copy => null()

        ! Redistribution counts/displacements for CSR arrays
        ! From NODECOMM (SUBCOMM local) to DEVCOMM_NODE (GPU balanced)
        integer(int64), dimension(:), allocatable :: row_counts_send
        integer(int64), dimension(:), allocatable :: row_displs_send
        integer(int64), dimension(:), allocatable :: row_counts_recv
        integer(int64), dimension(:), allocatable :: row_displs_recv

    contains

        procedure :: max_comm_size => wavefront_sparse_max_comm_size
        procedure :: store_constraints => wavefront_sparse_store_constraints
        procedure :: plan => wavefront_sparse_plan
        procedure :: gen_operator => wavefront_sparse_gen_operator
        procedure :: propagate => wavefront_sparse_propagate
        procedure :: destroy => wavefront_sparse_destroy

    end type sparse_propagator

contains

    subroutine wavefront_sparse_max_comm_size(self, ci, error_code)
        !! The sparse propagator is compatible with any valid configuration
        !! so nothing to do here - ci remains unchanged.
        class(sparse_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr

        error_code = 0

        call ci%set_requires_device_work_buffer(.true., error_code)
        if (error_code /= 0) return

        call MPI_Barrier(ci%get_SUBCOMM(), ierr)

    end subroutine wavefront_sparse_max_comm_size

    subroutine wavefront_sparse_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! No-op: sparse has no constraints.
        class(sparse_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine wavefront_sparse_store_constraints

    subroutine wavefront_sparse_plan(self, context, error_code)

        class(sparse_propagator), intent(inout) :: self
        type(wavefront_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        error_code = 0

        self%context => context

    end subroutine wavefront_sparse_plan

    subroutine wavefront_sparse_gen_operator(self, array_ptrs, array_sizes, error_code)

        class(sparse_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        ! Original inputs from Python (1-based data, distributed over SUBCOMM)
        integer(int64), dimension(:), pointer :: local_row_starts
        integer(int64), dimension(:), pointer :: local_col_indexes
        complex(real64), dimension(:), pointer :: local_values

        integer(int32) :: i, j, n_arrays
        integer(int32) :: n_local_subcomm, nnz_local_subcomm
        integer(int32) :: n_local_dev
        integer(int32) :: NODECOMM_rank, NODECOMM_size
        integer(int64) :: row_start_offset
        integer(int64) :: dev_nnz_total

        integer(int32) :: ierr, dev_flock
        integer(int32) :: ci_subcomm, ci_nodecomm, ci_devcomm, ci_devcomm_node
        integer(int64) :: ci_local_i, ci_device_local_i, ci_system_size

        ! Temporary arrays for redistribution
        integer(int64), dimension(:), allocatable :: nnz_per_row_local
        integer(int64), dimension(:), allocatable :: nnz_per_row_dev
        integer(int64), dimension(:), allocatable :: col_counts_send, col_displs_send
        integer(int64), dimension(:), allocatable :: col_counts_recv, col_displs_recv

        error_code = 0

        ! Determine if values are provided (3 arrays) or implicit ones (2 arrays)
        n_arrays = size(array_sizes)

        ! Map array pointers to original inputs
        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, local_row_starts, [array_sizes(1)])
        array_ptr = transfer(array_ptrs(2), array_ptr)
        call c_f_pointer(array_ptr, local_col_indexes, [array_sizes(2)])

        if (n_arrays >= 3) then
            array_ptr = transfer(array_ptrs(3), array_ptr)
            call c_f_pointer(array_ptr, local_values, [array_sizes(3)])
            self%generator%has_values = .true.
        else
            nullify (local_values)
            self%generator%has_values = .false.
        end if

        ci_subcomm = self%context%ci%get_SUBCOMM()
        ci_nodecomm = self%context%ci%get_NODECOMM()
        ci_devcomm = self%context%ci%get_DEVCOMM()
        ci_devcomm_node = self%context%ci%get_DEVCOMM_NODE()
        ci_local_i = self%context%ci%get_local_i()
        ci_device_local_i = self%context%ci%get_device_local_i()
        ci_system_size = self%context%ci%get_system_size()

        ! Get communicator info
        call MPI_Comm_rank(ci_nodecomm, NODECOMM_rank, ierr)
        call MPI_Comm_size(ci_nodecomm, NODECOMM_size, ierr)

        if (ci_devcomm /= MPI_COMM_NULL) then
            call MPI_Comm_size(ci_devcomm, dev_flock, ierr)
        else
            dev_flock = 0
        end if

        ! Number of local rows from SUBCOMM distribution
        n_local_subcomm = int(ci_local_i, int32)

        ! row_starts from Python has size n_local_subcomm + 1
        row_start_offset = local_row_starts(1) ! The global 1-based offset
        nnz_local_subcomm = int(local_row_starts(n_local_subcomm + 1) - row_start_offset)

        ! Compute nnz per row for redistribution
        allocate (nnz_per_row_local(n_local_subcomm))
        do i = 1, n_local_subcomm
            nnz_per_row_local(i) = local_row_starts(i + 1) - local_row_starts(i)
        end do

        ! Number of local rows for DEVCOMM distribution (on this GPU)
        n_local_dev = int(ci_device_local_i, int32)

        ! Per-rank overlap redistribution NODECOMM(host) -> DEVCOMM_NODE(device).
        ! Indexed by NODECOMM rank (not DEVCOMM_NODE rank) because GPU ranks may
        ! be non-contiguous in NODECOMM (e.g. Setonix NUMA: positions {0,8}).
        ! Uses NODECOMM_counts/displs and DEVCOMM_NODE_counts/displs from context_setup.

        allocate (self%row_counts_send(NODECOMM_size))
        allocate (self%row_displs_send(NODECOMM_size))
        allocate (self%row_counts_recv(NODECOMM_size))
        allocate (self%row_displs_recv(NODECOMM_size))

        block
            integer(int64) :: my_host_start, my_host_end
            integer(int64) :: my_dev_start, my_dev_end
            integer(int64) :: partner_dev_start, partner_dev_end
            integer(int64) :: partner_host_start, partner_host_end
            integer(int64) :: overlap_start, overlap_end

            my_host_start = self%context%NODECOMM_displs(NODECOMM_rank + 1)
            my_host_end = my_host_start + self%context%NODECOMM_counts(NODECOMM_rank + 1)

            my_dev_start = self%context%DEVCOMM_NODE_displs(NODECOMM_rank + 1)
            my_dev_end = my_dev_start + self%context%DEVCOMM_NODE_counts(NODECOMM_rank + 1)

            ! Send counts: overlap of MY host range with each rank's device range
            do i = 1, NODECOMM_size
                partner_dev_start = self%context%DEVCOMM_NODE_displs(i)
                partner_dev_end = partner_dev_start + self%context%DEVCOMM_NODE_counts(i)
                overlap_start = max(my_host_start, partner_dev_start)
                overlap_end = min(my_host_end, partner_dev_end)
                if (overlap_end > overlap_start) then
                    self%row_counts_send(i) = overlap_end - overlap_start
                    self%row_displs_send(i) = overlap_start - my_host_start
                else
                    self%row_counts_send(i) = 0
                    self%row_displs_send(i) = 0
                end if
            end do

            ! Recv counts: overlap of each rank's host range with MY device range
            do i = 1, NODECOMM_size
                partner_host_start = self%context%NODECOMM_displs(i)
                partner_host_end = partner_host_start + self%context%NODECOMM_counts(i)
                overlap_start = max(my_dev_start, partner_host_start)
                overlap_end = min(my_dev_end, partner_host_end)
                if (overlap_end > overlap_start) then
                    self%row_counts_recv(i) = overlap_end - overlap_start
                    self%row_displs_recv(i) = overlap_start - my_dev_start
                else
                    self%row_counts_recv(i) = 0
                    self%row_displs_recv(i) = 0
                end if
            end do
        end block

        ! Redistribute nnz_per_row to get device layout
        if (n_local_dev > 0) then
            allocate (nnz_per_row_dev(n_local_dev))
        else
            allocate (nnz_per_row_dev(1))
        end if

        call MPI_Alltoallv(nnz_per_row_local, &
                           int(self%row_counts_send), &
                           int(self%row_displs_send), &
                           MPI_INTEGER8, &
                           nnz_per_row_dev, &
                           int(self%row_counts_recv), &
                           int(self%row_displs_recv), &
                           MPI_INTEGER8, &
                           ci_nodecomm, &
                           ierr)

        ! Compute total nnz for device partition
        dev_nnz_total = 0
        do i = 1, n_local_dev
            dev_nnz_total = dev_nnz_total + nnz_per_row_dev(i)
        end do

        ! Build col_indexes redistribution counts from nnz_per_row
        allocate (col_counts_send(NODECOMM_size))
        allocate (col_displs_send(NODECOMM_size))
        allocate (col_counts_recv(NODECOMM_size))
        allocate (col_displs_recv(NODECOMM_size))

        ! Compute send counts: for each destination, sum nnz of rows going there
        col_counts_send = 0
        do i = 1, NODECOMM_size
            ! Rows going to rank i are from displs_send(i)+1 to displs_send(i)+counts_send(i)
            if (self%row_counts_send(i) > 0) then
                do j = 1, int(self%row_counts_send(i))
                    col_counts_send(i) = col_counts_send(i) + &
                                         nnz_per_row_local(int(self%row_displs_send(i)) + j)
                end do
            end if
        end do

        ! Compute send displacements
        col_displs_send(1) = 0
        do i = 2, NODECOMM_size
            col_displs_send(i) = col_displs_send(i - 1) + col_counts_send(i - 1)
        end do

        ! Exchange to get recv counts
        call MPI_Alltoall(col_counts_send, 1, MPI_INTEGER8, &
                          col_counts_recv, 1, MPI_INTEGER8, &
                          ci_nodecomm, ierr)

        ! Compute recv displacements
        col_displs_recv(1) = 0
        do i = 2, NODECOMM_size
            col_displs_recv(i) = col_displs_recv(i - 1) + col_counts_recv(i - 1)
        end do

        ! Allocate the persistent CSR storage directly so the redistribution
        ! Alltoallv writes straight into the long-lived buffers (no temporary
        ! dev_* copies).  At N > 2^31 this saves ~32 bytes/nnz of peak
        ! gen_operator overhead.
        if (n_local_dev > 0) then
            allocate (self%row_starts_0based(n_local_dev + 1))
            allocate (self%col_indexes_0based(dev_nnz_total))
            if (self%generator%has_values) then
                allocate (self%values_copy(dev_nnz_total))
            end if
        else
            allocate (self%row_starts_0based(1))
            self%row_starts_0based(1) = 0
            allocate (self%col_indexes_0based(1))
            self%col_indexes_0based(1) = 0
            if (self%generator%has_values) then
                allocate (self%values_copy(1))
            end if
        end if

        ! Redistribute col_indexes (Python pre-normalizes to 0-based via
        ! _normalize_sparse_csr_operator_args; pass through unchanged).
        call MPI_Alltoallv(local_col_indexes, &
                           int(col_counts_send), &
                           int(col_displs_send), &
                           MPI_INTEGER8, &
                           self%col_indexes_0based, &
                           int(col_counts_recv), &
                           int(col_displs_recv), &
                           MPI_INTEGER8, &
                           ci_nodecomm, &
                           ierr)

        ! Redistribute values if present
        if (self%generator%has_values) then
            call MPI_Alltoallv(local_values, &
                               int(col_counts_send), &
                               int(col_displs_send), &
                               MPI_DOUBLE_COMPLEX, &
                               self%values_copy, &
                               int(col_counts_recv), &
                               int(col_displs_recv), &
                               MPI_DOUBLE_COMPLEX, &
                               ci_nodecomm, &
                               ierr)
        end if

        ! Build row_starts from nnz_per_row_dev (0-based offsets) directly
        ! into the persistent buffer.
        if (n_local_dev > 0) then
            self%row_starts_0based(1) = 0
            do i = 2, n_local_dev + 1
                self%row_starts_0based(i) = &
                    self%row_starts_0based(i - 1) + nnz_per_row_dev(i - 1)
            end do
        end if

        ! Build partition table over DEVCOMM
        if (ci_devcomm /= MPI_COMM_NULL) then
            block
                integer(int32), allocatable :: local_sizes(:)
                allocate (local_sizes(dev_flock))
                call MPI_Allgather(n_local_dev, &
                                   1, &
                                   MPI_INTEGER, &
                                   local_sizes, &
                                   1, &
                                   MPI_INTEGER, &
                                   ci_devcomm, &
                                   ierr)
                allocate (self%partition_table(dev_flock + 1))
                self%partition_table(1) = 1_int64
                do i = 2, dev_flock + 1
                    self%partition_table(i) = self%partition_table(i - 1) + int(local_sizes(i - 1), int64)
                end do
                deallocate (local_sizes)
            end block
        else
            allocate (self%partition_table(1))
            self%partition_table(1) = 1_int64
        end if

        ! Set up CSR structure
        self%generator%rows = int(ci_system_size, int32)
        self%generator%columns = int(ci_system_size, int32)
        self%generator%row_starts => self%row_starts_0based
        self%generator%col_indexes => self%col_indexes_0based

        if (self%generator%has_values) then
            self%generator%values => self%values_copy
        else
            nullify (self%generator%values)
        end if

        ! Validate the CSR sort precondition required by halo metadata and
        ! GPU SpMV.  Alltoallv preserves intra-row ordering, but only if the
        ! sender's rows arrive contiguously to a single destination; if a row
        ! were ever split across recv blocks, ordering would break.  Catch
        ! that here rather than in the kernel.
        if (self%context%has_device .and. n_local_dev > 0) then
            call check_csr_sorted(self%row_starts_0based, &
                                  self%col_indexes_0based, &
                                  n_local_dev, error_code)
            if (error_code /= 0) return
        end if

        ! Setup graph communicator on DEVCOMM (GPU processes only)
        if (self%context%has_device) then
            call setup_graph_communications(self%generator, self%partition_table, &
                                            ci_devcomm)

            ! Transfer CSR to device
            call csr_to_device(self%generator)

            ! Estimate spectral radius
            call estimate_spectral_radius(self%generator, &
                                          self%partition_table, &
                                          ci_devcomm, &
                                          self%spectral_radius)
        else
            self%spectral_radius = 0.0_real64
        end if

        ! Broadcast spectral radius to all ranks
        if (ci_devcomm_node /= MPI_COMM_NULL) then
            call MPI_Bcast(self%spectral_radius, 1, MPI_DOUBLE, 0, &
                           ci_devcomm_node, ierr)
        end if
        call MPI_Bcast(self%spectral_radius, 1, MPI_DOUBLE, 0, &
                       ci_nodecomm, ierr)

        ! Cleanup temporary arrays
        deallocate (nnz_per_row_local)
        deallocate (nnz_per_row_dev)
        deallocate (col_counts_send, col_displs_send)
        deallocate (col_counts_recv, col_displs_recv)

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine wavefront_sparse_gen_operator

    subroutine wavefront_sparse_propagate(self, ts, error_code)

        class(sparse_propagator), intent(inout) :: self
        real(real64), intent(in) :: ts(:)
        integer(int32), intent(out) :: error_code
        real(real64) :: t
        complex(real64), dimension(:), pointer :: ptr_tmp
        integer(int32) :: ierr
        integer(int32) :: ci_subcomm, ci_devcomm

        error_code = 0

        t = ts(1)
        ci_subcomm = self%context%ci%get_SUBCOMM()
        ci_devcomm = self%context%ci%get_DEVCOMM()

        if (self%context%has_device) then
            ! State is on GPU (context%state), call chebyshev_multiply
            ! The GPU implementation uses device pointers directly
            call chebyshev_multiply(self%generator, &
                                    self%context%state, &
                                    t, &
                                    self%partition_table, &
                                    self%context%work, &
                                    ci_devcomm, &
                                    self%spectral_radius)

            ! Swap state and work pointers
            ptr_tmp => self%context%state
            self%context%state => self%context%work
            self%context%work => ptr_tmp
        end if

        call MPI_Barrier(ci_subcomm, ierr)

    end subroutine wavefront_sparse_propagate

    subroutine wavefront_sparse_destroy(self)

        class(sparse_propagator), intent(inout) :: self

        if (allocated(self%partition_table)) then
            deallocate (self%partition_table)
        end if

        ! Free graph communicator resources (also frees device CSR memory if allocated)
        call cleanup_graph_communications(self%generator)

        ! Deallocate local copies
        if (associated(self%row_starts_0based)) then
            deallocate (self%row_starts_0based)
        end if
        if (associated(self%col_indexes_0based)) then
            deallocate (self%col_indexes_0based)
        end if
        if (associated(self%values_copy)) then
            deallocate (self%values_copy)
        end if

        if (allocated(self%row_counts_send)) deallocate (self%row_counts_send)
        if (allocated(self%row_displs_send)) deallocate (self%row_displs_send)
        if (allocated(self%row_counts_recv)) deallocate (self%row_counts_recv)
        if (allocated(self%row_displs_recv)) deallocate (self%row_displs_recv)

        self%generator%row_starts => null()
        self%generator%col_indexes => null()
        self%generator%values => null()

    end subroutine wavefront_sparse_destroy

    subroutine check_csr_sorted(row_starts, col_indexes, n_local, error_code)
        !! Validate that column indices within each CSR row are sorted in
        !! ascending order (precondition for halo metadata and GPU SpMV).
        !! Mirrors mpi_sparse: row_starts and col_indexes are 0-based.
        integer(int64), intent(in) :: row_starts(:)
        integer(int64), intent(in) :: col_indexes(:)
        integer(int32), intent(in) :: n_local
        integer(int32), intent(out) :: error_code

        integer(int32) :: row, j
        integer(int64) :: lo, hi

        error_code = 0

        do row = 1, n_local
            lo = row_starts(row) + 1
            hi = row_starts(row + 1)
            do j = int(lo) + 1, int(hi)
                if (col_indexes(j) < col_indexes(j - 1)) then
                    write (error_unit, '(A,I0,A)') &
                        'wavefront_sparse: CSR column indices in row ', row, &
                        ' are not sorted in ascending order after redistribution.'
                    error_code = 1
                    return
                end if
            end do
        end do

    end subroutine check_csr_sorted

end module wavefront_sparse
