module mpi_sparse

    use iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use iso_c_binding, only: c_f_pointer, c_ptr
    use mpi
    use mpi_backend, only: mpi_context
    use sparse, only: cleanup_graph_communications, csr, setup_graph_communications
    use chebyshev, only: chebyshev_multiply, estimate_spectral_radius
    use comm_info_module, only: quop_mpi_layout_t

    implicit none

    private

    public :: sparse_propagator

    type sparse_propagator

        type(mpi_context), pointer :: context => null()
        integer(int64), dimension(:), allocatable :: partition_table
        type(CSR) :: generator
        real(real64) :: spectral_radius

        ! Local allocatable arrays for CSR data (converted to 0-based)
        integer(int64), dimension(:), pointer :: row_starts_0based => null()
        integer(int64), dimension(:), pointer :: col_indexes_0based => null()
        complex(real64), dimension(:), pointer :: values_copy => null()

    contains

        procedure :: max_comm_size => mpi_sparse_max_comm_size
        procedure :: store_constraints => mpi_sparse_store_constraints
        procedure :: plan => mpi_sparse_plan
        procedure :: gen_operator => mpi_sparse_gen_operator
        procedure :: propagate => mpi_sparse_propagate
        procedure :: destroy => mpi_sparse_destroy

    end type sparse_propagator

contains

    subroutine mpi_sparse_max_comm_size(self, ci, error_code)
        !! The sparse propagator is compatible with any valid configuration
        !! so nothing to do here - ci remains unchanged.
        class(sparse_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code
        error_code = 0
    end subroutine mpi_sparse_max_comm_size

    subroutine mpi_sparse_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! No-op: sparse has no constraints.
        class(sparse_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine mpi_sparse_store_constraints

    subroutine mpi_sparse_plan(self, context, error_code)

        class(sparse_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code
        integer :: alloc_status
        integer(int32) :: ierr, local_error, synced_error
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_alloc_local

        error_code = 0

        self%context => context

        ci_subcomm = self%context%ci%get_SUBCOMM()
        ci_alloc_local = self%context%ci%get_alloc_local()

        local_error = 0

        if (.not. associated(self%context%work)) then
            allocate (self%context%work(ci_alloc_local), stat=alloc_status)
            if (alloc_status /= 0) then
                local_error = 1
            else
                self%context%work = cmplx(0.0_real64, 0.0_real64, real64)
            end if
        else if (size(self%context%work) < ci_alloc_local) then
            local_error = 1
        end if

        synced_error = local_error
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)
        error_code = synced_error

    end subroutine mpi_sparse_plan

    subroutine mpi_sparse_gen_operator(self, array_ptrs, array_sizes, error_code)

        class(sparse_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        ! Python-facing CSR inputs arrive with 1-based indexing.
        integer(int64), dimension(:), pointer :: local_row_starts
        integer(int64), dimension(:), pointer :: local_col_indexes
        complex(real64), dimension(:), pointer :: local_values

        integer(int32) :: i, n_local, nnz_local
        integer(int64) :: lb, ub
        integer(int32) :: n_arrays
        integer(int64) :: row_start_offset

        integer(int32) :: ierr, rank, flock
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_system_size

        error_code = 0

        n_arrays = size(array_sizes)

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
        ci_system_size = self%context%ci%get_system_size()
        call MPI_Comm_size(ci_subcomm, flock, ierr)
        call MPI_Comm_rank(ci_subcomm, rank, ierr)

        block
            integer(int64), pointer :: pt(:)
            pt => self%context%ci%get_partition_table()
            allocate (self%partition_table(size(pt)))
            self%partition_table(:) = pt(:)
        end block

        lb = self%partition_table(rank + 1)
        ub = self%partition_table(rank + 2) - 1
        n_local = int(ub - lb + 1)

        ! Convert to 0-based CSR expected by sparse kernels.
        nnz_local = int(local_row_starts(n_local + 1) - local_row_starts(1))
        row_start_offset = local_row_starts(1)

        allocate (self%row_starts_0based(n_local + 1))
        allocate (self%col_indexes_0based(nnz_local))

        do i = 1, n_local + 1
            self%row_starts_0based(i) = local_row_starts(i) - row_start_offset
        end do

        do i = 1, nnz_local
            self%col_indexes_0based(i) = local_col_indexes(i) - 1
        end do

        if (self%generator%has_values) then
            allocate (self%values_copy(nnz_local))
            do i = 1, nnz_local
                self%values_copy(i) = local_values(i)
            end do
        end if

        ! The sparse backend requires column indices within each row to be
        ! sorted in ascending order (binary search in spmv_cpu).
        call check_csr_sorted(self%row_starts_0based, self%col_indexes_0based, &
                              n_local, error_code)
        if (error_code /= 0) return

        self%generator%rows = int(ci_system_size)
        self%generator%columns = int(ci_system_size)
        self%generator%row_starts => self%row_starts_0based
        self%generator%col_indexes => self%col_indexes_0based

        if (self%generator%has_values) then
            self%generator%values => self%values_copy
        else
            nullify (self%generator%values)
        end if

        call setup_graph_communications(self%generator, self%partition_table, ci_subcomm)

        call estimate_spectral_radius(self%generator, &
                                      self%partition_table, &
                                      ci_subcomm, &
                                      self%spectral_radius)

    end subroutine mpi_sparse_gen_operator

    subroutine mpi_sparse_propagate(self, ts, error_code)

        class(sparse_propagator), intent(inout) :: self
        real(real64), intent(in) :: ts(:)
        integer(int32), intent(out) :: error_code
        real(real64) :: t

        error_code = 0

        t = ts(1)

        call chebyshev_multiply(self%generator, &
                                self%context%state, &
                                t, &
                                self%partition_table, &
                                self%context%work, &
                                self%context%ci%get_SUBCOMM(), &
                                self%spectral_radius)

        ! Copy the propagated state back into ctx%state.  We deliberately
        ! avoid pointer-swapping ctx%state and ctx%work: the CPython context
        ! wrapper attaches a Python-owned NumPy buffer to ctx%state via
        ! cw_attach_state and caches that pointer for zero-copy access from
        ! Python.  Swapping would silently rebind ctx%state to a Fortran-
        ! allocated buffer, so subsequent get_state/set_state calls (and
        ! ctx%destroy) would operate on the wrong memory.  The O(n) copy is
        ! negligible relative to the O(m_order * spmv) Chebyshev cost.
        self%context%state(:) = self%context%work(:)

    end subroutine mpi_sparse_propagate

    subroutine check_csr_sorted(row_starts, col_indexes, n_local, error_code)
        !! Validate that column indices within each CSR row are sorted in
        !! ascending order.  Sets error_code = 1 and writes a diagnostic
        !! to error_unit if the precondition is violated.
        !! row_starts and col_indexes are 0-based.
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
                        'mpi_sparse: CSR column indices in row ', row, &
                        ' are not sorted in ascending order.'
                    error_code = 1
                    return
                end if
            end do
        end do

    end subroutine check_csr_sorted

    subroutine mpi_sparse_destroy(self)

        class(sparse_propagator), intent(inout) :: self
        integer(int32) :: ierr

        if (allocated(self%partition_table)) then
            deallocate (self%partition_table)
        end if

        call cleanup_graph_communications(self%generator)

        if (associated(self%row_starts_0based)) then
            deallocate (self%row_starts_0based)
        end if
        if (associated(self%col_indexes_0based)) then
            deallocate (self%col_indexes_0based)
        end if
        if (associated(self%values_copy)) then
            deallocate (self%values_copy)
        end if

        self%generator%row_starts => null()
        self%generator%col_indexes => null()
        self%generator%values => null()

    end subroutine mpi_sparse_destroy

end module mpi_sparse
