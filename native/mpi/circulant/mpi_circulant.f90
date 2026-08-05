module mpi_circulant

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use, intrinsic :: iso_c_binding
    use mpi
    use mpi_circulant_operators, only: graph_eigenvalues
    use sparse_vector, only: to_sparse_vector
    use mpi_backend, only: mpi_context
    use comm_info_module, only: quop_mpi_layout_t
    use fftw_mpi_init_guard, only: ensure_fftw_mpi_init

    implicit none

    private

    public :: circulant_propagator

    include 'fftw3-mpi.f03'

    type circulant_propagator

        type(mpi_context), pointer :: context => null()
        type(c_ptr) :: fftw_plan_forward = c_null_ptr
        type(c_ptr) :: fftw_plan_backward = c_null_ptr
        logical :: planned = .false.
        real(real64), dimension(:), allocatable :: eigenvalues

        integer(C_INTPTR_T) :: local_o = 0
        integer(C_INTPTR_T) :: local_o_offset = 0

    contains

        procedure :: max_comm_size => mpi_circulant_max_comm_size
        procedure :: store_constraints => mpi_circulant_store_constraints
        procedure :: plan => mpi_circulant_plan
        procedure :: gen_operator => mpi_circulant_gen_operator
        procedure :: propagate => mpi_circulant_propagate
        procedure :: destroy => mpi_circulant_destroy

    end type circulant_propagator

contains

    subroutine mpi_circulant_release_plans(self)
        class(circulant_propagator), intent(inout) :: self

        ! Clear the lifecycle flag first so repeated destroy/replan calls
        ! stay safe even if cleanup is re-entered.
        self%planned = .false.

        if (c_associated(self%fftw_plan_backward)) then
            call fftw_destroy_plan(self%fftw_plan_backward)
        end if
        self%fftw_plan_backward = c_null_ptr

        if (c_associated(self%fftw_plan_forward)) then
            call fftw_destroy_plan(self%fftw_plan_forward)
        end if
        self%fftw_plan_forward = c_null_ptr
    end subroutine mpi_circulant_release_plans

    subroutine mpi_circulant_reset_state(self)
        class(circulant_propagator), intent(inout) :: self

        call mpi_circulant_release_plans(self)

        if (allocated(self%eigenvalues)) then
            deallocate (self%eigenvalues)
        end if

        self%context => null()
        self%local_o = 0
        self%local_o_offset = 0
    end subroutine mpi_circulant_reset_state

    subroutine mpi_circulant_max_comm_size(self, ci, error_code)
        !! Query FFTW MPI for the 1-D distribution it will use over SUBCOMM.
        !! May lower ci%n_processes if FFTW leaves trailing ranks with 0 work.
        !! Overrides ci%local_i / ci%local_i_offset / ci%alloc_local.
        class(circulant_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(C_INTPTR_T) :: system_size, local_i, local_i_offset
        integer(C_INTPTR_T) :: alloc_local
        integer(int32) :: ierr, comm_size, comm_rank
        integer(int32) :: n_active, local_active

        call ensure_fftw_mpi_init()

        error_code = 0

        call MPI_Comm_size(ci%get_SUBCOMM(), comm_size, ierr)
        call MPI_Comm_rank(ci%get_SUBCOMM(), comm_rank, ierr)

        ! Avoid the distributed FFTW query when a singleton communicator is
        ! already forced. Some MPI/FFTW combinations report a zero-rank
        ! distribution here, which incorrectly drives negotiate() into a
        ! shrink-to-zero path.
        if (comm_size == 1) then
            call ci%set_n_processes(1_int64, error_code)
            if (error_code /= 0) return
            call ci%set_partitioning(ci%get_system_size(), 0_int64, error_code=error_code)
            if (error_code /= 0) return
            call ci%set_alloc_local(max(ci%get_system_size(), ci%get_alloc_local()), error_code)
            self%local_o = int(ci%get_system_size(), C_INTPTR_T)
            self%local_o_offset = 0
            return
        end if

        ! FFTW MPI cannot handle size-1 DFTs; for system_size <= 1 keep only
        ! rank 0 active and avoid querying FFTW entirely.
        if (ci%get_system_size() <= 1) then
            call ci%set_n_processes(1_int64, error_code)
            if (error_code /= 0) return
            if (comm_rank == 0) then
                call ci%set_partitioning(1_int64, 0_int64, error_code=error_code)
                if (error_code /= 0) return
                call ci%set_alloc_local(1_int64, error_code)
            else
                call ci%set_partitioning(0_int64, 0_int64, error_code=error_code)
                if (error_code /= 0) return
                call ci%set_alloc_local(0_int64, error_code)
            end if
            return
        end if

        system_size = int(ci%get_system_size(), C_INTPTR_T)
        local_i = int(ci%get_local_i(), C_INTPTR_T)
        local_i_offset = int(ci%get_local_i_offset(), C_INTPTR_T)
        self%local_o = int(ci%get_local_i(), C_INTPTR_T)
        self%local_o_offset = int(ci%get_local_i_offset(), C_INTPTR_T)

        ! Query FFTW for the distribution it will use.
        alloc_local = fftw_mpi_local_size_1d(system_size, &
                                             ci%get_SUBCOMM(), &
                                             FFTW_FORWARD, &
                                             FFTW_ESTIMATE, &
                                             local_i, &
                                             local_i_offset, &
                                             self%local_o, &
                                             self%local_o_offset)


        alloc_local = max(alloc_local, ci%get_alloc_local())

        ! Count how many ranks have local_i > 0
        if (local_i > 0) then
            local_active = 1
        else
            local_active = 0
        end if
        call MPI_Allreduce(local_active, n_active, 1, MPI_INTEGER, MPI_SUM, ci%get_SUBCOMM(), ierr)

        if (n_active == 0) then
            write (error_unit, '(A,I0,A)') &
                "ERROR: fftw_mpi_local_size_1d reported zero active ranks for system_size=", &
                int(system_size, int64), "."
            error_code = 1
            return
        end if

        ! Update layout with FFTW distribution
        call ci%set_n_processes(int(n_active, int64), error_code)
        if (error_code /= 0) return
        call ci%set_partitioning(int(local_i, int64), int(local_i_offset, int64), &
                                 error_code=error_code)
        if (error_code /= 0) return
        call ci%set_alloc_local(int(alloc_local, int64), error_code)

    end subroutine mpi_circulant_max_comm_size

    subroutine mpi_circulant_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! No-op: circulant has no constraints.
        class(circulant_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes
    end subroutine mpi_circulant_store_constraints

    subroutine mpi_circulant_plan(self, context, error_code)
        class(circulant_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        integer(C_INTPTR_T) :: alloc_local, system_size
        integer(C_INTPTR_T) :: local_i, local_i_offset
        integer(int32) :: ierr, local_error, synced_error, n_active, local_active
        integer(int64) :: ci_local_i, ci_local_i_offset, ci_alloc_local

        error_code = 0

        ! Release any prior FFTW/eigenvalue state before replanning this
        ! native propagator instance.
        call mpi_circulant_reset_state(self)
        self%context => context

        system_size = int(self%context%ci%get_system_size(), C_INTPTR_T)

        ! Handle trivial case: system_size == 1
        ! FFTW MPI cannot handle 1D DFTs of size 1 (crashes with invalid pointer)
        ! For size 1, the DFT is the identity transformation, so we skip FFTW
        if (system_size <= 1) then
            self%local_o = int(self%context%ci%get_local_i(), C_INTPTR_T)
            self%local_o_offset = int(self%context%ci%get_local_i_offset(), C_INTPTR_T)
            ! Don't set planned=.true. since we didn't create FFTW plans
            return
        end if

        call ensure_fftw_mpi_init()

        alloc_local = fftw_mpi_local_size_1d(system_size, &
                                             self%context%ci%get_SUBCOMM(), &
                                             FFTW_FORWARD, &
                                             FFTW_ESTIMATE, &
                                             local_i, &
                                             local_i_offset, &
                                             self%local_o, &
                                             self%local_o_offset)

        if (local_i > 0) then
            local_active = 1
        else
            local_active = 0
        end if
        call MPI_Allreduce(local_active, n_active, 1, MPI_INTEGER, MPI_SUM, &
                           self%context%ci%get_SUBCOMM(), ierr)

        alloc_local = max(alloc_local, int(self%context%ci%get_alloc_local(), C_INTPTR_T))

        ci_local_i = self%context%ci%get_local_i()
        ci_local_i_offset = self%context%ci%get_local_i_offset()
        ci_alloc_local = self%context%ci%get_alloc_local()

        local_error = 0
        if (int(local_i, int64) /= ci_local_i) then
            write (error_unit, '(A,I0,A,I0)') &
                "ERROR: negotiate/local_i mismatch in mpi_circulant: ci=", &
                ci_local_i, ", fftw=", int(local_i, int64)
            local_error = 1
        end if
        if (int(local_i_offset, int64) /= ci_local_i_offset) then
            write (error_unit, '(A,I0,A,I0)') &
                "ERROR: negotiate/local_i_offset mismatch in mpi_circulant: ci=", &
                ci_local_i_offset, ", fftw=", int(local_i_offset, int64)
            local_error = 1
        end if
        if (int(alloc_local, int64) /= ci_alloc_local) then
            write (error_unit, '(A,I0,A,I0)') &
                "ERROR: negotiate/alloc_local mismatch in mpi_circulant: ci=", &
                ci_alloc_local, ", fftw=", int(alloc_local, int64)
            local_error = 1
        end if

        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           self%context%ci%get_SUBCOMM(), ierr)
        error_code = synced_error
        if (synced_error /= 0) return

        self%fftw_plan_forward = fftw_mpi_plan_dft_1d(system_size, &
                                                      self%context%state, &
                                                      self%context%state, &
                                                      self%context%ci%get_SUBCOMM(), &
                                                      FFTW_FORWARD, &
                                                      FFTW_MEASURE)

        self%fftw_plan_backward = fftw_mpi_plan_dft_1d(system_size, &
                                                       self%context%state, &
                                                       self%context%state, &
                                                       self%context%ci%get_SUBCOMM(), &
                                                       FFTW_BACKWARD, &
                                                       FFTW_MEASURE)

        self%planned = c_associated(self%fftw_plan_forward) .and. &
                       c_associated(self%fftw_plan_backward)
        if (.not. self%planned) then
            call mpi_circulant_release_plans(self)
        end if

    end subroutine mpi_circulant_plan

    subroutine mpi_circulant_gen_operator(self, array_ptrs, array_sizes, error_code)

        class(circulant_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        real(real64), dimension(:), pointer :: graph_array
        integer(int32) :: nnz
        real(real64), dimension(:), allocatable :: values
        integer(int64), dimension(:), allocatable :: indexes

        error_code = 0

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, graph_array, [array_sizes(1)])

        if (allocated(self%eigenvalues)) then
            deallocate (self%eigenvalues)
        end if

        if (self%local_o > 0) then
            allocate (self%eigenvalues(int(self%local_o)))
        else
            allocate (self%eigenvalues(0))
        end if

        if (array_sizes(1) == 1) then
            ! Complete graph: eigenvalue is N-1 for k=0, and -1 for k!=0
            if (self%local_o > 0) then
                self%eigenvalues = -1 ! Set all to -1 first
                if (self%local_o_offset == 0) then
                    self%eigenvalues(1) = self%context%ci%get_system_size() - 1 ! k=0 case
                end if
            end if
        else
            if (self%local_o > 0) then
                call to_sparse_vector(graph_array, &
                                      nnz, &
                                      indexes, &
                                      values, &
                                      self%local_o, &
                                      self%local_o_offset, &
                                      self%context%ci%get_SUBCOMM())

                call graph_eigenvalues(self%context%ci%get_system_size(), &
                                       self%local_o, &
                                       self%local_o_offset, &
                                       nnz, &
                                       indexes, &
                                       values, &
                                       self%eigenvalues)
            end if
        end if

    end subroutine mpi_circulant_gen_operator

    subroutine mpi_circulant_propagate(self, ts, error_code)

        class(circulant_propagator), intent(inout) :: self
        real(real64), dimension(:), intent(in) :: ts
        integer(int32), intent(out) :: error_code
        integer(int64) :: ci_system_size
        integer(C_INTPTR_T) :: ci_local_i

        error_code = 0
        ci_system_size = self%context%ci%get_system_size()

        ! Handle trivial case: system_size == 1
        ! For size 1, the circulant propagator just applies a phase shift
        ! The eigenvalue for a single-element circulant is just the single element itself
        if (ci_system_size <= 1) then
            ci_local_i = int(self%context%ci%get_local_i(), C_INTPTR_T)
            if (ci_local_i > 0) then
                self%context%state(1:ci_local_i) = exp(cmplx(0.0_real64, -ts(1) * self%eigenvalues(1), kind=real64)) * &
                                                   self%context%state(1:ci_local_i)
            end if
            return
        end if

        call fftw_mpi_execute_dft(self%fftw_plan_forward, self%context%state, self%context%state)

        self%context%state(1:self%local_o) = exp(cmplx(0.0_real64, -ts(1) * self%eigenvalues, kind=real64)) * &
                                             self%context%state(1:self%local_o)

        self%context%state(1:self%local_o) = self%context%state(1:self%local_o) &
                                             / real(self%context%ci%get_system_size(), real64)

        call fftw_mpi_execute_dft(self%fftw_plan_backward, self%context%state, self%context%state)

    end subroutine mpi_circulant_propagate

    subroutine mpi_circulant_destroy(self)
        class(circulant_propagator), intent(inout) :: self

        call mpi_circulant_reset_state(self)

    end subroutine mpi_circulant_destroy

end module mpi_circulant
