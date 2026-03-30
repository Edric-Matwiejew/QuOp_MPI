module mpi_composite

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use, intrinsic :: iso_c_binding
    use mpi
    use mpi_backend, only: mpi_context
    use mpi_circulant_operators, only: graph_eigenvalues
    use cartesian, only: get_index
    use comm_info_module, only: quop_mpi_layout_t
    use fftw_mpi_init_guard, only: ensure_fftw_mpi_init

    implicit none

    private

    public :: composite_propagator

    include 'fftw3-mpi.f03'

    type composite_propagator

        type(mpi_context), pointer :: context => null()
        integer(int64), allocatable, dimension(:) :: strides
        integer(int32), allocatable, dimension(:) :: Ns
        integer(int64) :: local_n0_offset
        integer(int64) :: local_n0

        type(C_PTR) :: plan_forward
        type(C_PTR) :: plan_backward

        real(real64), allocatable, dimension(:, :) :: eigenvalues
        real(real64), allocatable, dimension(:) :: mixer
        logical :: operator_generated = .false.

        ! Constraints stored by store_constraints for negotiate loop
        integer(int32) :: leading_dim = 0
        integer(int32) :: n_dim_constraint = 0
        integer(int32), allocatable, dimension(:) :: constraint_Ns

    contains

        procedure :: max_comm_size => mpi_composite_max_comm_size
        procedure :: store_constraints => mpi_composite_store_constraints
        procedure :: plan => mpi_composite_plan
        procedure :: gen_operator => mpi_composite_gen_operator
        procedure :: propagate => mpi_composite_propagate
        procedure :: destroy => mpi_composite_destroy

    end type composite_propagator
contains

    subroutine mpi_composite_max_comm_size(self, ci, error_code)
        !! Constrain the communicator layout for FFTW.
        !!
        !! 1D: The gen_operator path uses fftw_mpi_local_size_many_1d
        !! (six-step algorithm) whose distribution differs from a simple
        !! block partition.  We query that same function here to get the
        !! true local_i / local_i_offset / alloc_local, and count active
        !! ranks (mirroring the circulant propagator).
        !!
        !! nD (n >= 2): constrain n_processes to divide the leading
        !! dimension, then (once no shrink is pending) query FFTW slab
        !! decomposition and store local_i/local_i_offset/alloc_local.
        class(composite_propagator), intent(inout) :: self
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int32) :: i, available_ranks, max_size, ierr
        integer(C_INTPTR_T) :: alloc_local, local_n0, local_n0_offset
        integer(C_INTPTR_T) :: local_no, local_o_offset
        integer(int32) :: n_active, local_active
        integer(int64) :: slab_size, local_i_64, local_i_offset_64

        call ensure_fftw_mpi_init()

        error_code = 0

        available_ranks = int(ci%get_n_processes(), int32)

        if (self%leading_dim <= 0) return ! no constraint stored yet

        if (self%n_dim_constraint == 1) then
            ! 1D: query fftw_mpi_local_size_many_1d to match gen_operator
            alloc_local = fftw_mpi_local_size_many_1d( &
                          int(self%constraint_Ns(1), C_INTPTR_T), &
                          1_C_INTPTR_T, &
                          ci%get_SUBCOMM(), &
                          FFTW_FORWARD, &
                          FFTW_ESTIMATE, &
                          local_n0, &
                          local_n0_offset, &
                          local_no, &
                          local_o_offset)

            ! Count ranks with work (mirroring circulant)
            if (local_n0 > 0) then
                local_active = 1
            else
                local_active = 0
            end if
            call MPI_Allreduce(local_active, n_active, 1, MPI_INTEGER, &
                               MPI_SUM, ci%get_SUBCOMM(), ierr)

            call ci%set_n_processes(int(n_active, int64), error_code)
            if (error_code /= 0) return
            call ci%set_partitioning(int(local_n0, int64), int(local_n0_offset, int64), &
                                     error_code=error_code)
            if (error_code /= 0) return
            call ci%set_alloc_local(int(alloc_local, int64), error_code)
            if (error_code /= 0) return
        else
            ! nD: constrain comm size to divide leading dimension evenly.
            if (available_ranks < self%leading_dim) then
                do i = available_ranks, 1, -1
                    max_size = i
                    if (mod(self%leading_dim, i) == 0) exit
                end do
            else
                max_size = self%leading_dim
            end if

            call ci%set_n_processes(int(max_size, int64), error_code)
            if (error_code /= 0) return

            ! If a shrink is still pending, SUBCOMM does not yet match max_size.
            ! Defer FFTW distribution query until the next negotiate iteration.
            if (max_size == available_ranks) then
                alloc_local = fftw_mpi_local_size(self%n_dim_constraint, &
                                                  int(self%constraint_Ns, C_INTPTR_T), &
                                                  ci%get_SUBCOMM(), &
                                                  local_n0, &
                                                  local_n0_offset)

                slab_size = product(int(self%constraint_Ns(2:self%n_dim_constraint), int64))
                local_i_64 = int(local_n0, int64) * slab_size
                local_i_offset_64 = int(local_n0_offset, int64) * slab_size

                call ci%set_partitioning(local_i_64, local_i_offset_64, &
                                         error_code=error_code)
                if (error_code /= 0) return

                call ci%set_alloc_local(int(alloc_local, int64), error_code)
                if (error_code /= 0) return
            end if
        end if

    end subroutine mpi_composite_max_comm_size

    subroutine mpi_composite_store_constraints(self, constraint_ptrs, constraint_sizes)
        !! Store the full Ns array for subsequent max_comm_size calls.
        !! constraint_ptrs(1) is a c_ptr (as int64) to an int32 array.
        class(composite_propagator), intent(inout) :: self
        integer(int64), intent(in), dimension(:) :: constraint_ptrs
        integer(int64), intent(in), dimension(:) :: constraint_sizes

        type(c_ptr) :: ptr
        integer(int32), dimension(:), pointer :: arr
        integer(int32) :: nd

        if (size(constraint_ptrs) > 0 .and. constraint_sizes(1) > 0) then
            ptr = transfer(constraint_ptrs(1), ptr)
            nd = int(constraint_sizes(1), int32)
            call c_f_pointer(ptr, arr, [nd])
            self%leading_dim = arr(1)
            self%n_dim_constraint = nd
            if (allocated(self%constraint_Ns)) deallocate (self%constraint_Ns)
            allocate (self%constraint_Ns(nd))
            self%constraint_Ns = arr(1:nd)
        end if
    end subroutine mpi_composite_store_constraints

    subroutine mpi_composite_plan(self, context, error_code)
        class(composite_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context
        integer(int32), intent(out) :: error_code

        error_code = 0

        self%context => context

    end subroutine mpi_composite_plan

    subroutine mpi_composite_gen_operator(self, array_ptrs, array_sizes, error_code)
        class(composite_propagator), intent(inout) :: self
        integer(int64), intent(inout), dimension(:) :: array_ptrs
        integer(int64), intent(in), dimension(:) :: array_sizes
        integer(int32), intent(out) :: error_code

        type(c_ptr) :: array_ptr

        integer(int32), dimension(:), pointer :: Ns
        real(real64), dimension(:, :), pointer :: graph_arrays

        integer(C_INTPTR_T) :: alloc_local
        integer(int32) :: n_dim
        integer(int32) :: i, flock, ierr, local_error, synced_error
        integer(int64) :: slab_size, expected_local_i, expected_local_i_offset
        integer(int32) :: ci_subcomm
        integer(int64) :: ci_local_i, ci_local_i_offset, ci_alloc_local

        error_code = 0

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, Ns, [array_sizes(1)])

        allocate (self%Ns(array_sizes(1)))
        self%Ns = Ns
        n_dim = size(self%Ns)
        ci_subcomm = self%context%ci%get_SUBCOMM()
        ci_local_i = self%context%ci%get_local_i()
        ci_local_i_offset = self%context%ci%get_local_i_offset()
        ci_alloc_local = self%context%ci%get_alloc_local()

        call MPI_COMM_SIZE(ci_subcomm, flock, ierr)

        local_error = 0
        if (n_dim > 1 .and. mod(self%Ns(1), flock) /= 0) then
            write (error_unit, '(A)') &
                "ERROR: MPI communicator size must divide the number of grid points in the first dimension."
            local_error = 1
        end if
        call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                           ci_subcomm, ierr)
        error_code = synced_error
        if (synced_error /= 0) return

        if (.not. self%operator_generated) then
            call ensure_fftw_mpi_init()

            if (n_dim == 1) then
                ! 1D: six-step algorithm via fftw_mpi_local_size_many_1d
                alloc_local = fftw_mpi_local_size_many_1d( &
                              int(self%Ns(1), C_INTPTR_T), &
                              1_C_INTPTR_T, &
                              ci_subcomm, &
                              FFTW_FORWARD, &
                              FFTW_ESTIMATE, &
                              self%local_n0, &
                              self%local_n0_offset, &
                              self%local_n0, &
                              self%local_n0_offset)

                allocate (self%strides(1))
                self%strides(1) = 1
            else
                ! nD: slab decomposition via fftw_mpi_local_size
                alloc_local = fftw_mpi_local_size(n_dim, &
                                                  int(self%Ns, C_INTPTR_T), &
                                                  ci_subcomm, &
                                                  self%local_n0, &
                                                  self%local_n0_offset)

                allocate (self%strides(n_dim))
                self%strides(n_dim) = 1
                do i = n_dim - 1, 1, -1
                    self%strides(i) = self%strides(i + 1) * self%Ns(i + 1)
                end do
            end if

            if (n_dim > 1) then
                slab_size = product(int(self%Ns(2:n_dim), int64))
            else
                slab_size = 1_int64
            end if
            expected_local_i = int(self%local_n0, int64) * slab_size
            expected_local_i_offset = int(self%local_n0_offset, int64) * slab_size

            local_error = 0
            if (expected_local_i /= ci_local_i) then
                write (error_unit, '(A,I0,A,I0)') &
                    "ERROR: negotiate/local_i mismatch in mpi_composite: ci=", &
                    ci_local_i, ", fftw=", expected_local_i
                local_error = 1
            end if
            if (expected_local_i_offset /= ci_local_i_offset) then
                write (error_unit, '(A,I0,A,I0)') &
                    "ERROR: negotiate/local_i_offset mismatch in mpi_composite: ci=", &
                    ci_local_i_offset, ", fftw=", expected_local_i_offset
                local_error = 1
            end if
            if (int(alloc_local, int64) /= ci_alloc_local) then
                write (error_unit, '(A,I0,A,I0)') &
                    "ERROR: negotiate/alloc_local mismatch in mpi_composite: ci=", &
                    ci_alloc_local, ", fftw=", int(alloc_local, int64)
                local_error = 1
            end if

            call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, MPI_MAX, &
                               ci_subcomm, ierr)
            error_code = synced_error
            if (synced_error /= 0) return

            self%plan_forward = fftw_mpi_plan_many_dft(n_dim, &
                                                       int(self%Ns, C_INTPTR_T), &
                                                       1_C_INTPTR_T, &
                                                       self%local_n0, &
                                                       self%local_n0, &
                                                       self%context%state, &
                                                       self%context%state, &
                                                       ci_subcomm, &
                                                       FFTW_FORWARD, &
                                                       FFTW_MEASURE)

            self%plan_backward = fftw_mpi_plan_many_dft(n_dim, &
                                                        int(self%Ns, C_INTPTR_T), &
                                                        1_C_INTPTR_T, &
                                                        self%local_n0, &
                                                        self%local_n0, &
                                                        self%context%state, &
                                                        self%context%state, &
                                                        ci_subcomm, &
                                                        FFTW_BACKWARD, &
                                                        FFTW_MEASURE)

            allocate (self%eigenvalues(maxval(self%Ns), size(self%Ns)))
            allocate (self%mixer(ci_local_i))

        end if

        if (array_sizes(2) == size(Ns)) then

            do i = 1, size(Ns)
                self%eigenvalues(1, i) = Ns(i) - 1
                self%eigenvalues(2:, i) = -1
            end do

        else

            array_ptr = transfer(array_ptrs(2), array_ptr)
            call c_f_pointer(array_ptr, graph_arrays, [ &
                             int(array_sizes(2) / size(Ns), int32), size(Ns)])

            do i = 1, size(graph_arrays, 2)

                call graph_eigenvalues(int(self%Ns(i), int64), &
                                       int(self%Ns(i), int64), &
                                       0_int64, &
                                       graph_arrays(:, i), &
                                       self%eigenvalues(:, i))
            end do

        end if

        self%operator_generated = .true.

    end subroutine mpi_composite_gen_operator

    subroutine mpi_composite_propagate(self, t, error_code)

        class(composite_propagator), intent(inout) :: self
        real(real64), dimension(:), intent(in) :: t
        integer(int32), intent(out) :: error_code

        real(real64), allocatable :: t_temp(:)

        real(real64), allocatable :: inds(:)
        integer(int32) :: n_dim
        integer(int32) :: ierr
        integer(int32) :: j
        integer(int64) :: i
        integer(int64) :: ci_local_i, ci_local_i_offset, ci_system_size

        error_code = 0

        n_dim = size(self%Ns)
        ci_local_i = self%context%ci%get_local_i()
        ci_local_i_offset = self%context%ci%get_local_i_offset()
        ci_system_size = self%context%ci%get_system_size()

        allocate (t_temp(n_dim), inds(n_dim))

        if (size(t) == 1) then
            t_temp = t(1)
        else
            t_temp = t
        end if

        call fftw_mpi_execute_dft(self%plan_forward, self%context%state, self%context%state)

        self%mixer = 0
        do i = ci_local_i_offset + 1, ci_local_i + ci_local_i_offset
            call get_index(int(i, int32), n_dim, self%Ns, self%strides, inds)
            do j = 1, n_dim
                self%mixer(i - ci_local_i_offset) = &
                    self%mixer(i - ci_local_i_offset) &
                    + t_temp(j) * self%eigenvalues(int(inds(j)), j)
            end do
        end do

        self%context%state(1:ci_local_i) = &
            exp(cmplx(0.0_real64, -self%mixer, real64)) * self%context%state(1:ci_local_i)

        call fftw_mpi_execute_dft(self%plan_backward, self%context%state, self%context%state)

        self%context%state(1:ci_local_i) = &
            self%context%state(1:ci_local_i) / ci_system_size

    end subroutine mpi_composite_propagate

    subroutine mpi_composite_destroy(self)
        class(composite_propagator), intent(inout) :: self
        self%context => null()
        if (self%operator_generated) then
            deallocate (self%eigenvalues)
            deallocate (self%mixer)
            call fftw_destroy_plan(self%plan_backward)
            call fftw_destroy_plan(self%plan_forward)
            call fftw_mpi_cleanup()
        end if
        if (allocated(self%constraint_Ns)) deallocate (self%constraint_Ns)
        self%operator_generated = .false.
    end subroutine mpi_composite_destroy

end module mpi_composite
