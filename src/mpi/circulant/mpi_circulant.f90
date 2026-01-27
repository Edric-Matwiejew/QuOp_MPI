module mpi_circulant

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding
    use mpi_circulant_operators
    use sparse_vector
    use mpi_backend

    implicit none

    private

    public :: circulant_propagator

    include 'fftw3-mpi.f03'

    type circulant_propagator

        type(mpi_context), pointer :: context
        type(c_ptr) :: fftw_plan_forward = c_null_ptr
        type(c_ptr) :: fftw_plan_backward = c_null_ptr
        logical :: planned = .false.
        real(dp), dimension(:), allocatable :: eigenvalues

        integer(C_INTPTR_T) :: system_size
        integer(C_INTPTR_T) :: local_i
        integer(C_INTPTR_T) :: local_i_offset
        integer(C_INTPTR_T) :: local_o
        integer(C_INTPTR_T) :: local_o_offset


    contains

        procedure :: max_comm_size => mpi_circulant_max_comm_size
        procedure :: plan => mpi_circulant_plan
        procedure :: gen_operator => mpi_circulant_gen_operator
        procedure :: propagate => mpi_circulant_propagate
        procedure :: destroy => mpi_circulant_destroy

    end type circulant_propagator


contains

    subroutine mpi_circulant_max_comm_size(self, system_size, available_ranks, &
        constraint_ptrs, constraint_sizes, max_size, COMM)
        class(circulant_propagator), intent(inout) :: self
        integer(dp), intent(in) :: system_size
        integer(sp), intent(in) :: available_ranks
        integer(dp), intent(inout), dimension(:) :: constraint_ptrs
        integer(dp), intent(in), dimension(:) :: constraint_sizes
        integer(sp), intent(out) :: max_size
        integer(sp), intent(in) :: COMM

        integer(C_INTPTR_T) :: local_i, local_i_offset, local_o, local_o_offset
        integer(C_INTPTR_T) :: alloc_local
        integer(C_INTPTR_T) :: min_local_i
        integer(sp) :: ierr, comm_size, comm_rank
        integer(sp) :: n_active

        call MPI_Comm_size(COMM, comm_size, ierr)
        call MPI_Comm_rank(COMM, comm_rank, ierr)

        ! Query FFTW for the distribution it will use
        alloc_local = fftw_mpi_local_size_1d(int(system_size, C_INTPTR_T), &
                                              COMM, &
                                              FFTW_FORWARD, &
                                              FFTW_ESTIMATE, &
                                              local_i, &
                                              local_i_offset, &
                                              local_o, &
                                              local_o_offset)

        ! Find minimum local_i across all ranks
        call MPI_Allreduce(local_i, min_local_i, 1, MPI_INTEGER8, MPI_MIN, COMM, ierr)

        ! Count how many ranks have local_i > 0
        if (local_i > 0) then
            n_active = 1
        else
            n_active = 0
        endif
        call MPI_Allreduce(MPI_IN_PLACE, n_active, 1, MPI_INTEGER, MPI_SUM, COMM, ierr)

        ! Return the number of active ranks (those with local_i > 0)
        max_size = n_active

    end subroutine mpi_circulant_max_comm_size


    subroutine mpi_circulant_plan(self, context)
        class(circulant_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context

        integer(sp) :: alloc_local, rank, ierr

        self%context => context

        self%system_size = self%context%system_size

        ! Handle trivial case: system_size == 1
        ! FFTW MPI cannot handle 1D DFTs of size 1 (crashes with invalid pointer)
        ! For size 1, the DFT is the identity transformation, so we skip FFTW
        if (self%system_size <= 1) then
            self%local_i = self%context%local_i
            self%local_i_offset = self%context%local_i_offset
            self%local_o = self%context%local_i
            self%local_o_offset = self%context%local_i_offset
            ! Don't set planned=.true. since we didn't create FFTW plans
            return
        endif

        alloc_local = fftw_mpi_local_size_1d(   self%system_size, &
                                                self%context%SUBCOMM, &
                                                FFTW_FORWARD, &
                                                FFTW_MEASURE, &
                                                self%local_i, &
                                                self%local_i_offset, &
                                                self%local_o, &
                                                self%local_o_offset)

        ! Update context with FFTW's required distribution
        if (self%context%alloc_local < alloc_local) then
            self%context%alloc_local = alloc_local
            deallocate(self%context%initial_state)
            allocate(self%context%initial_state(alloc_local))
        endif

        ! Update partition info from FFTW's distribution
        self%context%local_i = self%local_i
        self%context%local_i_offset = self%local_i_offset

        self%fftw_plan_forward = fftw_mpi_plan_dft_1d(self%system_size, &
                                            self%context%initial_state, &
                                            self%context%initial_state, &
                                            self%context%SUBCOMM, &
                                            FFTW_FORWARD, &
                                            FFTW_MEASURE)

        self%fftw_plan_backward = fftw_mpi_plan_dft_1d(self%system_size, &
                                             self%context%initial_state, &
                                             self%context%initial_state, &
                                             self%context%SUBCOMM, &
                                             FFTW_BACKWARD, &
                                             FFTW_MEASURE)

        self%planned = .true.

    end subroutine mpi_circulant_plan

    subroutine mpi_circulant_gen_operator(self, array_ptrs, array_sizes)

        class(circulant_propagator), intent(inout) :: self
        integer(dp), intent(inout), dimension(:) :: array_ptrs
        integer(dp), intent(in), dimension(:) :: array_sizes 

        type(c_ptr) :: array_ptr

        real(dp), dimension(:), pointer :: graph_array
        integer(sp) :: nnz
        real(dp), dimension(:), allocatable :: values
        integer(dp), dimension(:), allocatable :: indexes
        
        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, graph_array, [array_sizes(1)])

        allocate(self%eigenvalues(int(self%local_o)))

        if (array_sizes(1) == 1) then
            ! Complete graph: eigenvalue is N-1 for k=0, and -1 for k!=0
            self%eigenvalues = -1  ! Set all to -1 first
            if (self%local_o_offset == 0) then
                self%eigenvalues(1) = self%context%system_size - 1  ! k=0 case
            endif
        else

            call to_sparse_vector(  graph_array, &
                                    nnz, &
                                    indexes, &
                                    values, &
                                    self%local_o, &
                                    self%local_o_offset, &
                                    self%context%SUBCOMM)

            call graph_eigenvalues( self%context%system_size, &
                                    self%local_o, &
                                    self%local_o_offset, &
                                    nnz, &
                                    indexes, &
                                    values, &
                                    self%eigenvalues)
        endif

    end subroutine mpi_circulant_gen_operator

    subroutine mpi_circulant_propagate(self, ts)

        class(circulant_propagator), intent(inout) :: self
        real(dp), dimension(:), intent(in) :: ts

        ! Handle trivial case: system_size == 1
        ! For size 1, the circulant propagator just applies a phase shift
        ! The eigenvalue for a single-element circulant is just the single element itself
        if (self%system_size <= 1) then
            if (self%local_i > 0) then
                self%context%initial_state(1:self%local_i) = exp(cmplx(0.0_dp, -ts(1)*self%eigenvalues(1), kind=dp)) * &
                                                             self%context%initial_state(1:self%local_i)
            endif
            return
        endif

        call fftw_mpi_execute_dft(self%fftw_plan_forward, self%context%initial_state, self%context%initial_state)

        self%context%initial_state(1:self%local_o) = exp(cmplx(0.0_dp, -ts(1)*self%eigenvalues, kind=dp))* &
                                                         self%context%initial_state(1:self%local_o)

        self%context%initial_state(1:self%local_o) = self%context%initial_state(1:self%local_o) &
                                                         /real(self%context%system_size, dp)

        call fftw_mpi_execute_dft(self%fftw_plan_backward, self%context%initial_state, self%context%initial_state)

    end subroutine mpi_circulant_propagate

    subroutine mpi_circulant_destroy(self)
        class(circulant_propagator), intent(inout) :: self

        if (self%planned) then
                call fftw_destroy_plan(self%fftw_plan_backward)
                call fftw_destroy_plan(self%fftw_plan_forward)
                self%planned = .false.
        endif

        if (allocated(self%eigenvalues)) then
                deallocate (self%eigenvalues)
        endif

        self%context => null()

    end subroutine mpi_circulant_destroy

end module mpi_circulant
