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

        max_size = available_ranks

    end subroutine mpi_circulant_max_comm_size


    subroutine mpi_circulant_plan(self, context)
        class(circulant_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context

        integer(sp) :: alloc_local, rank, ierr

        self%context => context

        self%system_size = self%context%system_size

        alloc_local = fftw_mpi_local_size_1d(   self%system_size, &
                                                self%context%SUBCOMM, &
                                                FFTW_FORWARD, &
                                                FFTW_MEASURE, &
                                                self%local_i, &
                                                self%local_i_offset, &
                                                self%local_o, &
                                                self%local_o_offset)

        if ((self%context%alloc_local < alloc_local) .or. (self%local_i /= self%context%local_i) ) then
            call MPI_Comm_rank(self%context%SUBCOMM, rank, ierr)
            if (rank == 0) then
                write(*,*) 'Warning: Input size inconsistency between FFTW (circulant propagator) ', &
                        'requirements and context instance resizing state array to statisfy FFTW constraints.'
            endif
            self%context%alloc_local = alloc_local
            self%context%local_i = self%local_i
            self%context%local_i_offset = self%local_i_offset
            deallocate(self%context%initial_state)
            allocate(self%context%initial_state(alloc_local))
        endif

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
            self%eigenvalues(2:) = -1
            if (self%local_o_offset == 0) then
                self%eigenvalues(1) = self%context%system_size - 1
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

        self%context%initial_state = self%context%initial_state

        call fftw_mpi_execute_dft(self%fftw_plan_forward, self%context%initial_state, self%context%initial_state)
        self%context%initial_state(1:self%local_o) = exp(cmplx(0, -ts(1)*self%eigenvalues))* &
                                                         self%context%initial_state(1:self%local_o)

        self%context%initial_state(1:self%local_o) = self%context%initial_state(1:self%local_o) &
                                                         /real(self%context%system_size, dp)

        call fftw_mpi_execute_dft(self%fftw_plan_backward, self%context%initial_state, self%context%initial_state)

        !self%context%initial_state = self%context%initial_state

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
