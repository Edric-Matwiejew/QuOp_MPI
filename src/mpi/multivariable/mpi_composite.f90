module mpi_composite

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding
    use MPI
    use mpi_backend
    use mpi_circulant_operators
    use cartesian

    implicit none

    private

    public :: composite_propagator

    include 'fftw3-mpi.f03'

    type composite_propagator
    
        type(mpi_context), pointer :: context
        integer(dp), allocatable, dimension(:) :: strides
        integer(sp), allocatable, dimension(:) :: Ns
        integer(dp) :: local_n0_offset
        integer(dp) :: local_n0

        type(C_PTR) :: plan_forward
        type(C_PTR) :: plan_backward

        real(dp), allocatable, dimension(:, :) :: eigenvalues
        real(dp), allocatable, dimension(:) :: mixer
        integer(sp) :: rank
        logical :: operator_generated = .false.

    contains
    
        procedure :: max_comm_size => mpi_composite_max_comm_size
        procedure :: plan => mpi_composite_plan
        procedure :: gen_operator => mpi_composite_gen_operator
        procedure :: propagate => mpi_composite_propagate
        procedure :: destroy => mpi_composite_destroy


    end type composite_propagator
contains
    
    subroutine mpi_composite_max_comm_size(self, system_size, available_ranks, &
        constraint_ptrs, constraint_sizes, max_size, COMM)

        class(composite_propagator), intent(inout) :: self
        integer(dp), intent(in) :: system_size
        integer(sp), intent(in) :: available_ranks
        integer(dp), intent(inout), dimension(:) :: constraint_ptrs
        integer(dp), intent(in), dimension(:) :: constraint_sizes
        integer(sp), intent(out) :: max_size
        integer(sp), intent(in) :: COMM
        integer(sp) :: i


        type(c_ptr) :: leading_dim_ptr
        integer(sp), dimension(:), pointer :: leading_dim_array

        leading_dim_ptr = transfer(constraint_ptrs(1), leading_dim_ptr)
        call c_f_pointer(leading_dim_ptr, leading_dim_array, [constraint_sizes(1)])

        ! constraint is an array of size one that contains the leading dimension of the tensor
        if (available_ranks < leading_dim_array(1)) then
            ! find the highest divisor of the constraint variable 
            do i = available_ranks, 0, -1
                max_size = i
                if (real(leading_dim_array(1))/max_size == leading_dim_array(1)/max_size) then
                    exit
                endif
            enddo
        elseif (available_ranks >= leading_dim_array(1)) then
            max_size = leading_dim_array(1)
        endif

    end subroutine mpi_composite_max_comm_size


    subroutine mpi_composite_plan(self, context)
        class(composite_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context

        self%context => context
        
    end subroutine mpi_composite_plan

    subroutine mpi_composite_gen_operator(self, array_ptrs, array_sizes)
        class(composite_propagator), intent(inout) :: self
        integer(dp), intent(inout), dimension(:) :: array_ptrs
        integer(dp), intent(in), dimension(:) :: array_sizes 

        type(c_ptr) :: array_ptr

        integer(sp), dimension(:), pointer :: Ns
        real(dp), dimension(:,:), pointer :: graph_arrays


        integer(sp) :: alloc_local
        integer(sp) :: n_dim
        integer(sp) :: i, flock, ierr

        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, Ns, [array_sizes(1)])
        array_ptr = transfer(array_ptrs(2), array_ptr)
        call c_f_pointer(array_ptr, graph_arrays, [int(maxval(Ns), sp), size(Ns)])

        allocate(self%Ns(array_sizes(1)))
        self%Ns = Ns
        n_dim = size(self%Ns)

        call MPI_COMM_rank(self%context%SUBCOMM, self%rank, ierr)
        call MPI_COMM_SIZE(self%context%SUBCOMM, flock, ierr)

        if (mod(self%Ns(1), flock) /= 0) then
            write (*, *) "Error: MPI communicator size must be a divisor of the number of grid points in the first dimension."
            stop
        end if

        if (.not. self%operator_generated) then
            call fftw_mpi_init()

            alloc_local = fftw_mpi_local_size(n_dim, int(self%Ns,dp), &
            self%context%SUBCOMM, self%local_n0, self%local_n0_offset)

            allocate(self%strides(n_dim))
            self%strides(n_dim) = 1
            do i = n_dim - 1, 1, -1
                self%strides(i) = self%strides(i + 1)*self%Ns(i + 1)
            end do
        
            self%plan_forward = fftw_mpi_plan_many_dft(n_dim, &
                                                  int(self%Ns, dp), &
                                                  1_dp, &
                                                  self%local_n0, &
                                                  FFTW_MPI_DEFAULT_BLOCK, &
                                                  self%context%initial_state, &
                                                  self%context%initial_state, &
                                                  self%context%SUBCOMM, &
                                                  FFTW_FORWARD, &
                                                  FFTW_MEASURE)

            self%plan_backward = fftw_mpi_plan_many_dft(n_dim, &
                                                   int(self%Ns, dp), &
                                                   1_dp, &
                                                   self%local_n0, &
                                                   FFTW_MPI_DEFAULT_BLOCK, &
                                                   self%context%initial_state, &
                                                   self%context%initial_state, &
                                                   self%context%SUBCOMM, &
                                                   FFTW_BACKWARD, &
                                                   FFTW_MEASURE)

            allocate (self%eigenvalues(maxval(self%Ns), size(self%Ns)))
            allocate(self%mixer(self%context%local_i))

        endif

        if (array_sizes(2) == size(Ns)) then

            do i = 1, size(Ns)
                self%eigenvalues(1, i) = Ns(i) - 1
                self%eigenvalues(2:, i) = -1
            enddo

        else

            do i = 1, size(graph_arrays, 2)

                call graph_eigenvalues(int(self%Ns(i),dp), &
                                       int(self%Ns(i),dp), &
                                       0_dp, &
                                       graph_arrays(:,i), &
                                       self%eigenvalues(:, i))
            end do

        endif

        self%operator_generated = .true.

    end subroutine mpi_composite_gen_operator

    subroutine mpi_composite_propagate(self, t)

        class(composite_propagator), intent(inout) :: self
        real(dp), dimension(:), intent(in) :: t

        real(dp), allocatable :: t_temp(:)

        real(dp), allocatable :: inds(:)
        integer(sp) :: n_dim
        integer(sp) :: ierr
        integer(sp) :: i, j

        n_dim = size(self%Ns)

        allocate (t_temp(n_dim), inds(n_dim))

        if (size(t) == 1) then
            t_temp = t(1)
        else
            t_temp = t
        end if

        call fftw_mpi_execute_dft(self%plan_forward, self%context%initial_state, self%context%initial_state)

        self%mixer = 0
        do i = self%context%local_i_offset + 1, self%context%local_i + self%context%local_i_offset
            call get_index(i, n_dim, self%Ns, self%strides, inds)
            do j = 1, n_dim 
                self%mixer(i - self%context%local_i*self%rank) = &
                self%mixer(i - self%context%local_i*self%rank) &
                + t_temp(j)*self%eigenvalues(int(inds(j)), j)
            end do
        end do

        self%context%initial_state(1:self%context%local_i) = &
            exp(cmplx(0.0_dp, -self%mixer, dp))*self%context%initial_state(1:self%context%local_i)

        call fftw_mpi_execute_dft(self%plan_backward, self%context%initial_state, self%context%initial_state)


        self%context%initial_state(1:self%context%local_i) = &
            self%context%initial_state(1:self%context%local_i)/self%context%system_size

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
        endif
        self%operator_generated = .false.
    end subroutine mpi_composite_destroy

end module mpi_composite
