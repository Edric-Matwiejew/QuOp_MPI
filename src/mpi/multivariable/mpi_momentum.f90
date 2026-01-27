module mpi_momentum

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding
    use MPI
    use mpi_backend
    use cartesian

    implicit none

    private

    public :: momentum_propagator

    include 'fftw3-mpi.f03'

    ! Constants
    complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
    real(dp), parameter :: PI = 3.141592653589793_dp

    type momentum_propagator
    
        type(mpi_context), pointer :: context
        integer(dp), allocatable, dimension(:) :: strides
        integer(sp), allocatable, dimension(:) :: Ns
        integer(dp) :: local_n0_offset
        integer(dp) :: local_n0

        type(C_PTR) :: plan_forward
        type(C_PTR) :: plan_backward

        ! Momentum-space eigenvalues (kinetic energy)
        real(dp), allocatable, dimension(:, :) :: eigenvalues
        complex(dp), allocatable, dimension(:) :: mixer
        
        ! Phase factors for position and momentum space
        complex(dp), allocatable, dimension(:) :: phase_k
        complex(dp), allocatable, dimension(:) :: phase_q
        
        ! Grid parameters
        real(dp), allocatable, dimension(:) :: minsq   ! position space minima
        real(dp), allocatable, dimension(:) :: minsk   ! momentum space minima
        real(dp), allocatable, dimension(:) :: deltasq ! position space deltas
        real(dp), allocatable, dimension(:) :: deltask ! momentum space deltas
        
        integer(sp) :: rank
        logical :: operator_generated = .false.

    contains
    
        procedure :: max_comm_size => mpi_momentum_max_comm_size
        procedure :: plan => mpi_momentum_plan
        procedure :: gen_operator => mpi_momentum_gen_operator
        procedure :: propagate => mpi_momentum_propagate
        procedure :: destroy => mpi_momentum_destroy

    end type momentum_propagator

contains
    
    subroutine mpi_momentum_max_comm_size(self, system_size, available_ranks, &
        constraint_ptrs, constraint_sizes, max_size, COMM)

        class(momentum_propagator), intent(inout) :: self
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

    end subroutine mpi_momentum_max_comm_size


    subroutine mpi_momentum_plan(self, context)
        class(momentum_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context

        self%context => context
        
    end subroutine mpi_momentum_plan

    subroutine mpi_momentum_gen_operator(self, array_ptrs, array_sizes)
        class(momentum_propagator), intent(inout) :: self
        integer(dp), intent(inout), dimension(:) :: array_ptrs
        integer(dp), intent(in), dimension(:) :: array_sizes 

        type(c_ptr) :: array_ptr

        integer(sp), dimension(:), pointer :: Ns
        real(dp), dimension(:), pointer :: minsq_ptr, minsk_ptr
        real(dp), dimension(:), pointer :: deltasq_ptr, deltask_ptr

        integer(sp) :: alloc_local
        integer(sp) :: n_dim
        integer(sp) :: i, j, flock, ierr
        real(dp) :: inds(16)  ! max dimensions
        real(dp) :: grid_point(16)
        real(dp) :: k_val

        ! Unpack arrays from pointers
        ! array_ptrs(1) = Ns
        ! array_ptrs(2) = minsq
        ! array_ptrs(3) = minsk
        ! array_ptrs(4) = deltasq
        ! array_ptrs(5) = deltask
        
        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, Ns, [array_sizes(1)])
        
        n_dim = int(array_sizes(1), sp)
        
        array_ptr = transfer(array_ptrs(2), array_ptr)
        call c_f_pointer(array_ptr, minsq_ptr, [n_dim])
        
        array_ptr = transfer(array_ptrs(3), array_ptr)
        call c_f_pointer(array_ptr, minsk_ptr, [n_dim])
        
        array_ptr = transfer(array_ptrs(4), array_ptr)
        call c_f_pointer(array_ptr, deltasq_ptr, [n_dim])
        
        array_ptr = transfer(array_ptrs(5), array_ptr)
        call c_f_pointer(array_ptr, deltask_ptr, [n_dim])

        allocate(self%Ns(n_dim))
        allocate(self%minsq(n_dim))
        allocate(self%minsk(n_dim))
        allocate(self%deltasq(n_dim))
        allocate(self%deltask(n_dim))
        
        self%Ns = Ns
        self%minsq = minsq_ptr
        self%minsk = minsk_ptr
        self%deltasq = deltasq_ptr
        self%deltask = deltask_ptr

        call MPI_COMM_rank(self%context%SUBCOMM, self%rank, ierr)
        call MPI_COMM_SIZE(self%context%SUBCOMM, flock, ierr)

        if (mod(self%Ns(1), flock) /= 0) then
            write (*, *) "Error: MPI communicator size must be a divisor of the number of grid points in the first dimension."
            stop
        end if

        if (.not. self%operator_generated) then
            call fftw_mpi_init()

            alloc_local = fftw_mpi_local_size(n_dim, int(self%Ns, dp), &
                self%context%SUBCOMM, self%local_n0, self%local_n0_offset)

            allocate(self%strides(n_dim))
            self%strides(n_dim) = 1
            do i = n_dim - 1, 1, -1
                self%strides(i) = self%strides(i + 1) * self%Ns(i + 1)
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

            allocate(self%eigenvalues(maxval(self%Ns), n_dim))
            allocate(self%mixer(self%context%local_i))
            allocate(self%phase_k(self%context%local_i))
            allocate(self%phase_q(self%context%local_i))

        endif

        ! Generate momentum-space eigenvalues (k^2 for kinetic energy)
        do j = 1, n_dim
            do i = 1, self%Ns(j)
                ! Centered momentum: k = minsk + (i-1)*deltask
                k_val = self%minsk(j) + real(i - 1, dp) * self%deltask(j)
                self%eigenvalues(i, j) = k_val * k_val
            end do
        end do

        ! Generate phase factors for position and momentum space transforms
        do i = self%context%local_i_offset + 1, self%context%local_i + self%context%local_i_offset
            call get_index(int(i, sp), int(n_dim, sp), self%Ns, self%strides, inds(1:n_dim))
            
            ! phase_k = exp(-i * sum(k * minsq)) for position->momentum transform
            grid_point(1:n_dim) = self%minsk(1:n_dim) + (inds(1:n_dim) - 1.0_dp) * self%deltask(1:n_dim)
            self%phase_k(i - self%context%local_i_offset) = exp(-cI * sum(grid_point(1:n_dim) * self%minsq(1:n_dim)))
            
            ! phase_q = exp(i * sum(q * minsk)) for momentum->position transform
            grid_point(1:n_dim) = self%minsq(1:n_dim) + (inds(1:n_dim) - 1.0_dp) * self%deltasq(1:n_dim)
            self%phase_q(i - self%context%local_i_offset) = exp(cI * sum(grid_point(1:n_dim) * self%minsk(1:n_dim)))
        end do

        self%operator_generated = .true.

    end subroutine mpi_momentum_gen_operator

    subroutine mpi_momentum_propagate(self, t)

        class(momentum_propagator), intent(inout) :: self
        real(dp), dimension(:), intent(in) :: t

        real(dp), allocatable :: t_temp(:)
        real(dp), allocatable :: inds(:)
        integer(sp) :: n_dim
        integer(sp) :: i, j

        n_dim = size(self%Ns)

        allocate(t_temp(n_dim), inds(n_dim))

        if (size(t) == 1) then
            t_temp = t(1)
        else
            t_temp = t
        end if

        ! Apply checkerboard phase for centered FFT
        do i = self%context%local_i_offset + 1, self%context%local_i + self%context%local_i_offset
            call get_index(int(i, sp), n_dim, self%Ns, self%strides, inds)
            self%context%initial_state(i - self%context%local_i_offset) = &
                ((-1.0_dp)**real(sum(inds - 1), dp)) * self%context%initial_state(i - self%context%local_i_offset)
        end do

        ! Forward FFT (position -> momentum)
        call fftw_mpi_execute_dft(self%plan_forward, self%context%initial_state, self%context%initial_state)

        ! Apply phase_k
        self%context%initial_state(1:self%context%local_i) = &
            self%phase_k * self%context%initial_state(1:self%context%local_i)

        ! Build momentum-space mixer (kinetic energy evolution)
        self%mixer = cmplx(0.0_dp, 0.0_dp, dp)
        do i = self%context%local_i_offset + 1, self%context%local_i + self%context%local_i_offset
            call get_index(int(i, sp), n_dim, self%Ns, self%strides, inds)
            do j = 1, n_dim
                self%mixer(i - self%context%local_i_offset) = self%mixer(i - self%context%local_i_offset) &
                    + t_temp(j) * self%eigenvalues(int(inds(j)), j)
            end do
        end do

        ! Apply kinetic energy evolution
        self%context%initial_state(1:self%context%local_i) = &
            exp(-cI * self%mixer) * self%context%initial_state(1:self%context%local_i)

        ! Apply checkerboard phase before inverse FFT
        do i = self%context%local_i_offset + 1, self%context%local_i + self%context%local_i_offset
            call get_index(int(i, sp), n_dim, self%Ns, self%strides, inds)
            self%context%initial_state(i - self%context%local_i_offset) = &
                ((-1.0_dp)**real(sum(inds - 1), dp)) * self%context%initial_state(i - self%context%local_i_offset)
        end do

        ! Backward FFT (momentum -> position)
        call fftw_mpi_execute_dft(self%plan_backward, self%context%initial_state, self%context%initial_state)

        ! Apply phase_q and normalize
        self%context%initial_state(1:self%context%local_i) = &
            self%phase_q * self%context%initial_state(1:self%context%local_i) / self%context%system_size

        deallocate(t_temp, inds)

    end subroutine mpi_momentum_propagate

    subroutine mpi_momentum_destroy(self)
        class(momentum_propagator), intent(inout) :: self
        
        self%context => null()
        
        if (self%operator_generated) then
            if (allocated(self%eigenvalues)) deallocate(self%eigenvalues)
            if (allocated(self%mixer)) deallocate(self%mixer)
            if (allocated(self%phase_k)) deallocate(self%phase_k)
            if (allocated(self%phase_q)) deallocate(self%phase_q)
            if (allocated(self%Ns)) deallocate(self%Ns)
            if (allocated(self%strides)) deallocate(self%strides)
            if (allocated(self%minsq)) deallocate(self%minsq)
            if (allocated(self%minsk)) deallocate(self%minsk)
            if (allocated(self%deltasq)) deallocate(self%deltasq)
            if (allocated(self%deltask)) deallocate(self%deltask)
            call fftw_destroy_plan(self%plan_backward)
            call fftw_destroy_plan(self%plan_forward)
            call fftw_mpi_cleanup()
        endif
        
        self%operator_generated = .false.
        
    end subroutine mpi_momentum_destroy

end module mpi_momentum
