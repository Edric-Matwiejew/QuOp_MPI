module qhost_fourier

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding
    use MPI
    use host

    implicit none
 
    include 'fftw3-mpi.f03'

    integer(dp), allocatable, dimension(:) :: strides, Ns
    integer(dp) :: local_n0, local_n0_offset
    integer(sp) :: n_dim
 
    type(C_PTR) :: plan_forward
    type(C_PTR) :: plan_backward

    complex(dp), allocatable, dimension(:) :: phase_k
    complex(dp), allocatable, dimension(:) :: phase_q
 
    real(dp), allocatable, dimension(:,:) :: eigenvalues

    ! constants
    complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
    real(dp) :: PI = 3.141592653589793_dp; 
 
    contains

    subroutine plan_partition(  n_dim, & 
                                Ns, & 
                                MPI_communicator, &
                                alloc_local, &
                                local_i, &
                                local_i_offset, &
                                local_n0, &
                                local_n0_offset, &
                                strides)
    

    
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim)
        integer(sp), intent(in) :: MPI_communicator 
        integer(dp), intent(out) :: alloc_local
        integer(dp), intent(out) :: local_i
        integer(dp), intent(out) :: local_i_offset
        integer(dp), intent(out) :: local_n0
        integer(dp), intent(out) :: local_n0_offset
        integer(dp), intent(out) :: strides(n_dim)

        integer(sp) :: rank
        integer(sp) :: flock
        integer(sp) :: ierr
    
        integer(sp) :: i
    
        call MPI_COMM_RANK(MPI_communicator, rank, ierr)
        call MPI_COMM_SIZE(MPI_communicator, flock, ierr)

        if (mod(Ns(1), flock) /= 0) then
                write(*,*) "Error: MPI communicator size must be a divisor of the number of grid points in the first dimension." 
                stop
        endif

        call fftw_mpi_init()
   
        alloc_local = fftw_mpi_local_size(n_dim, Ns, MPI_communicator, local_n0, local_n0_offset)

        local_i = local_n0
        local_i_offset = local_n0_offset
        do i = 2, size(Ns)
            local_i = local_i * Ns(i) 
            local_i_offset = local_i_offset * Ns(i)
        enddo

        strides(n_dim) = 1
        do i = n_dim - 1, 1, -1
            strides(i) = strides(i + 1)*Ns(i + 1)
        enddo

    end subroutine plan_partition

    subroutine evolve_ft(  N, & 
                        n_dim, &
                        Ns, &
                        alloc_local, &
                        local_i, &
                        local_i_offset, &
                        local_n0, &
                        strides, &
                        n_t, &
                        t, &
                        N_max, &
                        eigenvalues, &
                        phase_k, &
                        phase_q, &
                        state, &
                        MPI_communicator, &
                        flag)
    
        use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
        use, intrinsic :: iso_c_binding
        use MPI

        implicit none

        include 'fftw3-mpi.f03'
 
        complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
        real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
 
    
        integer(dp), intent(in) :: N
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim)
        integer(dp), intent(in) :: alloc_local
        integer(dp), intent(in) :: local_i
        integer(dp), intent(in) :: local_i_offset
        integer(dp), intent(in) :: local_n0
        integer(dp), intent(in) :: strides(n_dim)
        integer(sp), intent(in) :: n_t
        real(dp), intent(in) :: t(n_t)
        integer(dp), intent(in) :: N_max
        complex(dp), intent(in) :: eigenvalues(N_max, n_dim)
       complex(dp), intent(inout), target :: state(alloc_local)
        integer(sp), intent(in) :: MPI_communicator
        integer(sp), intent(in) :: flag

        complex(dp) :: mixer(local_i)
        real(dp), allocatable :: t_temp(:)

           real(dp) :: inds(n_dim)
        integer(sp) :: ierr
        integer(sp), save :: rank
        integer(sp) :: i,j
    
        if (flag > 0) then
                
            call MPI_COMM_RANK(MPI_communicator, rank, ierr)

            plan_forward = fftw_mpi_plan_many_dft(      n_dim, Ns, 1_dp, & 
                                                        int(local_n0, dp), &
                                                        FFTW_MPI_DEFAULT_BLOCK, &
                                                        state, state, &
                                                        MPI_communicator, &
                                                        FFTW_FORWARD, &
                                                        FFTW_MEASURE)
 
            plan_backward = fftw_mpi_plan_many_dft(     n_dim, Ns, 1_dp, & 
                                                        int(local_n0, dp), &
                                                        FFTW_MPI_DEFAULT_BLOCK, &
                                                        state, state, &
                                                        MPI_communicator, &
                                                        FFTW_BACKWARD, &
                                                        FFTW_MEASURE)
            endif
    
        if (flag == 0) then

             allocate(t_temp(n_dim))

             if (n_t == 1) then
                t_temp = t(1)
             else
                t_temp = t
             endif

             do i = local_i_offset + 1, local_i + local_i_offset
                 call get_index(i, n_dim, Ns, strides, inds)
                 state(i - local_i*rank) = ((-1.0_dp)**real(sum(inds - 1), dp))*state(i - local_i*rank)
             enddo

            call fftw_mpi_execute_dft(plan_forward, state, state)
 
            state(1:local_i) = phase_k*state(1:local_i)

            mixer = 0
            do i = local_i_offset + 1, local_i + local_i_offset
                call get_index(i, n_dim, Ns, strides, inds)
                do j = 1, n_dim
                        mixer(i - local_i*rank) = t_temp(j)*(mixer(i - local_i*rank) + eigenvalues(inds(j),j))
                enddo
            enddo

            state(1:local_i) = exp(-cmplx(0.0_dp, 1.0_dp, dp)*mixer)*state(1:local_i)

            do i = local_i_offset + 1, local_i + local_i_offset
                call get_index(i, n_dim, Ns, strides, inds)
                state(i - local_i*rank) = ((-1.0_dp)**real(sum(inds - 1), dp))*state(i - local_i*rank)
            enddo
 
            call fftw_mpi_execute_dft(plan_backward, state, state)
 
            state(1:local_i) = phase_q*state(1:local_i)/real(N,8)

        endif
    
        if (flag < 0) then
            call fftw_destroy_plan(plan_backward)
            call fftw_destroy_plan(plan_forward)
        endif
    
    end subroutine evolve_ft
    
end module qhost_fourier
