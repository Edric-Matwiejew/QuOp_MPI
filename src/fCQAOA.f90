module continuous
    
    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding
    use MPI

    implicit none

    include 'fftw3-mpi.f03'
 
    complex(dp), parameter :: cI = cmplx(0.0_dp, 1.0_dp, dp)
    real(dp), parameter :: pi = 4.0_dp*atan(1.0_dp)
    
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
            strides(i) = strides(i + 1)*Ns(i)
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
    
        integer(sp), intent(in) :: N
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim)
        integer(dp), intent(in) :: alloc_local
        integer(dp), intent(in) :: local_i
        integer(dp), intent(in) :: local_i_offset
        integer(dp), intent(in) :: local_n0
        integer(sp), intent(in) :: strides(n_dim)
        integer(sp), intent(in) :: n_t
        real(dp), intent(in) :: t(n_t)
        integer(sp), intent(in) :: N_max
        complex(dp), intent(in) :: eigenvalues(N_max, n_dim)
        complex(dp), intent(in) :: phase_k(local_i)
        complex(dp), intent(in) :: phase_q(local_i)
        complex(dp), intent(inout), target :: state(alloc_local)
        integer(sp), intent(in) :: MPI_communicator
        integer(sp), intent(in) :: flag

        complex(dp) :: mixer(local_i)
        real(dp), allocatable :: t_temp(:)
    
        type(C_PTR), save :: plan_forward
        type(C_PTR), save :: plan_backward

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
                        mixer(i - local_i*rank) = mixer(i - local_i*rank) + t_temp(j)*eigenvalues(inds(j),j)
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
    

    subroutine evolve_n_dft(  N, & 
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
                        state, &
                        MPI_communicator, &
                        flag)
    
        integer(sp), intent(in) :: N
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim)
        integer(dp), intent(in) :: alloc_local
        integer(dp), intent(in) :: local_i
        integer(dp), intent(in) :: local_i_offset
        integer(dp), intent(in) :: local_n0
        integer(sp), intent(in) :: strides(n_dim)
        integer(sp), intent(in) :: n_t
        real(dp), intent(in) :: t(n_t)
        integer(sp), intent(in) :: N_max
        complex(dp), intent(in) :: eigenvalues(N_max, n_dim)
        complex(dp), intent(inout), target :: state(alloc_local)
        integer(sp), intent(in) :: MPI_communicator
        integer(sp), intent(in) :: flag

        complex(dp) :: mixer(local_i)
        real(dp), allocatable :: t_temp(:)
    
        type(C_PTR), save :: plan_forward
        type(C_PTR), save :: plan_backward

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

            call fftw_mpi_execute_dft(plan_forward, state, state)
 
            mixer = 0
            do i = local_i_offset + 1, local_i + local_i_offset
                call get_index(i, n_dim, Ns, strides, inds)
                do j = 1, n_dim
                        mixer(i - local_i*rank) = t_temp(j)*(mixer(i - local_i*rank) + eigenvalues(inds(j),j))
                enddo
            enddo

            state(1:local_i) = exp(-cmplx(0.0_dp, 1.0_dp, dp)*mixer)*state(1:local_i)

            call fftw_mpi_execute_dft(plan_backward, state, state)

            state(1:local_i) = state(1:local_i)/real(N,8)
                
        endif
    
        if (flag < 0) then
            call fftw_destroy_plan(plan_backward)
            call fftw_destroy_plan(plan_forward)
        endif
    
    end subroutine evolve_n_dft
    
    subroutine dist_vector( f, &
                            n_dim, &
                            Ns, &
                            strides, &
                            deltas, &
                            mins, &
                            local_i_offset, &
                            local_i, &
                            vec)
    
! f2py intent(callback) f
        external :: f
        complex(dp) :: f_temp
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim) ! the number of grid points in each dimension
        integer(sp), intent(in) :: strides(n_dim)
        real(dp), intent(in) :: deltas(n_dim) ! the grid-size in each dimension
        real(dp), intent(in) :: mins(n_dim) ! the minimum in each dimension, maxs = mins + Ns*deltas
        integer, intent(in) :: local_i_offset ! Starting index alogn n0 dimension.
        integer, intent(in) :: local_i ! Number of indices alogn the n0 dimension at this rank.
        complex(dp), intent(inout) :: vec(local_i)
    
        real(dp) :: grid_point(n_dim)
        integer :: i, j
    
        do i = local_i_offset + 1, local_i + local_i_offset
            call get_index(i, n_dim, Ns, strides, grid_point)
            grid_point = mins + (grid_point - 1.0_dp)*deltas
            call f(grid_point, f_temp, n_dim)
            vec(i - local_i_offset) = f_temp
        enddo
    
    end subroutine dist_vector

    subroutine get_index(i, n_dim, Ns, strides, inds)

        integer(sp), intent(in) :: i
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim)
        integer(sp), intent(in) :: strides(n_dim)
        real(dp), intent(out) :: inds(n_dim)

        integer(sp) :: j

        do j = 1, n_dim
            inds(j) = mod((i - 1)/strides(j), Ns(j)) + 1
        enddo

    end subroutine get_index
    
    subroutine gen_local_grid(  N, &
                                n_dim, &
                                Ns, &
                                strides, &
                                deltas, &
                                mins, &
                                local_i_offset, &
                                local_i, &
                                local_grid)
    
        integer(dp), intent(in) :: N
        integer(sp), intent(in) :: n_dim
        integer(dp), intent(in) :: Ns(n_dim) ! the number of grid points in each dimension
        integer(sp), intent(in) :: strides(n_dim)
        real(dp), intent(in) :: deltas(n_dim) ! the grid-size in each dimension
        real(dp), intent(in) :: mins(n_dim) ! the minimum in each dimension, maxs = mins + Ns*deltas
        integer, intent(in) :: local_i_offset ! Starting index alogn n0 dimension.
        integer, intent(in) :: local_i ! Number of indices alogn the n0 dimension at this rank.
        real(dp), intent(out) :: local_grid(local_i, n_dim)
    
        real(dp) :: grid_point(n_dim)
        integer :: i
    
        do i = local_i_offset + 1, local_i + local_i_offset
            call get_index(i, n_dim, Ns, strides, grid_point)
            local_grid(i - local_i_offset, :) = mins + (grid_point - 1.0_dp)*deltas
        enddo
    
    end subroutine gen_local_grid

end module continuous
