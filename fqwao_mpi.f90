subroutine mpi_local_size(  N, &
                                MPI_communicator, &
                                alloc_local, &
                                local_i, &
                                local_i_offset, &
                                local_o, &
                                local_o_offset)

    use, intrinsic :: iso_c_binding
    use :: mpi
    implicit none
    include 'fftw3-mpi.f03'

    integer, intent(in) :: N
    integer, intent(in) :: MPI_communicator
    integer, intent(out) :: alloc_local
    integer, intent(out) :: local_i
    integer, intent(out) :: local_i_offset
    integer, intent(out) :: local_o
    integer, intent(out) :: local_o_offset

    integer(C_INTPTR_T) :: N_temp
    integer(C_INTPTR_T) :: local_i_temp
    integer(C_INTPTR_T) :: local_i_offset_temp
    integer(C_INTPTR_T) :: local_o_temp
    integer(C_INTPTR_T) :: local_o_offset_temp

    N_temp = N

    alloc_local = fftw_mpi_local_size_1d(   N_temp, &
                                            MPI_communicator, &
                                            FFTW_FORWARD, &
                                            FFTW_MEASURE, &
                                            local_i_temp, &
                                            local_i_offset_temp, &
                                            local_o_temp, &
                                            local_o_offset_temp)

    local_i = local_i_temp
    local_i_offset = local_i_offset_temp
    local_o = local_o_temp
    local_o_offset = local_i_offset_temp

end subroutine mpi_local_size

subroutine qwao_state(  N, &
                        p, &
                        alloc_local, &
                        local_i, &
                        local_o, &
                        betas, &
                        gammas, &
                        qualities,  &
                        lambdas,  &
                        state, &
                        final_state, &
                        MPI_communicator, &
                        flag)

    use, intrinsic :: iso_c_binding
    use :: mpi
    implicit none
    include 'fftw3-mpi.f03'

    integer, intent(in) :: N
    integer, intent(in) :: p
    integer, intent(in) :: alloc_local
    integer, intent(in) :: local_i
    integer, intent(in) :: local_o
    real(8), intent(in) :: betas(p)
    real(8), intent(in) :: gammas(p)
    real(8), intent(in) :: qualities(local_i)
    complex(8), intent(in) :: lambdas(local_o)
    complex(8), intent(in) :: state(alloc_local)
    complex(8), intent(inout), target :: final_state(alloc_local)
    integer, intent(in) :: MPI_communicator
    integer, intent(in) :: flag

    integer(C_INTPTR_T) :: N_temp, alloc_local_temp
    type(C_PTR), save :: plan_forward, plan_backward, cdata
    complex(C_DOUBLE_COMPLEX), pointer, save :: data(:)

    integer :: i

    if (flag >  0) then

        N_temp = N
        alloc_local_temp = alloc_local

        cdata = fftw_alloc_complex(alloc_local_temp)
        call c_f_pointer(cdata, data, [alloc_local_temp])

        plan_forward = fftw_mpi_plan_dft_1d(N_temp, &
                                            data, &
                                            data, &
                                            MPI_communicator, &
                                            FFTW_FORWARD, &
                                            FFTW_MEASURE)

        plan_backward = fftw_mpi_plan_dft_1d(   N_temp, &
                                                data, &
                                                data, &
                                                MPI_communicator, &
                                                FFTW_BACKWARD, &
                                                FFTW_MEASURE)
        data => final_state


    endif


    if (flag == 0) then

        final_state = state

        do i = 1, p

            data= exp(-complex(0,1d0) * gammas(i) * qualities) *data

            call fftw_mpi_execute_dft(plan_forward,data,data)

            data= exp(-complex(0,1d0) * betas(i) * lambdas) *data/real(N,8)

            call fftw_mpi_execute_dft(plan_backward,data,data)

        enddo
    endif


    if (flag < 0) then
        call fftw_destroy_plan(plan_backward)
        call fftw_destroy_plan(plan_forward)
        call fftw_free(cdata)
    endif


end subroutine qwao_state
