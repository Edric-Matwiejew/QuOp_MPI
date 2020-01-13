subroutine mpi_local_size(  N, &
                            MPI_communicator, &
                            alloc_local, &
                            local_i, &
                            local_i_offset, &
                            local_o, &
                            local_o_offset)

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding

    implicit none

    include 'fftw3-mpi.f03'

    integer(dp), intent(in) :: N
    integer(sp), intent(in) :: MPI_communicator
    integer(sp), intent(out) :: alloc_local
    integer(sp), intent(out) :: local_i
    integer(sp), intent(out) :: local_i_offset
    integer(sp), intent(out) :: local_o
    integer(sp), intent(out) :: local_o_offset

    integer(C_INTPTR_T) :: N_temp
    integer(C_INTPTR_T) :: local_i_temp
    integer(C_INTPTR_T) :: local_i_offset_temp
    integer(C_INTPTR_T) :: local_o_temp
    integer(C_INTPTR_T) :: local_o_offset_temp

    write(*,*) N
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

    write(*,*) N

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

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use, intrinsic :: iso_c_binding

    implicit none

    include 'fftw3-mpi.f03'

    integer(dp), intent(in) :: N
    integer(sp), intent(in) :: p
    integer(sp), intent(in) :: alloc_local
    integer(sp), intent(in) :: local_i
    integer(sp), intent(in) :: local_o
    real(dp), intent(in) :: betas(p)
    real(dp), intent(in) :: gammas(p)
    real(dp), intent(in) :: qualities(local_i)
    complex(dp), intent(in) :: lambdas(local_o)
    complex(dp), intent(in) :: state(alloc_local)
    complex(dp), intent(inout), target :: final_state(alloc_local)
    integer(sp), intent(in) :: MPI_communicator
    integer(sp), intent(in) :: flag

    integer(C_INTPTR_T) :: N_temp, alloc_local_temp
    type(C_PTR), save :: plan_forward, plan_backward, cdata
    complex(C_DOUBLE_COMPLEX), pointer, save :: fdata(:)

    integer :: i

    if (flag >  0) then

        N_temp = N

        alloc_local_temp = alloc_local

        cdata = fftw_alloc_complex(alloc_local_temp)
        call c_f_pointer(cdata, fdata, [alloc_local_temp])

        plan_forward = fftw_mpi_plan_dft_1d(N_temp, &
                                            fdata, &
                                            fdata, &
                                            MPI_communicator, &
                                            FFTW_FORWARD, &
                                            FFTW_MEASURE)

        plan_backward = fftw_mpi_plan_dft_1d(   N_temp, &
                                                fdata, &
                                                fdata, &
                                                MPI_communicator, &
                                                FFTW_BACKWARD, &
                                                FFTW_MEASURE)
        fdata => final_state


    endif


    if (flag == 0) then

        final_state = state

        do i = 1, p

            fdata= exp(-complex(0,1d0) * gammas(i) * qualities) *fdata

            call fftw_mpi_execute_dft(plan_forward, fdata, fdata)

            fdata= exp(-complex(0,1d0) * betas(i) * lambdas) *fdata/real(N,8)

            call fftw_mpi_execute_dft(plan_backward, fdata, fdata)

        enddo
    endif


    if (flag < 0) then
        call fftw_destroy_plan(plan_backward)
        call fftw_destroy_plan(plan_forward)
        call fftw_free(cdata)
    endif

end subroutine qwao_state
