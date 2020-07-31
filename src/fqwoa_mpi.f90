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
    local_o_offset = local_o_offset_temp

end subroutine mpi_local_size

subroutine qwoa_state(  N, &
                        p, &
                        alloc_local, &
                        local_i, &
                        local_o, &
                        gammas, &
                        ts, &
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
    real(dp), intent(in) :: gammas(p)
    real(dp), intent(in) :: ts(p)
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

            call fftw_mpi_execute_dft(plan_forward, fdata, fdata)

            fdata(1:local_o) = exp(-complex(0,1d0) * gammas(i) * lambdas) *fdata(1:local_o)/real(N,8)

            call fftw_mpi_execute_dft(plan_backward, fdata, fdata)

            fdata(1:local_i) = exp(-complex(0,1d0) * ts(i) * qualities) *fdata(1:local_i)

        enddo
    endif


    if (flag < 0) then
        call fftw_destroy_plan(plan_backward)
        call fftw_destroy_plan(plan_forward)
        call fftw_free(cdata)
    endif

end subroutine qwoa_state

subroutine save_dist_complex(   file_name, &
                                group_name, &
                                dataset_name, &
                                access_type, &
                                N, &
                                local_i, &
                                local_i_offset, &
                                complex_array, &
                                MPI_communicator)

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use HDF5
    use MPI

    implicit none

    character(len = 128), intent(in) :: file_name
    character(len = 128), intent(in) :: group_name
    character(len = 128), intent(in) :: dataset_name
    character(len = 1), intent(in) :: access_type
    integer(dp), intent(in) :: N
    integer(sp), intent(in) :: local_i
    integer(sp), intent(in) :: local_i_offset
    complex(dp), dimension(local_i), intent(in) :: complex_array
    integer(sp), intent(in) :: MPI_communicator

    ! HDF5 variables.
    integer(HID_T) :: file_id          ! File identifier
    integer(HID_T) :: plist_id         ! Property list identifier.
    integer(HID_T) :: dset_id          ! Dataspace identifier.
    integer(HID_T) :: file_data_id     ! Dataspace identifier in file.
    integer(HID_T) :: memspace_id      ! Dataspace identifier in memory.
    integer(HID_T) :: group_id         ! Group identifier.

    integer(HSIZE_T), dimension(1) :: ftn_dimensions
    integer(HSIZE_T), dimension(1) :: h5_dimensions

    integer(HSIZE_T), dimension(2) :: local_count
    integer(HSIZE_T), dimension(1) :: data_offset
    integer(sp) :: dataset_rank = 1

    integer(HID_T) ::  cmplx_type_id
    integer(HID_T) ::  cmplx_type_size
    integer(HID_T) ::  re_size, im_size
    integer(HID_T) ::  type_offset
    integer(HID_T) ::  real_id
    integer(HID_T) ::  imag_id

    logical :: group_exists
    logical :: dataset_exists
    integer(sp) :: error

    ! MPI variables
    integer(sp) :: info
    integer(sp) :: ierr

    ftn_dimensions = N
    h5_dimensions = N
    local_count = local_i
    data_offset = local_i_offset

    info = MPI_INFO_NULL

    call h5open_f(error)

    call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
    call h5pset_fapl_mpio_f(plist_id, MPI_communicator, info, error)

    if (access_type == "a") then

        if (access(trim(file_name) // ".h5", "r") == 0) then

            call H5fopen_f(trim(file_name) // ".h5", H5F_ACC_RDWR_F, file_id, &
                           error, access_prp = plist_id)

        else

            call h5fcreate_f(trim(file_name) // ".h5", H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
            call h5fcreate_f(trim(file_name) // ".h5", H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
        endif

    elseif (access_type == "w") then

        call h5fcreate_f(trim(file_name) // ".h5", H5F_ACC_TRUNC_F, file_id, error, &
                         access_prp = plist_id)

    endif

    call h5lexists_f(file_id, trim(group_name), group_exists, error)

    if (.not. group_exists) then
        call h5gcreate_f (file_id, trim(group_name), group_id, error)
    else
        call h5gopen_f(file_id, trim(group_name), group_id, error)
    endif

    call h5pclose_f(plist_id, error)

    call h5screate_simple_f(dataset_rank, ftn_dimensions, file_data_id, error)

    call h5tget_size_f(H5T_NATIVE_DOUBLE, re_size, error)
    call h5tget_size_f(H5T_NATIVE_DOUBLE, im_size, error)
    cmplx_type_size = re_size + im_size
    call h5tcreate_f(H5T_COMPOUND_F, cmplx_type_size, cmplx_type_id, error)
    type_offset = 0
    call h5tinsert_f(cmplx_type_id, "real", type_offset, H5T_NATIVE_DOUBLE, error)
    type_offset = re_size
    call h5tinsert_f(cmplx_type_id, "imag", type_offset, H5T_NATIVE_DOUBLE, error)

    call h5dcreate_f(file_id, trim(group_name) // trim(dataset_name), cmplx_type_id, &
                     file_data_id, dset_id, error)

    call h5sclose_f(file_data_id, error)

    call h5screate_simple_f(dataset_rank, local_count, memspace_id, error)

    call h5dget_space_f(dset_id, file_data_id, error)
    call h5sselect_hyperslab_f(file_data_id, H5S_SELECT_SET_F, data_offset, &
                               local_count, error)

    call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
    call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

    type_offset = 0
    call h5tcreate_f(H5T_COMPOUND_F, re_size, real_id, error)
    call h5tinsert_f(real_id, "real", type_offset, H5T_NATIVE_DOUBLE, error)
    call h5tcreate_f(H5T_COMPOUND_F, im_size, imag_id, error)
    call h5tinsert_f(imag_id, "imag", type_offset, H5T_NATIVE_DOUBLE, error)

    call h5dwrite_f(dset_id, real_id,  real(complex_array), ftn_dimensions, error, &
                    file_space_id = file_data_id, mem_space_id = memspace_id, &
                    xfer_prp = plist_id)

    call h5dwrite_f(dset_id, imag_id,  aimag(complex_array), ftn_dimensions, error, &
                    file_space_id = file_data_id, mem_space_id = memspace_id, &
                    xfer_prp = plist_id)

    call h5sclose_f(file_data_id, error)
    call h5sclose_f(memspace_id, error)

    CALL h5dclose_f(dset_id, error)
    CALL h5pclose_f(plist_id, error)

    call h5gclose_f (group_id, error)
    call h5fclose_f(file_id, error)

    call h5close_f(error)

    call MPI_barrier(MPI_communicator, ierr)

end subroutine save_dist_complex

subroutine save_dist_real(  file_name, &
                            group_name, &
                            dataset_name, &
                            access_type, &
                            N, &
                            local_i, &
                            local_i_offset, &
                            real_array, &
                            MPI_communicator)

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
    use HDF5
    use MPI

    implicit none

    character(len = 128), intent(in) :: file_name
    character(len = 128), intent(in) :: group_name
    character(len = 128), intent(in) :: dataset_name
    character(len = 1), intent(in) :: access_type
    integer(dp), intent(in) :: N
    integer(sp), intent(in) :: local_i
    integer(sp), intent(in) :: local_i_offset
    real(dp), dimension(local_i) :: real_array
    integer(sp), intent(in) :: MPI_communicator

    ! HDF5 variables.
    integer(HID_T) :: file_id          ! File identifier
    integer(HID_T) :: plist_id         ! Property list identifier.
    integer(HID_T) :: dset_id          ! Dataspace identifier.
    integer(HID_T) :: file_data_id     ! Dataspace identifier in file.
    integer(HID_T) :: memspace_id      ! Dataspace identifier in memory.
    integer(HID_T) :: group_id          ! Group identifier in file.

    integer(HSIZE_T), dimension(1) :: ftn_dimensions
    integer(HSIZE_T), dimension(1) :: h5_dimensions

    integer(HSIZE_T), dimension(2) :: local_count
    integer(HSIZE_T), dimension(1) :: data_offset
    integer(sp) :: dataset_rank = 1

    logical :: group_exists
    logical :: dataset_exists
    integer(sp) :: error

    ! MPI variables
    integer(sp) :: info
    integer(sp) :: ierr

    ftn_dimensions = N
    h5_dimensions = N
    local_count = local_i
    data_offset = local_i_offset

    info = MPI_INFO_NULL

    call h5open_f(error)

    call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
    call h5pset_fapl_mpio_f(plist_id, MPI_communicator, info, error)

    if (access_type == "a") then

        if (access(trim(file_name) // ".h5", "r") == 0) then

            call H5fopen_f(trim(file_name) // ".h5", H5F_ACC_RDWR_F, file_id, &
                           error, access_prp = plist_id)

        else

            call h5fcreate_f(trim(file_name) // ".h5", H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
            call h5fcreate_f(trim(file_name) // ".h5", H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
        endif

    elseif (access_type == "w") then

        call h5fcreate_f(trim(file_name) // ".h5", H5F_ACC_TRUNC_F, file_id, error, &
                         access_prp = plist_id)

    endif

    call h5lexists_f(file_id, trim(group_name), group_exists, error)

    if (.not. group_exists) then
        call h5gcreate_f (file_id, trim(group_name), group_id, error)
    else
        call h5gopen_f(file_id, trim(group_name), group_id, error)
    endif

    call h5pclose_f(plist_id, error)

    call h5screate_simple_f(dataset_rank, ftn_dimensions, file_data_id, error)

    call h5dcreate_f(file_id, trim(group_name) // trim(dataset_name), H5T_NATIVE_DOUBLE, file_data_id, &
                     dset_id, error)

    call h5sclose_f(file_data_id, error)

    call h5screate_simple_f(dataset_rank, local_count, memspace_id, error)

    call h5dget_space_f(dset_id, file_data_id, error)
    call h5sselect_hyperslab_f(file_data_id, H5S_SELECT_SET_F, data_offset, &
                               local_count, error)

    call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
    call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)

    call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, real_array, ftn_dimensions, error, &
                    file_space_id = file_data_id, mem_space_id = memspace_id, &
                    xfer_prp = plist_id)

    CALL h5sclose_f(file_data_id, error)
    CALL h5sclose_f(memspace_id, error)

    CALL h5dclose_f(dset_id, error)
    CALL h5pclose_f(plist_id, error)

    call h5gclose_f (group_id, error)
    CALL h5fclose_f(file_id, error)

    CALL h5close_f(error)

    call MPI_barrier(MPI_communicator, ierr)
    
end subroutine save_dist_real
