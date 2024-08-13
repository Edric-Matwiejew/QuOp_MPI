module io

        contains

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

    character(len = 512), intent(in) :: file_name
    character(len = 512), intent(in) :: group_name
    character(len = 512), intent(in) :: dataset_name
    character(len = 1), intent(in) :: access_type
    integer(dp), intent(in) :: N
    integer(sp), intent(in) :: local_i
    integer(sp), intent(in) :: local_i_offset
    complex(dp), dimension(local_i), intent(in) :: complex_array
    integer(sp), intent(in) :: MPI_communicator

!f2py  character*512 intent(in) :: file_name
!f2py  character*512 intent(in) :: group_name
!f2py  character*512 intent(in) :: dataset_name
!f2py  character*1 intent(in) :: access_type
!f2py  integer(kind=dp) intent(in) :: n
!f2py  integer(kind=sp), optional,intent(in),check(shape(complex_array, 0) == local_i),depend(complex_array) :: local_i=shape(complex_array, 0)
!f2py  integer(kind=sp) intent(in) :: local_i_offset
!f2py  complex(kind=dp) dimension(local_i),intent(in) :: complex_array
!f2py  integer(kind=sp) intent(in) :: mpi_communicator
 

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

        if (access(trim(file_name), "r") == 0) then

            call H5fopen_f(trim(file_name), H5F_ACC_RDWR_F, file_id, &
                           error, access_prp = plist_id)

        else

            call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
            call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
        endif

    elseif (access_type == "w") then

        call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
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

    character(len = 512), intent(in) :: file_name
    character(len = 512), intent(in) :: group_name
    character(len = 512), intent(in) :: dataset_name
    character(len = 1), intent(in) :: access_type
    integer(dp), intent(in) :: N
    integer(sp), intent(in) :: local_i
    integer(sp), intent(in) :: local_i_offset
    real(dp), dimension(local_i) :: real_array
    integer(sp), intent(in) :: MPI_communicator

!f2py  character*512 intent(in) :: file_name
!f2py  character*512 intent(in) :: group_name
!f2py  character*512 intent(in) :: dataset_name
!f2py  character*1 intent(in) :: access_type
!f2py  integer(kind=dp) intent(in) :: n
!f2py  integer(kind=sp), optional,intent(in),check(shape(real_array, 0) == local_i),depend(real_array) :: local_i=shape(real_array, 0)
!f2py  integer(kind=sp) intent(in) :: local_i_offset
!f2py  real(kind=dp) dimension(local_i) :: real_array
!f2py  integer(kind=sp) intent(in) :: mpi_communicator
 

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

        if (access(trim(file_name), "r") == 0) then

            call H5fopen_f(trim(file_name), H5F_ACC_RDWR_F, file_id, &
                           error, access_prp = plist_id)

        else

            call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
            call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp = plist_id)
        endif

    elseif (access_type == "w") then

        call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
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

end module io
