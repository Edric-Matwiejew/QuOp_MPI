module io

    implicit none

    private
    public :: save_dist_complex, save_dist_real

contains

    subroutine check_hdf5(error_code, MPI_communicator, routine_name)

        use MPI

        implicit none

        integer, intent(in) :: error_code
        integer, intent(in) :: MPI_communicator
        character(len=*), intent(in) :: routine_name

        integer :: ierr
        integer :: abort_code

        if (error_code == 0) return

        abort_code = error_code
        if (abort_code == 0) abort_code = 1

        write (*, '(A,1X,A,1X,I0)') 'parallel_io HDF5 failure:', trim(routine_name), error_code
        call MPI_Abort(MPI_communicator, abort_code, ierr)

    end subroutine check_hdf5

    subroutine check_mpi(ierr, MPI_communicator, routine_name)

        use MPI

        implicit none

        integer, intent(in) :: ierr
        integer, intent(in) :: MPI_communicator
        character(len=*), intent(in) :: routine_name

        integer :: abort_code
        integer :: abort_ierr

        if (ierr == MPI_SUCCESS) return

        abort_code = ierr
        if (abort_code == 0) abort_code = 1

        write (*, '(A,1X,A,1X,I0)') 'parallel_io MPI failure:', trim(routine_name), ierr
        call MPI_Abort(MPI_communicator, abort_code, abort_ierr)

    end subroutine check_mpi

    subroutine save_dist_complex(file_name, &
                                 group_name, &
                                 dataset_name, &
                                 access_type, &
                                 N, &
                                 local_i, &
                                 local_i_offset, &
                                 complex_array, &
                                 MPI_communicator)

        use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
        use HDF5
        use MPI

        implicit none

        character(len=512), intent(in) :: file_name
        character(len=512), intent(in) :: group_name
        character(len=512), intent(in) :: dataset_name
        character(len=1), intent(in) :: access_type
        integer(int64), intent(in) :: N
        integer(int32), intent(in) :: local_i
        integer(int32), intent(in) :: local_i_offset
        complex(real64), dimension(local_i), intent(in) :: complex_array
        integer(int32), intent(in) :: MPI_communicator

!f2py  character*512 intent(in) :: file_name
!f2py  character*512 intent(in) :: group_name
!f2py  character*512 intent(in) :: dataset_name
!f2py  character*1 intent(in) :: access_type
!f2py  integer(kind=int64) intent(in) :: n
!f2py  integer(kind=int32), optional,intent(in),check(shape(complex_array, 0) == local_i),depend(complex_array) :: local_i=shape(complex_array, 0)
!f2py  integer(kind=int32) intent(in) :: local_i_offset
!f2py  complex(kind=real64) dimension(local_i),intent(in) :: complex_array
!f2py  integer(kind=int32) intent(in) :: mpi_communicator

        ! HDF5 variables.
        integer(HID_T) :: file_id ! File identifier
        integer(HID_T) :: plist_id ! Property list identifier.
        integer(HID_T) :: dset_id ! Dataspace identifier.
        integer(HID_T) :: file_data_id ! Dataspace identifier in file.
        integer(HID_T) :: memspace_id ! Dataspace identifier in memory.
        integer(HID_T) :: group_id ! Group identifier.

        integer(HSIZE_T), dimension(1) :: ftn_dimensions
        integer(HSIZE_T), dimension(1) :: h5_dimensions

        integer(HSIZE_T), dimension(1) :: local_count
        integer(HSIZE_T), dimension(1) :: data_offset
        integer(int32) :: dataset_rank = 1

        integer(HID_T) ::  cmplx_type_id
        integer(HID_T) ::  cmplx_type_size
        integer(HID_T) ::  re_size, im_size
        integer(HID_T) ::  type_offset
        integer(HID_T) ::  real_id
        integer(HID_T) ::  imag_id

        logical :: group_exists
        logical :: dataset_exists
        logical :: file_exists
        integer(int32) :: error

        ! MPI variables
        integer(int32) :: info
        integer(int32) :: ierr

        ftn_dimensions = N
        h5_dimensions = N
        local_count = local_i
        data_offset = local_i_offset

        info = MPI_INFO_NULL

        call h5open_f(error)
        call check_hdf5(error, MPI_communicator, 'h5open_f')

        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pcreate_f(file_access)')
        call h5pset_fapl_mpio_f(plist_id, MPI_communicator, info, error)
        call check_hdf5(error, MPI_communicator, 'h5pset_fapl_mpio_f')

        if (access_type == "a") then

            inquire (file=trim(file_name), exist=file_exists)
            if (file_exists) then

                call H5fopen_f(trim(file_name), H5F_ACC_RDWR_F, file_id, &
                               error, access_prp=plist_id)
                call check_hdf5(error, MPI_communicator, 'h5fopen_f')

            else

                call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                                 access_prp=plist_id)
                call check_hdf5(error, MPI_communicator, 'h5fcreate_f')
            end if

        elseif (access_type == "w") then

            call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp=plist_id)
            call check_hdf5(error, MPI_communicator, 'h5fcreate_f')

        end if

        call h5lexists_f(file_id, trim(group_name), group_exists, error)
        call check_hdf5(error, MPI_communicator, 'h5lexists_f')

        if (.not. group_exists) then
            call h5gcreate_f(file_id, trim(group_name), group_id, error)
            call check_hdf5(error, MPI_communicator, 'h5gcreate_f')
        else
            call h5gopen_f(file_id, trim(group_name), group_id, error)
            call check_hdf5(error, MPI_communicator, 'h5gopen_f')
        end if

        call h5pclose_f(plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pclose_f(file_access)')

        call h5screate_simple_f(dataset_rank, ftn_dimensions, file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5screate_simple_f(file_data)')

        call h5tget_size_f(H5T_NATIVE_DOUBLE, re_size, error)
        call check_hdf5(error, MPI_communicator, 'h5tget_size_f(real)')
        call h5tget_size_f(H5T_NATIVE_DOUBLE, im_size, error)
        call check_hdf5(error, MPI_communicator, 'h5tget_size_f(imag)')
        cmplx_type_size = re_size + im_size
        call h5tcreate_f(H5T_COMPOUND_F, cmplx_type_size, cmplx_type_id, error)
        call check_hdf5(error, MPI_communicator, 'h5tcreate_f(complex)')
        type_offset = 0
        call h5tinsert_f(cmplx_type_id, "real", type_offset, H5T_NATIVE_DOUBLE, error)
        call check_hdf5(error, MPI_communicator, 'h5tinsert_f(complex.real)')
        type_offset = re_size
        call h5tinsert_f(cmplx_type_id, "imag", type_offset, H5T_NATIVE_DOUBLE, error)
        call check_hdf5(error, MPI_communicator, 'h5tinsert_f(complex.imag)')

        call h5dcreate_f(file_id, trim(group_name)//trim(dataset_name), cmplx_type_id, &
                         file_data_id, dset_id, error)
        call check_hdf5(error, MPI_communicator, 'h5dcreate_f(complex)')

        call h5sclose_f(file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5sclose_f(file_data_initial)')

        call h5screate_simple_f(dataset_rank, local_count, memspace_id, error)
        call check_hdf5(error, MPI_communicator, 'h5screate_simple_f(memspace)')

        call h5dget_space_f(dset_id, file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5dget_space_f')
        call h5sselect_hyperslab_f(file_data_id, H5S_SELECT_SET_F, data_offset, &
                                   local_count, error)
        call check_hdf5(error, MPI_communicator, 'h5sselect_hyperslab_f')

        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pcreate_f(dataset_xfer)')
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        call check_hdf5(error, MPI_communicator, 'h5pset_dxpl_mpio_f')

        type_offset = 0
        call h5tcreate_f(H5T_COMPOUND_F, re_size, real_id, error)
        call check_hdf5(error, MPI_communicator, 'h5tcreate_f(real_field)')
        call h5tinsert_f(real_id, "real", type_offset, H5T_NATIVE_DOUBLE, error)
        call check_hdf5(error, MPI_communicator, 'h5tinsert_f(real_field.real)')
        call h5tcreate_f(H5T_COMPOUND_F, im_size, imag_id, error)
        call check_hdf5(error, MPI_communicator, 'h5tcreate_f(imag_field)')
        call h5tinsert_f(imag_id, "imag", type_offset, H5T_NATIVE_DOUBLE, error)
        call check_hdf5(error, MPI_communicator, 'h5tinsert_f(imag_field.imag)')

        call h5dwrite_f(dset_id, real_id, real(complex_array), ftn_dimensions, error, &
                        file_space_id=file_data_id, mem_space_id=memspace_id, &
                        xfer_prp=plist_id)
        call check_hdf5(error, MPI_communicator, 'h5dwrite_f(complex.real)')

        call h5dwrite_f(dset_id, imag_id, aimag(complex_array), ftn_dimensions, error, &
                        file_space_id=file_data_id, mem_space_id=memspace_id, &
                        xfer_prp=plist_id)
        call check_hdf5(error, MPI_communicator, 'h5dwrite_f(complex.imag)')

        call h5sclose_f(file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5sclose_f(file_data)')
        call h5sclose_f(memspace_id, error)
        call check_hdf5(error, MPI_communicator, 'h5sclose_f(memspace)')

        call h5tclose_f(real_id, error)
        call check_hdf5(error, MPI_communicator, 'h5tclose_f(real_field)')
        call h5tclose_f(imag_id, error)
        call check_hdf5(error, MPI_communicator, 'h5tclose_f(imag_field)')
        call h5tclose_f(cmplx_type_id, error)
        call check_hdf5(error, MPI_communicator, 'h5tclose_f(complex)')

        call h5dclose_f(dset_id, error)
        call check_hdf5(error, MPI_communicator, 'h5dclose_f')
        call h5pclose_f(plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pclose_f(dataset_xfer)')

        call h5gclose_f(group_id, error)
        call check_hdf5(error, MPI_communicator, 'h5gclose_f')
        call h5fclose_f(file_id, error)
        call check_hdf5(error, MPI_communicator, 'h5fclose_f')

        call h5close_f(error)
        call check_hdf5(error, MPI_communicator, 'h5close_f')

        call MPI_barrier(MPI_communicator, ierr)
        call check_mpi(ierr, MPI_communicator, 'MPI_Barrier')

    end subroutine save_dist_complex

    subroutine save_dist_real(file_name, &
                              group_name, &
                              dataset_name, &
                              access_type, &
                              N, &
                              local_i, &
                              local_i_offset, &
                              real_array, &
                              MPI_communicator)

        use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
        use HDF5
        use MPI

        implicit none

        character(len=512), intent(in) :: file_name
        character(len=512), intent(in) :: group_name
        character(len=512), intent(in) :: dataset_name
        character(len=1), intent(in) :: access_type
        integer(int64), intent(in) :: N
        integer(int32), intent(in) :: local_i
        integer(int32), intent(in) :: local_i_offset
        real(real64), dimension(local_i), intent(in) :: real_array
        integer(int32), intent(in) :: MPI_communicator

!f2py  character*512 intent(in) :: file_name
!f2py  character*512 intent(in) :: group_name
!f2py  character*512 intent(in) :: dataset_name
!f2py  character*1 intent(in) :: access_type
!f2py  integer(kind=int64) intent(in) :: n
!f2py  integer(kind=int32), optional,intent(in),check(shape(real_array, 0) == local_i),depend(real_array) :: local_i=shape(real_array, 0)
!f2py  integer(kind=int32) intent(in) :: local_i_offset
!f2py  real(kind=real64) dimension(local_i) :: real_array
!f2py  integer(kind=int32) intent(in) :: mpi_communicator

        ! HDF5 variables.
        integer(HID_T) :: file_id ! File identifier
        integer(HID_T) :: plist_id ! Property list identifier.
        integer(HID_T) :: dset_id ! Dataspace identifier.
        integer(HID_T) :: file_data_id ! Dataspace identifier in file.
        integer(HID_T) :: memspace_id ! Dataspace identifier in memory.
        integer(HID_T) :: group_id ! Group identifier in file.

        integer(HSIZE_T), dimension(1) :: ftn_dimensions
        integer(HSIZE_T), dimension(1) :: h5_dimensions

        integer(HSIZE_T), dimension(1) :: local_count
        integer(HSIZE_T), dimension(1) :: data_offset
        integer(int32) :: dataset_rank = 1

        logical :: group_exists
        logical :: dataset_exists
        logical :: file_exists
        integer(int32) :: error

        ! MPI variables
        integer(int32) :: info
        integer(int32) :: ierr

        ftn_dimensions = N
        h5_dimensions = N
        local_count = local_i
        data_offset = local_i_offset

        info = MPI_INFO_NULL

        call h5open_f(error)
        call check_hdf5(error, MPI_communicator, 'h5open_f')

        call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pcreate_f(file_access)')
        call h5pset_fapl_mpio_f(plist_id, MPI_communicator, info, error)
        call check_hdf5(error, MPI_communicator, 'h5pset_fapl_mpio_f')

        if (access_type == "a") then

            inquire (file=trim(file_name), exist=file_exists)
            if (file_exists) then

                call H5fopen_f(trim(file_name), H5F_ACC_RDWR_F, file_id, &
                               error, access_prp=plist_id)
                call check_hdf5(error, MPI_communicator, 'h5fopen_f')

            else

                call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                                 access_prp=plist_id)
                call check_hdf5(error, MPI_communicator, 'h5fcreate_f')
            end if

        elseif (access_type == "w") then

            call h5fcreate_f(trim(file_name), H5F_ACC_TRUNC_F, file_id, error, &
                             access_prp=plist_id)
            call check_hdf5(error, MPI_communicator, 'h5fcreate_f')

        end if

        call h5lexists_f(file_id, trim(group_name), group_exists, error)
        call check_hdf5(error, MPI_communicator, 'h5lexists_f')

        if (.not. group_exists) then
            call h5gcreate_f(file_id, trim(group_name), group_id, error)
            call check_hdf5(error, MPI_communicator, 'h5gcreate_f')
        else
            call h5gopen_f(file_id, trim(group_name), group_id, error)
            call check_hdf5(error, MPI_communicator, 'h5gopen_f')
        end if

        call h5pclose_f(plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pclose_f(file_access)')

        call h5screate_simple_f(dataset_rank, ftn_dimensions, file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5screate_simple_f(file_data)')

        call h5dcreate_f(file_id, trim(group_name)//trim(dataset_name), H5T_NATIVE_DOUBLE, file_data_id, &
                         dset_id, error)
        call check_hdf5(error, MPI_communicator, 'h5dcreate_f(real)')

        call h5sclose_f(file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5sclose_f(file_data_initial)')

        call h5screate_simple_f(dataset_rank, local_count, memspace_id, error)
        call check_hdf5(error, MPI_communicator, 'h5screate_simple_f(memspace)')

        call h5dget_space_f(dset_id, file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5dget_space_f')
        call h5sselect_hyperslab_f(file_data_id, H5S_SELECT_SET_F, data_offset, &
                                   local_count, error)
        call check_hdf5(error, MPI_communicator, 'h5sselect_hyperslab_f')

        call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pcreate_f(dataset_xfer)')
        call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
        call check_hdf5(error, MPI_communicator, 'h5pset_dxpl_mpio_f')

        call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, real_array, ftn_dimensions, error, &
                        file_space_id=file_data_id, mem_space_id=memspace_id, &
                        xfer_prp=plist_id)
        call check_hdf5(error, MPI_communicator, 'h5dwrite_f(real)')

        call h5sclose_f(file_data_id, error)
        call check_hdf5(error, MPI_communicator, 'h5sclose_f(file_data)')
        call h5sclose_f(memspace_id, error)
        call check_hdf5(error, MPI_communicator, 'h5sclose_f(memspace)')

        call h5dclose_f(dset_id, error)
        call check_hdf5(error, MPI_communicator, 'h5dclose_f')
        call h5pclose_f(plist_id, error)
        call check_hdf5(error, MPI_communicator, 'h5pclose_f(dataset_xfer)')

        call h5gclose_f(group_id, error)
        call check_hdf5(error, MPI_communicator, 'h5gclose_f')
        call h5fclose_f(file_id, error)
        call check_hdf5(error, MPI_communicator, 'h5fclose_f')

        call h5close_f(error)
        call check_hdf5(error, MPI_communicator, 'h5close_f')

        call MPI_barrier(MPI_communicator, ierr)
        call check_mpi(ierr, MPI_communicator, 'MPI_Barrier')

    end subroutine save_dist_real

end module io
