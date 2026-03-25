! shafft_handler_nd.f90
!
! N-D FFT handler using SHAFFT ND API (Cartesian decomposition)
! Extends fft_handler_base for distributed N-D FFTs (e.g., composite/momentum propagators)
!
! Note: SHAFFT ND requires at least 2 dimensions for Cartesian decomposition

module shafft_handler_nd

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use, intrinsic :: iso_c_binding
    use MPI
    use hipfort
    use hipfort_check
    use shafft
    use shafft_handler_base, only: fft_handler_base, fft_layout_info
    use shafft_comm_dims_utils, only: comm_dims_contiguous_prefer_low_nda

    implicit none

    private

    public :: fft_nd_handler

    !---------------------------------------------------------------------------
    ! N-D FFT handler - concrete implementation of fft_handler_base
    !---------------------------------------------------------------------------
    type, extends(fft_handler_base) :: fft_nd_handler
        integer(int32), dimension(:), allocatable :: COMM_DIMS ! SHAFFT Cartesian grid
        integer(c_size_t), dimension(:), allocatable :: subsize ! Local tensor size (initial)
        integer(c_size_t), dimension(:), allocatable :: offset ! Local tensor offset (initial)
        integer(int32), dimension(:), allocatable :: strides ! Global strides (initial layout)
        integer(c_size_t), dimension(:), allocatable :: subsize_transformed
        integer(c_size_t), dimension(:), allocatable :: offset_transformed
        integer(int32), dimension(:), allocatable :: strides_transformed
    contains
        procedure :: configure => fft_nd_configure
        procedure :: init => fft_nd_init
        procedure :: forward => fft_nd_forward
        procedure :: backward => fft_nd_backward
        procedure :: destroy => fft_nd_destroy

        ! Additional ND-specific methods
        procedure :: get_subsize => fft_nd_get_subsize
        procedure :: get_offset => fft_nd_get_offset
        procedure :: get_subsize_transformed => fft_nd_get_subsize_transformed
        procedure :: get_offset_transformed => fft_nd_get_offset_transformed
        procedure :: get_strides => fft_nd_get_strides
        procedure :: get_strides_transformed => fft_nd_get_strides_transformed

        final :: fft_nd_finalize
    end type fft_nd_handler

contains

    !---------------------------------------------------------------------------
    ! configure: Query SHAFFT for ND partitioning
    !---------------------------------------------------------------------------
    subroutine fft_nd_configure(self, dims, DEVCOMM, layout_out, ierr)
        class(fft_nd_handler), intent(inout) :: self
        integer(int32), dimension(:), intent(in) :: dims
        integer(int32), intent(in) :: DEVCOMM
        type(fft_layout_info), intent(out) :: layout_out
        integer(int32), intent(out) :: ierr

        integer(c_size_t), dimension(:), allocatable :: subsize, offset
        integer(c_size_t), dimension(:), allocatable :: transformed_subsize, transformed_offset
        integer(int32), dimension(:), allocatable :: strides, transformed_strides
        integer(int32) :: total_gpus, nda
        integer(int32) :: devcomm_size, used_ranks
        integer(int32) :: ierr_mpi, ierr_dims
        integer(c_int) :: ierr_c
        integer(int64) :: free_mem, total_mem
        integer(c_size_t) :: total_elements
        integer(int32), allocatable :: requested_comm_dims(:)
        type(c_ptr) :: temp_plan

        ierr = 0
        self%n_dims = size(dims)
        temp_plan = c_null_ptr

        ! Store tensor dimensions
        if (allocated(self%tensor_dims)) deallocate (self%tensor_dims)
        allocate (self%tensor_dims(self%n_dims))
        self%tensor_dims = dims

        ! Allocate working arrays
        allocate (subsize(self%n_dims), offset(self%n_dims))
        allocate (transformed_subsize(self%n_dims), transformed_offset(self%n_dims))
        allocate (strides(self%n_dims), transformed_strides(self%n_dims))
        subsize(:) = 0
        offset(:) = 0
        transformed_subsize(:) = 0
        transformed_offset(:) = 0
        strides(:) = 0
        transformed_strides(:) = 0

        ! Initialize output
        layout_out%local_i = 0
        layout_out%local_i_offset = 0
        layout_out%transformed_local_i = 0
        layout_out%transformed_local_i_offset = 0
        layout_out%alloc_size = 0

        if (DEVCOMM /= MPI_COMM_NULL) then
            ! Allocate COMM_DIMS for SHAFFT grid
            if (allocated(self%COMM_DIMS)) deallocate (self%COMM_DIMS)
            allocate (self%COMM_DIMS(self%n_dims))
            self%COMM_DIMS(:) = 1
            allocate (requested_comm_dims(self%n_dims))

            ! Get available GPU memory
            call hipCheck(hipMemGetInfo(free_mem, total_mem))
            free_mem = int(total_mem / 2.5_real64, int64) ! Account for state + work + observables

            call MPI_Comm_size(DEVCOMM, devcomm_size, ierr_mpi)
            if (ierr_mpi /= MPI_SUCCESS) then
                ierr = SHAFFT_ERR_MPI
                deallocate (subsize, offset)
                return
            end if

            call comm_dims_contiguous_prefer_low_nda(devcomm_size, dims, &
                                                     self%COMM_DIMS, nda, used_ranks, ierr_dims)
            if (ierr_dims /= 0) then
                ierr = SHAFFT_ERR_INVALID_DIM
                deallocate (subsize, offset)
                return
            end if
            if (used_ranks < 1) then
                ierr = SHAFFT_ERR_INVALID_DECOMP
                deallocate (subsize, offset)
                return
            end if
            requested_comm_dims = self%COMM_DIMS

            call shafftConfigurationND(dims, &
                                       SHAFFT_Z2Z, &
                                       self%COMM_DIMS, &
                                       nda, &
                                       subsize, &
                                       offset, &
                                       total_gpus, &
                                       SHAFFT_MINIMIZE_NDA, &
                                       int(free_mem, c_size_t), &
                                       DEVCOMM, &
                                       ierr)
            if (ierr /= SHAFFT_SUCCESS) then
                deallocate (subsize, offset)
                return
            end if
            if (any(self%COMM_DIMS /= requested_comm_dims)) then
                ierr = SHAFFT_ERR_INVALID_DECOMP
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if

            call shafftNDCreate(temp_plan, ierr_c)
            if (ierr_c /= SHAFFT_SUCCESS) then
                ierr = int(ierr_c, int32)
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if
            call shafftNDInit(temp_plan, self%COMM_DIMS, dims, &
                              SHAFFT_Z2Z, DEVCOMM, SHAFFT_LAYOUT_REDISTRIBUTED, ierr_c)
            if (ierr_c /= SHAFFT_SUCCESS) then
                ierr = int(ierr_c, int32)
                call shafftDestroy(temp_plan, ierr_c)
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if
            call shafftPlan(temp_plan, ierr_c)
            if (ierr_c /= SHAFFT_SUCCESS) then
                ierr = int(ierr_c, int32)
                call shafftDestroy(temp_plan, ierr_c)
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if

            call query_nd_layout_state(temp_plan, dims, SHAFFT_TENSOR_LAYOUT_INITIAL, &
                                       subsize, offset, strides, &
                                       layout_out%local_i, layout_out%local_i_offset, ierr)
            if (ierr /= SHAFFT_SUCCESS) then
                call shafftDestroy(temp_plan, ierr_c)
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if

            call query_nd_layout_state(temp_plan, dims, SHAFFT_TENSOR_LAYOUT_REDISTRIBUTED, &
                                       transformed_subsize, transformed_offset, transformed_strides, &
                                       layout_out%transformed_local_i, layout_out%transformed_local_i_offset, ierr)
            if (ierr /= SHAFFT_SUCCESS) then
                call shafftDestroy(temp_plan, ierr_c)
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if

            call shafftGetAllocSize(temp_plan, layout_out%alloc_size, ierr_c)
            if (ierr_c /= SHAFFT_SUCCESS) then
                ierr = int(ierr_c, int32)
                call shafftDestroy(temp_plan, ierr_c)
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if

            call shafftDestroy(temp_plan, ierr_c)
            temp_plan = c_null_ptr

            ! Contiguity / bounds check
            total_elements = int(product(int(dims, c_size_t)), c_size_t)
            if (layout_out%local_i_offset + layout_out%local_i > total_elements) then
                ierr = SHAFFT_ERR_INVALID_DECOMP
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if
            if (layout_out%transformed_local_i_offset + layout_out%transformed_local_i > total_elements) then
                ierr = SHAFFT_ERR_INVALID_DECOMP
                deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)
                return
            end if
        end if

        deallocate (subsize, offset, transformed_subsize, transformed_offset, strides, transformed_strides)

        self%layout = layout_out
        self%initialized = .true.

    end subroutine fft_nd_configure

    !---------------------------------------------------------------------------
    ! init: Create and initialize SHAFFT ND plan
    !---------------------------------------------------------------------------
    subroutine fft_nd_init(self, dims, local_i, local_i_offset, DEVCOMM, ierr)
        class(fft_nd_handler), intent(inout) :: self
        integer(int32), dimension(:), intent(in) :: dims
        integer(c_size_t), intent(in) :: local_i
        integer(c_size_t), intent(in) :: local_i_offset
        integer(int32), intent(in) :: DEVCOMM
        integer(int32), intent(out) :: ierr

        integer(c_size_t) :: total_elements
        integer(int32) :: devcomm_size, used_ranks, ierr_dims, ierr_mpi, nda_guess

        ierr = 0
        self%n_dims = size(dims)

        ! Check minimum dimensionality
        if (self%n_dims < 2) then
            ierr = -1 ! SHAFFT ND requires at least 2 dimensions
            return
        end if

        ! Store tensor dimensions
        if (allocated(self%tensor_dims)) deallocate (self%tensor_dims)
        allocate (self%tensor_dims(self%n_dims))
        self%tensor_dims = dims

        ! Store layout from context
        self%layout%local_i = local_i
        self%layout%local_i_offset = local_i_offset

        ! Allocate arrays (preserve COMM_DIMS from configure if present)
        if (.not. allocated(self%COMM_DIMS)) then
            allocate (self%COMM_DIMS(self%n_dims))
            self%COMM_DIMS(:) = 1
        else if (size(self%COMM_DIMS) /= self%n_dims) then
            deallocate (self%COMM_DIMS)
            allocate (self%COMM_DIMS(self%n_dims))
            self%COMM_DIMS(:) = 1
        end if

        if (all(self%COMM_DIMS == 1)) then
            if (DEVCOMM == MPI_COMM_NULL) then
                ierr = SHAFFT_ERR_INVALID_COMM
                return
            end if
            call MPI_Comm_size(DEVCOMM, devcomm_size, ierr_mpi)
            if (ierr_mpi /= MPI_SUCCESS) then
                ierr = SHAFFT_ERR_MPI
                return
            end if
            call comm_dims_contiguous_prefer_low_nda(devcomm_size, dims, &
                                                     self%COMM_DIMS, nda_guess, used_ranks, ierr_dims)
            if (ierr_dims /= 0) then
                ierr = SHAFFT_ERR_INVALID_DIM
                return
            end if
            if (used_ranks < 1) then
                ierr = SHAFFT_ERR_INVALID_DECOMP
                return
            end if
        end if

        if (allocated(self%subsize)) deallocate (self%subsize)
        if (allocated(self%offset)) deallocate (self%offset)
        if (allocated(self%strides)) deallocate (self%strides)
        if (allocated(self%subsize_transformed)) deallocate (self%subsize_transformed)
        if (allocated(self%offset_transformed)) deallocate (self%offset_transformed)
        if (allocated(self%strides_transformed)) deallocate (self%strides_transformed)

        allocate (self%subsize(self%n_dims))
        allocate (self%offset(self%n_dims))
        allocate (self%strides(self%n_dims))
        allocate (self%subsize_transformed(self%n_dims))
        allocate (self%offset_transformed(self%n_dims))
        allocate (self%strides_transformed(self%n_dims))

        ! Create SHAFFT ND plan
        call shafftNDCreate(self%shafft_plan, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        ! Initialize SHAFFT ND plan
        call shafftNDInit(self%shafft_plan, self%COMM_DIMS, dims, &
                          SHAFFT_Z2Z, DEVCOMM, SHAFFT_LAYOUT_REDISTRIBUTED, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        ! Create FFT plans
        call shafftPlan(self%shafft_plan, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        call query_nd_layout_state(self%shafft_plan, dims, SHAFFT_TENSOR_LAYOUT_INITIAL, &
                                   self%subsize, self%offset, self%strides, &
                                   self%layout%local_i, self%layout%local_i_offset, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        call query_nd_layout_state(self%shafft_plan, dims, SHAFFT_TENSOR_LAYOUT_REDISTRIBUTED, &
                                   self%subsize_transformed, self%offset_transformed, self%strides_transformed, &
                                   self%layout%transformed_local_i, self%layout%transformed_local_i_offset, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        ! Debug bounds/contiguity checks on linearised slabs
        total_elements = int(product(int(dims, c_size_t)), c_size_t)
        if (self%layout%local_i_offset + self%layout%local_i > total_elements) then
            ierr = SHAFFT_ERR_INVALID_DECOMP
            return
        end if
        if (self%layout%transformed_local_i_offset + self%layout%transformed_local_i > total_elements) then
            ierr = SHAFFT_ERR_INVALID_DECOMP
            return
        end if

        ! Get allocation size
        call shafftGetAllocSize(self%shafft_plan, self%layout%alloc_size, ierr)

        self%initialized = .true.
        self%planned = .true.

    end subroutine fft_nd_init

    !---------------------------------------------------------------------------
    ! forward: Execute forward FFT with normalization
    !---------------------------------------------------------------------------
    subroutine fft_nd_forward(self, state, work, ierr)
        class(fft_nd_handler), intent(inout) :: self
        complex(real64), pointer, intent(inout) :: state(:)
        complex(real64), pointer, intent(inout) :: work(:)
        integer(int32), intent(out) :: ierr

        integer(c_size_t) :: alloc_size
        complex(real64), pointer :: state_ptr(:), work_ptr(:), tmp(:)

        ierr = 0

        if (.not. self%planned .or. .not. c_associated(self%shafft_plan)) then
            return
        end if

        ! SHAFFT API requires pointer arrays
        state_ptr => state
        work_ptr => work

        call shafftSetBuffers(self%shafft_plan, state_ptr, work_ptr, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        call shafftExecute(self%shafft_plan, SHAFFT_FORWARD, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        call shafftNormalize(self%shafft_plan, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        ! Get updated buffer pointers
        call shafftGetAllocSize(self%shafft_plan, alloc_size, ierr)
        if (ierr /= SHAFFT_SUCCESS) return
        call shafftGetBuffers(self%shafft_plan, alloc_size, state_ptr, work_ptr, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        ! SHAFFT may swap data/work buffers after execute; keep caller pointers in sync.
        if (.not. c_associated(c_loc(state_ptr), c_loc(state))) then
            if (.not. c_associated(c_loc(state_ptr), c_loc(work)) .or. &
                .not. c_associated(c_loc(work_ptr), c_loc(state))) then
                ierr = SHAFFT_ERR_INVALID_LAYOUT
                return
            end if
            tmp => state
            state => work
            work => tmp
        end if

    end subroutine fft_nd_forward

    !---------------------------------------------------------------------------
    ! backward: Execute backward FFT with normalization
    !---------------------------------------------------------------------------
    subroutine fft_nd_backward(self, state, work, ierr)
        class(fft_nd_handler), intent(inout) :: self
        complex(real64), pointer, intent(inout) :: state(:)
        complex(real64), pointer, intent(inout) :: work(:)
        integer(int32), intent(out) :: ierr

        integer(c_size_t) :: alloc_size
        complex(real64), pointer :: state_ptr(:), work_ptr(:), tmp(:)

        ierr = 0

        if (.not. self%planned .or. .not. c_associated(self%shafft_plan)) then
            return
        end if

        ! SHAFFT API requires pointer arrays
        state_ptr => state
        work_ptr => work

        call shafftSetBuffers(self%shafft_plan, state_ptr, work_ptr, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        call shafftExecute(self%shafft_plan, SHAFFT_BACKWARD, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        call shafftNormalize(self%shafft_plan, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        ! Get updated buffer pointers
        call shafftGetAllocSize(self%shafft_plan, alloc_size, ierr)
        if (ierr /= SHAFFT_SUCCESS) return
        call shafftGetBuffers(self%shafft_plan, alloc_size, state_ptr, work_ptr, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        if (.not. c_associated(c_loc(state_ptr), c_loc(state))) then
            if (.not. c_associated(c_loc(state_ptr), c_loc(work)) .or. &
                .not. c_associated(c_loc(work_ptr), c_loc(state))) then
                ierr = SHAFFT_ERR_INVALID_LAYOUT
                return
            end if
            tmp => state
            state => work
            work => tmp
        end if

    end subroutine fft_nd_backward

    !---------------------------------------------------------------------------
    ! destroy: Clean up SHAFFT plan and arrays
    !---------------------------------------------------------------------------
    subroutine fft_nd_destroy(self)
        class(fft_nd_handler), intent(inout) :: self

        integer(c_int) :: ierr_c

        if (c_associated(self%shafft_plan)) then
            call shafftDestroy(self%shafft_plan, ierr_c)
            self%shafft_plan = c_null_ptr
        end if

        if (allocated(self%tensor_dims)) deallocate (self%tensor_dims)
        if (allocated(self%COMM_DIMS)) deallocate (self%COMM_DIMS)
        if (allocated(self%subsize)) deallocate (self%subsize)
        if (allocated(self%offset)) deallocate (self%offset)
        if (allocated(self%strides)) deallocate (self%strides)
        if (allocated(self%subsize_transformed)) deallocate (self%subsize_transformed)
        if (allocated(self%offset_transformed)) deallocate (self%offset_transformed)
        if (allocated(self%strides_transformed)) deallocate (self%strides_transformed)

        self%initialized = .false.
        self%planned = .false.
        self%n_dims = 0

    end subroutine fft_nd_destroy

    !---------------------------------------------------------------------------
    ! Accessor methods for ND-specific data
    !---------------------------------------------------------------------------

    function fft_nd_get_subsize(self) result(subsize)
        class(fft_nd_handler), intent(in) :: self
        integer(c_size_t), dimension(:), allocatable :: subsize
        if (allocated(self%subsize)) then
            allocate (subsize(size(self%subsize)))
            subsize = self%subsize
        end if
    end function fft_nd_get_subsize

    function fft_nd_get_offset(self) result(offset)
        class(fft_nd_handler), intent(in) :: self
        integer(c_size_t), dimension(:), allocatable :: offset
        if (allocated(self%offset)) then
            allocate (offset(size(self%offset)))
            offset = self%offset
        end if
    end function fft_nd_get_offset

    function fft_nd_get_subsize_transformed(self) result(subsize)
        class(fft_nd_handler), intent(in) :: self
        integer(c_size_t), dimension(:), allocatable :: subsize
        if (allocated(self%subsize_transformed)) then
            allocate (subsize(size(self%subsize_transformed)))
            subsize = self%subsize_transformed
        end if
    end function fft_nd_get_subsize_transformed

    function fft_nd_get_offset_transformed(self) result(offset)
        class(fft_nd_handler), intent(in) :: self
        integer(c_size_t), dimension(:), allocatable :: offset
        if (allocated(self%offset_transformed)) then
            allocate (offset(size(self%offset_transformed)))
            offset = self%offset_transformed
        end if
    end function fft_nd_get_offset_transformed

    function fft_nd_get_strides(self) result(strides)
        class(fft_nd_handler), intent(in) :: self
        integer(int32), dimension(:), allocatable :: strides
        if (allocated(self%strides)) then
            allocate (strides(size(self%strides)))
            strides = self%strides
        end if
    end function fft_nd_get_strides

    function fft_nd_get_strides_transformed(self) result(strides)
        class(fft_nd_handler), intent(in) :: self
        integer(int32), dimension(:), allocatable :: strides
        if (allocated(self%strides_transformed)) then
            allocate (strides(size(self%strides_transformed)))
            strides = self%strides_transformed
        end if
    end function fft_nd_get_strides_transformed

    subroutine query_nd_layout_state(plan, dims, layout_kind, subsize_out, offset_out, strides_out, &
                                     local_i_out, local_i_offset_out, ierr)
        type(c_ptr), intent(in) :: plan
        integer(int32), intent(in) :: dims(:)
        integer(c_int), intent(in) :: layout_kind
        integer(c_size_t), intent(out) :: subsize_out(:), offset_out(:)
        integer(int32), intent(out) :: strides_out(:)
        integer(c_size_t), intent(out) :: local_i_out, local_i_offset_out
        integer(int32), intent(out) :: ierr

        integer(int32) :: i
        integer(int32), allocatable :: ca(:), da(:)

        ierr = SHAFFT_SUCCESS

        call shafftGetLayout(plan, subsize_out, offset_out, layout_kind, ierr)
        if (ierr /= SHAFFT_SUCCESS) return

        allocate (ca(size(dims)), da(size(dims)))
        ca = -1
        da = -1
        call shafftGetAxes(plan, ca, da, layout_kind, ierr)
        if (ierr /= SHAFFT_SUCCESS) then
            deallocate (ca, da)
            return
        end if

        call compute_layout_strides_from_axes(dims, ca, da, strides_out, ierr)
        deallocate (ca, da)
        if (ierr /= SHAFFT_SUCCESS) return

        local_i_out = int(product(real(subsize_out, real64)), c_size_t)
        local_i_offset_out = 0_c_size_t
        do i = 1, size(dims)
            local_i_offset_out = local_i_offset_out + offset_out(i) * int(strides_out(i), c_size_t)
        end do
    end subroutine query_nd_layout_state

    subroutine compute_layout_strides_from_axes(dims, ca, da, strides_out, ierr)
        integer(int32), intent(in) :: dims(:)
        integer(int32), intent(in) :: ca(:), da(:)
        integer(int32), intent(out) :: strides_out(:)
        integer(int32), intent(out) :: ierr

        integer(int32) :: axis, i, n_dims, n_order
        integer(int32), allocatable :: axis_order(:)
        logical, allocatable :: used(:)

        ierr = SHAFFT_SUCCESS
        n_dims = size(dims)

        if (size(strides_out) /= n_dims) then
            ierr = SHAFFT_ERR_INVALID_DIM
            return
        end if

        allocate (axis_order(n_dims), used(n_dims))
        used = .false.
        n_order = 0

        ! SHAFFT reports ca/da in innermost->outermost stride order.
        ! Build a complete axis order, then derive per-axis strides from it.
        do i = 1, size(ca)
            axis = ca(i) + 1 ! C index -> Fortran index
            if (axis >= 1 .and. axis <= n_dims) then
                if (.not. used(axis)) then
                    n_order = n_order + 1
                    axis_order(n_order) = axis
                    used(axis) = .true.
                end if
            end if
        end do

        do i = 1, size(da)
            axis = da(i) + 1 ! C index -> Fortran index
            if (axis >= 1 .and. axis <= n_dims) then
                if (.not. used(axis)) then
                    n_order = n_order + 1
                    axis_order(n_order) = axis
                    used(axis) = .true.
                end if
            end if
        end do

        ! Fallback for any axis not explicitly listed by SHAFFT.
        do axis = 1, n_dims
            if (.not. used(axis)) then
                n_order = n_order + 1
                axis_order(n_order) = axis
                used(axis) = .true.
            end if
        end do

        if (n_order /= n_dims) then
            ierr = SHAFFT_ERR_INTERNAL
            return
        end if

        strides_out = 0
        strides_out(axis_order(1)) = 1
        do i = 2, n_dims
            strides_out(axis_order(i)) = strides_out(axis_order(i - 1)) * dims(axis_order(i - 1))
        end do

    end subroutine compute_layout_strides_from_axes

    !---------------------------------------------------------------------------
    ! finalize: FINAL procedure for automatic cleanup (F2003)
    !---------------------------------------------------------------------------
    subroutine fft_nd_finalize(self)
        type(fft_nd_handler), intent(inout) :: self
        call self%destroy()
    end subroutine fft_nd_finalize

end module shafft_handler_nd
