! shafft_handler_1d.f90
!
! 1D FFT handler using SHAFFT 1D API
! Extends fft_handler_base for distributed 1D FFTs (e.g., circulant propagator)

module shafft_handler_1d

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use, intrinsic :: iso_c_binding
    use MPI
    use shafft
    use shafft_handler_base, only: fft_handler_base, fft_layout_info

    implicit none

    private

    public :: fft_1d_handler

    !---------------------------------------------------------------------------
    ! 1D FFT handler - concrete implementation of fft_handler_base
    !---------------------------------------------------------------------------
    type, extends(fft_handler_base) :: fft_1d_handler
        integer(c_size_t) :: system_size = 0
        integer(c_size_t) :: local_N = 0
        integer(c_size_t) :: local_start = 0
    contains
        procedure :: configure => fft_1d_configure
        procedure :: init => fft_1d_init
        procedure :: forward => fft_1d_forward
        procedure :: backward => fft_1d_backward
        procedure :: destroy => fft_1d_destroy
        final :: fft_1d_finalize
    end type fft_1d_handler

contains

    !---------------------------------------------------------------------------
    ! configure: Query SHAFFT for 1D partitioning
    !---------------------------------------------------------------------------
    subroutine fft_1d_configure(self, dims, DEVCOMM, layout_out, ierr)
        class(fft_1d_handler), intent(inout) :: self
        integer(int32), dimension(:), intent(in) :: dims
        integer(int32), intent(in) :: DEVCOMM
        type(fft_layout_info), intent(out) :: layout_out
        integer(int32), intent(out) :: ierr

        integer(c_size_t) :: local_N, local_start, local_alloc_size
        integer(c_int) :: ierr_c

        ierr = 0
        self%n_dims = 1

        if (allocated(self%tensor_dims)) deallocate (self%tensor_dims)
        allocate (self%tensor_dims(1))
        self%tensor_dims(1) = dims(1)
        self%system_size = int(dims(1), c_size_t)

        ! Initialize output
        local_N = 0
        local_start = 0
        local_alloc_size = 0

        if (DEVCOMM /= MPI_COMM_NULL) then
            call shafftConfiguration1D(self%system_size, &
                                       local_N, local_start, local_alloc_size, &
                                       SHAFFT_Z2Z, DEVCOMM, ierr_c)
            if (ierr_c /= SHAFFT_SUCCESS) then
                ierr = int(ierr_c, int32)
                return
            end if
        end if

        ! Store in layout
        layout_out%local_i = local_N
        layout_out%local_i_offset = local_start
        layout_out%transformed_local_i = local_N ! Same for 1D
        layout_out%transformed_local_i_offset = local_start
        layout_out%alloc_size = local_alloc_size

        self%layout = layout_out
        self%initialized = .true.

    end subroutine fft_1d_configure

    !---------------------------------------------------------------------------
    ! init: Create and initialize SHAFFT 1D plan
    !---------------------------------------------------------------------------
    subroutine fft_1d_init(self, dims, local_i, local_i_offset, DEVCOMM, ierr)
        class(fft_1d_handler), intent(inout) :: self
        integer(int32), dimension(:), intent(in) :: dims
        integer(c_size_t), intent(in) :: local_i
        integer(c_size_t), intent(in) :: local_i_offset
        integer(int32), intent(in) :: DEVCOMM
        integer(int32), intent(out) :: ierr

        integer(c_int) :: ierr_c

        ierr = 0
        self%n_dims = 1

        if (allocated(self%tensor_dims)) deallocate (self%tensor_dims)
        allocate (self%tensor_dims(1))
        self%tensor_dims(1) = dims(1)
        self%system_size = int(dims(1), c_size_t)
        self%local_N = local_i
        self%local_start = local_i_offset

        ! Store layout info
        self%layout%local_i = local_i
        self%layout%local_i_offset = local_i_offset
        self%layout%transformed_local_i = local_i
        self%layout%transformed_local_i_offset = local_i_offset

        ! Handle trivial cases
        if (self%system_size <= 1 .or. local_i == 0) then
            self%initialized = .true.
            self%planned = .true.
            return
        end if

        ! Create SHAFFT 1D plan
        call shafft1DCreate(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        ! Initialize SHAFFT 1D plan
        call shafft1DInit(self%shafft_plan, self%system_size, &
                          self%local_N, self%local_start, &
                          SHAFFT_Z2Z, DEVCOMM, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        ! Create FFT plans
        call shafftPlan(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        ! Get allocation size
        call shafftGetAllocSize(self%shafft_plan, self%layout%alloc_size, ierr_c)

        self%initialized = .true.
        self%planned = .true.

    end subroutine fft_1d_init

    !---------------------------------------------------------------------------
    ! forward: Execute forward FFT with normalization
    !---------------------------------------------------------------------------
    subroutine fft_1d_forward(self, state, work, ierr)
        class(fft_1d_handler), intent(inout) :: self
        complex(real64), pointer, intent(inout) :: state(:)
        complex(real64), pointer, intent(inout) :: work(:)
        integer(int32), intent(out) :: ierr

        integer(c_int) :: ierr_c
        integer(c_size_t) :: alloc_size
        complex(real64), pointer :: state_ptr(:), work_ptr(:), tmp(:)

        ierr = 0

        if (.not. self%planned .or. .not. c_associated(self%shafft_plan)) then
            return ! No FFT needed (trivial case)
        end if

        ! SHAFFT API requires pointer arrays
        state_ptr => state
        work_ptr => work

        call shafftSetBuffers(self%shafft_plan, state_ptr, work_ptr, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        call shafftExecute(self%shafft_plan, SHAFFT_FORWARD, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        call shafftNormalize(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        ! Get updated buffer pointers (may change after transpose)
        call shafftGetAllocSize(self%shafft_plan, alloc_size, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if
        call shafftGetBuffers(self%shafft_plan, alloc_size, state_ptr, work_ptr, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

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

    end subroutine fft_1d_forward

    !---------------------------------------------------------------------------
    ! backward: Execute backward FFT with normalization
    !---------------------------------------------------------------------------
    subroutine fft_1d_backward(self, state, work, ierr)
        class(fft_1d_handler), intent(inout) :: self
        complex(real64), pointer, intent(inout) :: state(:)
        complex(real64), pointer, intent(inout) :: work(:)
        integer(int32), intent(out) :: ierr

        integer(c_int) :: ierr_c
        integer(c_size_t) :: alloc_size
        complex(real64), pointer :: state_ptr(:), work_ptr(:), tmp(:)

        ierr = 0

        if (.not. self%planned .or. .not. c_associated(self%shafft_plan)) then
            return ! No FFT needed (trivial case)
        end if

        ! SHAFFT API requires pointer arrays
        state_ptr => state
        work_ptr => work

        call shafftSetBuffers(self%shafft_plan, state_ptr, work_ptr, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        call shafftExecute(self%shafft_plan, SHAFFT_BACKWARD, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        call shafftNormalize(self%shafft_plan, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

        ! Get updated buffer pointers
        call shafftGetAllocSize(self%shafft_plan, alloc_size, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if
        call shafftGetBuffers(self%shafft_plan, alloc_size, state_ptr, work_ptr, ierr_c)
        if (ierr_c /= SHAFFT_SUCCESS) then
            ierr = int(ierr_c, int32)
            return
        end if

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

    end subroutine fft_1d_backward

    !---------------------------------------------------------------------------
    ! destroy: Clean up SHAFFT plan
    !---------------------------------------------------------------------------
    subroutine fft_1d_destroy(self)
        class(fft_1d_handler), intent(inout) :: self

        integer(c_int) :: ierr_c

        if (c_associated(self%shafft_plan)) then
            call shafftDestroy(self%shafft_plan, ierr_c)
            self%shafft_plan = c_null_ptr
        end if

        if (allocated(self%tensor_dims)) deallocate (self%tensor_dims)

        self%initialized = .false.
        self%planned = .false.
        self%system_size = 0
        self%local_N = 0
        self%local_start = 0

    end subroutine fft_1d_destroy

    !---------------------------------------------------------------------------
    ! finalize: FINAL procedure for automatic cleanup (F2003)
    !---------------------------------------------------------------------------
    subroutine fft_1d_finalize(self)
        type(fft_1d_handler), intent(inout) :: self
        call self%destroy()
    end subroutine fft_1d_finalize

end module shafft_handler_1d
