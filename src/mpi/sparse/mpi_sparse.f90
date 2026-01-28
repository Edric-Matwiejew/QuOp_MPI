module mpi_sparse

    use iso_fortran_env, only: sp => real32, dp => real64
    use iso_c_binding
    use mpi
    use mpi_backend
    use sparse
    use chebyshev

    implicit none

    private

    public :: sparse_propagator

    type sparse_propagator

        type(mpi_context), pointer :: context
        integer(sp), dimension(:), allocatable :: partition_table
        type(CSR) :: generator
        real(dp) :: spectral_radius

    contains

        procedure :: max_comm_size => mpi_sparse_max_comm_size
        procedure :: plan => mpi_sparse_plan
        procedure :: gen_operator => mpi_sparse_gen_operator
        procedure :: propagate => mpi_sparse_propagate
        procedure :: destroy => mpi_sparse_destroy

    end type sparse_propagator

contains

    subroutine mpi_sparse_max_comm_size(self, system_size, available_ranks, &
        constraint_ptrs, constraint_sizes, max_size, COMM)

        class(sparse_propagator), intent(inout) :: self
        integer(dp), intent(in) :: system_size
        integer(sp), intent(in) :: available_ranks
        integer(dp), intent(inout), dimension(:) :: constraint_ptrs
        integer(dp), intent(in), dimension(:) :: constraint_sizes
        integer(sp), intent(out) :: max_size
        integer(sp), intent(in) :: COMM

        max_size = available_ranks

    end subroutine mpi_sparse_max_comm_size

    subroutine mpi_sparse_plan(self, context)

        class(sparse_propagator), intent(inout) :: self
        type(mpi_context), target, intent(inout) :: context

        integer(sp) :: i, ierr, size

        self%context => context

        allocate(self%context%final_state(self%context%alloc_local))

    end subroutine mpi_sparse_plan

    subroutine mpi_sparse_gen_operator(self, array_ptrs, array_sizes)

        class(sparse_propagator), intent(inout) :: self
        integer(dp), intent(inout), dimension(:) :: array_ptrs
        integer(dp), intent(in), dimension(:) :: array_sizes 

        type(c_ptr) :: array_ptr

        ! original inputs
        integer(dp), dimension(:), pointer :: local_row_starts
        integer(dp), dimension(:), pointer :: local_col_indexes
        complex(dp), dimension(:), pointer :: local_values

        integer(sp) :: i, lb, ub, lb_elements, ub_elements
        integer(sp) :: n_arrays

        integer(sp) :: ierr, rank, flock

        ! Determine if values are provided (3 arrays) or implicit ones (2 arrays)
        n_arrays = size(array_sizes)

        ! map array pointers to original inputs
        array_ptr = transfer(array_ptrs(1), array_ptr)
        call c_f_pointer(array_ptr, local_row_starts, [array_sizes(1)])
        array_ptr = transfer(array_ptrs(2), array_ptr)
        call c_f_pointer(array_ptr, local_col_indexes, [array_sizes(2)])
        
        if (n_arrays >= 3) then
            ! Explicit values provided
            array_ptr = transfer(array_ptrs(3), array_ptr)
            call c_f_pointer(array_ptr, local_values, [array_sizes(3)])
            self%generator%has_values = .true.
        else
            ! Implicit ones - no values array
            nullify(local_values)
            self%generator%has_values = .false.
        end if

        ! moved from plan
        call MPI_Comm_size(self%context%SUBCOMM, flock, ierr)

        allocate (self%partition_table(flock + 1))

        self%partition_table(1) = 1

        call MPI_Allgather(int(self%context%local_i,sp), &
                                1, &
                                MPI_INTEGER, &
                                self%partition_table(2:flock + 1), &
                                1, &
                                MPI_INTEGER, &
                                self%context%SUBCOMM, &
                                ierr)

        do i = 2, flock + 1
            self%partition_table(i) = self%partition_table(i) + self%partition_table(i - 1)
        end do

        ! logic unchanged from original code
        call MPI_Comm_rank(self%context%SUBCOMM, rank, ierr)

        lb = self%partition_table(rank + 1)
        ub = self%partition_table(rank + 2) - 1

        lb_elements = local_row_starts(1)
        ub_elements = local_row_starts(size(local_row_starts)) - 1

        self%generator%rows = self%context%system_size
        self%generator%columns = self%context%system_size
        self%generator%row_starts(lb:ub + 1) => local_row_starts
        self%generator%col_indexes(lb_elements:ub_elements) => local_col_indexes
        
        if (self%generator%has_values) then
            self%generator%values(lb_elements:ub_elements) => local_values
            ! Note: Unlike expm, Chebyshev expects Hermitian H, not -i*H.
            ! The -i factors are in the Bessel coefficients.
        else
            nullify(self%generator%values)
        end if

        ! Setup graph communicator for efficient SpMV
        call Setup_Graph_Communications(self%generator, self%partition_table, self%context%SUBCOMM)
        
        ! Estimate spectral radius for Chebyshev expansion
        call Estimate_Spectral_Radius(self%generator, &
                                      self%partition_table, &
                                      self%context%SUBCOMM, &
                                      self%spectral_radius)

    end subroutine mpi_sparse_gen_operator

    subroutine mpi_sparse_propagate(self, ts)

        class(sparse_propagator), intent(inout) :: self
        real(dp), intent(in) :: ts(:)
        real(dp) :: t
        complex(dp), dimension(:), pointer :: ptr_tmp

        t = ts(1)

        call Chebyshev_Multiply(self%generator, &
                                self%context%initial_state, &
                                t, &
                                self%partition_table, &
                                self%context%final_state, &
                                self%context%SUBCOMM, &
                                self%spectral_radius)

        ptr_tmp => self%context%initial_state
        self%context%initial_state => self%context%final_state
        self%context%final_state => ptr_tmp

    end subroutine mpi_sparse_propagate

    subroutine mpi_sparse_destroy(self)

        class(sparse_propagator), intent(inout) :: self
        integer(sp) :: ierr

        if (allocated(self%partition_table)) then
            deallocate (self%partition_table)
        end if
        
        ! Free graph communicator resources
        call Cleanup_Graph_Communications(self%generator)

        self%generator%row_starts => null()
        self%generator%col_indexes => null()
        self%generator%values => null()

    end subroutine mpi_sparse_destroy

end module mpi_sparse
