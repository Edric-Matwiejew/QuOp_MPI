! test_sparse_cpu_spmv.f90
!
! Native MPI test for the CPU sparse-matrix-vector product (`spmv_graph` /
! `spmv_cpu`) used by the sparse propagator's Chebyshev path.
!
! Each subtest builds a deterministic, non-symmetric CSR with non-uniform
! complex values and a non-uniform complex initial vector, and compares
! `spmv_graph(A, x, ..., scalar)` against an independently computed
! reference for the locally-owned rows.  Subtests target distinct edge
! cases:
!
!   * basic         - 5 column offsets per row, scalar = (1, 0)
!   * scalar        - same matrix, scalar = (0.7, -0.4)  (covers alpha*A*x)
!   * empty_rows    - rows with zero nonzeros mixed with dense rows
!   * wide_stencil  - 11 column offsets per row spanning multiple
!                     neighbour ranks
!
! All subtests scale to an arbitrary process count.

program test_sparse_cpu_spmv

    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use sparse, only: CSR, setup_graph_communications, spmv_graph, &
                      cleanup_graph_communications
    use MPI

    implicit none

    real(real64), parameter :: tol = 1.0e-12_real64

    integer :: rank, nprocs, ierr
    integer :: failures, local_failures

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    if (rank == 0) then
        write (*, '(A,I0)') "test_sparse_cpu_spmv: ranks=", nprocs
    end if

    failures = 0

    call run_basic(local_failures);        failures = failures + local_failures
    call run_scalar(local_failures);       failures = failures + local_failures
    call run_empty_rows(local_failures);   failures = failures + local_failures
    call run_wide_stencil(local_failures); failures = failures + local_failures

    call MPI_Finalize(ierr)

    if (failures > 0) then
        if (rank == 0) write (*, '(A,I0,A)') "FAILED (", failures, " subtest(s))"
        call exit(1)
    else
        if (rank == 0) write (*, *) "PASSED"
        call exit(0)
    end if

contains

    !--------------------------------------------------------------------------
    ! Driver: build an N x N CSR with column offsets `offsets` per row
    ! (rows where empty_pattern is set are skipped), run spmv_graph with
    ! scalar `alpha`, and report whether the maximum difference from the
    ! serial reference exceeds `tol`.
    !--------------------------------------------------------------------------
    subroutine run_subtest(label, N, offsets, alpha, empty_pattern, fail_count)
        character(len=*), intent(in) :: label
        integer(int64), intent(in) :: N
        integer, intent(in) :: offsets(:)
        complex(real64), intent(in) :: alpha
        integer, intent(in) :: empty_pattern   ! 0 = none, 1 = every 4th row empty
        integer, intent(out) :: fail_count

        integer(int64), allocatable :: partition_table(:)
        integer(int64) :: lb, ub, n_local, i64
        type(CSR) :: A
        complex(real64), allocatable :: x_full(:)
        complex(real64), allocatable, target :: x_local(:)
        complex(real64), allocatable :: y_local(:), y_ref(:)
        real(real64) :: max_err, global_max_err

        fail_count = 0

        call build_partition_table(N, nprocs, partition_table)
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        allocate (x_full(N))
        do i64 = 1, N
            x_full(i64) = cmplx(sin(0.7_real64 * real(i64 - 1, real64) + 0.1_real64), &
                                cos(0.3_real64 * real(i64 - 1, real64) - 0.2_real64), &
                                real64)
        end do

        allocate (x_local(max(n_local, 1_int64)))
        if (n_local > 0) x_local(1:n_local) = x_full(lb:ub)

        ! Compute reference BEFORE building/calling spmv_graph: spmv_graph
        ! mutates A%col_indexes in place on CPU build to encode halo
        ! offsets, so we must compute y_ref from x_full + entry_value
        ! (independent of A's storage).
        allocate (y_ref(max(n_local, 1_int64)))
        if (n_local > 0) call serial_matvec(N, lb, ub, offsets, x_full, &
                                            empty_pattern, y_ref)

        call build_local_csr(A, lb, ub, N, offsets, empty_pattern)

        allocate (y_local(max(n_local, 1_int64)))
        y_local = (0.0_real64, 0.0_real64)

        if (n_local > 0) then
            call spmv_graph(A, x_local, partition_table, rank, y_local, &
                            alpha, MPI_COMM_WORLD)
        end if

        max_err = 0.0_real64
        do i64 = 1, n_local
            max_err = max(max_err, abs(y_local(i64) - alpha * y_ref(i64)))
        end do

        call MPI_Allreduce(max_err, global_max_err, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MPI_COMM_WORLD, ierr)

        if (rank == 0) then
            write (*, '(A,A,A,ES10.2)') "  [", trim(label), &
                "] max |y - alpha*A*x| = ", global_max_err
        end if

        if (global_max_err > tol) fail_count = 1

        call cleanup_graph_communications(A)
        if (associated(A%row_starts)) deallocate (A%row_starts)
        if (associated(A%col_indexes)) deallocate (A%col_indexes)
        if (associated(A%values)) deallocate (A%values)

        deallocate (x_local, x_full, y_local, y_ref, partition_table)
    end subroutine run_subtest

    subroutine run_basic(fc)
        integer, intent(out) :: fc
        integer :: offsets(5)
        offsets = [0, 1, 3, -2, 7]
        call run_subtest("basic", 31_int64 * int(nprocs, int64) + 5_int64, &
                         offsets, (1.0_real64, 0.0_real64), 0, fc)
    end subroutine run_basic

    subroutine run_scalar(fc)
        integer, intent(out) :: fc
        integer :: offsets(5)
        offsets = [0, 1, 3, -2, 7]
        call run_subtest("scalar", 31_int64 * int(nprocs, int64) + 5_int64, &
                         offsets, (0.7_real64, -0.4_real64), 0, fc)
    end subroutine run_scalar

    subroutine run_empty_rows(fc)
        integer, intent(out) :: fc
        integer :: offsets(5)
        offsets = [0, 1, 3, -2, 7]
        ! empty_pattern = 1 zeros out every 4th global row.  This exercises
        ! rows with row_starts(i) == row_starts(i+1) (no nonzeros), where
        ! diag_lo > diag_hi for that row and all three SpMV segments are
        ! empty.
        call run_subtest("empty_rows", 31_int64 * int(nprocs, int64) + 5_int64, &
                         offsets, (0.5_real64, 0.25_real64), 1, fc)
    end subroutine run_empty_rows

    subroutine run_wide_stencil(fc)
        integer, intent(out) :: fc
        integer :: offsets(11)
        ! 11 offsets, including offsets large enough to span more than one
        ! neighbouring rank when nprocs is small.  Negative and positive
        ! offsets are deliberately unbalanced so the pattern is not its
        ! own transpose.
        offsets = [-80, -17, -5, -2, -1, 0, 1, 4, 9, 23, 80]
        call run_subtest("wide_stencil", 67_int64 * int(nprocs, int64) + 11_int64, &
                         offsets, (-0.6_real64, 0.9_real64), 0, fc)
    end subroutine run_wide_stencil

    !--------------------------------------------------------------------------
    ! Build a 1-based partition table of size nprocs+1.  Rows are
    ! distributed by floor + remainder so n_local differs by at most 1.
    !--------------------------------------------------------------------------
    subroutine build_partition_table(N, nprocs, pt)
        integer(int64), intent(in) :: N
        integer, intent(in) :: nprocs
        integer(int64), allocatable, intent(out) :: pt(:)

        integer :: r
        integer(int64) :: q, rem

        allocate (pt(nprocs + 1))
        q = N / int(nprocs, int64)
        rem = mod(N, int(nprocs, int64))

        pt(1) = 1_int64
        do r = 1, nprocs
            if (int(r - 1, int64) < rem) then
                pt(r + 1) = pt(r) + q + 1_int64
            else
                pt(r + 1) = pt(r) + q
            end if
        end do
    end subroutine build_partition_table

    pure function is_empty_row(i_glob, pattern) result(empty)
        integer(int64), intent(in) :: i_glob
        integer, intent(in) :: pattern
        logical :: empty

        select case (pattern)
        case (1)
            empty = (mod(i_glob, 4_int64) == 3_int64)
        case default
            empty = .false.
        end select
    end function is_empty_row

    pure function entry_value(i_glob, j_glob) result(v)
        integer(int64), intent(in) :: i_glob, j_glob
        complex(real64) :: v
        real(real64) :: re, im

        re = 0.1_real64 * real(i_glob + 1_int64, real64) &
             + 0.01_real64 * real(j_glob + 1_int64, real64)
        im = 0.001_real64 * real(modulo(i_glob * 7_int64 + j_glob * 13_int64, &
                                        17_int64), real64)
        v = cmplx(re, im, real64)
    end function entry_value

    !--------------------------------------------------------------------------
    ! Build the locally-owned rows of A as a column-sorted CSR with 0-based
    ! offsets and 0-based column indices.  Rows flagged as empty are
    ! skipped (row_starts(i+1) == row_starts(i)).  Duplicate columns that
    ! arise from offset wraparound are deduplicated within each row, since
    ! the SpMV does not deduplicate at runtime.
    !--------------------------------------------------------------------------
    subroutine build_local_csr(A, lb, ub, N, offsets, empty_pattern)
        type(CSR), intent(out) :: A
        integer(int64), intent(in) :: lb, ub, N
        integer, intent(in) :: offsets(:)
        integer, intent(in) :: empty_pattern

        integer(int64) :: nrows, nnz, i_glob, idx
        integer :: k, m, ncols
        integer(int64), allocatable :: cols(:), uniq(:)
        integer :: ncols_max

        nrows = ub - lb + 1
        ncols_max = size(offsets)

        ! First pass: count nonzeros so we can allocate exactly.
        allocate (cols(ncols_max), uniq(ncols_max))
        nnz = 0_int64
        do i_glob = lb - 1_int64, ub - 1_int64
            if (is_empty_row(i_glob, empty_pattern)) cycle
            do k = 1, ncols_max
                cols(k) = modulo(i_glob + int(offsets(k), int64), N)
            end do
            call sort_int64_small(cols)
            call dedupe_sorted(cols, uniq, ncols)
            nnz = nnz + int(ncols, int64)
        end do

        A%rows = int(N, int32)
        A%columns = int(N, int32)
        A%structure = "GE"

        allocate (A%row_starts(nrows + 1))
        allocate (A%col_indexes(max(nnz, 1_int64)))
        allocate (A%values(max(nnz, 1_int64)))

        idx = 0_int64
        do i_glob = lb - 1_int64, ub - 1_int64
            A%row_starts(i_glob - (lb - 1_int64) + 1_int64) = idx
            if (is_empty_row(i_glob, empty_pattern)) cycle
            do k = 1, ncols_max
                cols(k) = modulo(i_glob + int(offsets(k), int64), N)
            end do
            call sort_int64_small(cols)
            call dedupe_sorted(cols, uniq, ncols)
            do m = 1, ncols
                idx = idx + 1_int64
                A%col_indexes(idx) = uniq(m)
                A%values(idx) = entry_value(i_glob, uniq(m))
            end do
        end do
        A%row_starts(nrows + 1) = idx

        deallocate (cols, uniq)
    end subroutine build_local_csr

    subroutine sort_int64_small(a)
        integer(int64), intent(inout) :: a(:)
        integer :: i, j
        integer(int64) :: tmp

        do i = 2, size(a)
            tmp = a(i)
            j = i - 1
            do while (j >= 1)
                if (a(j) > tmp) then
                    a(j + 1) = a(j)
                    j = j - 1
                else
                    exit
                end if
            end do
            a(j + 1) = tmp
        end do
    end subroutine sort_int64_small

    subroutine dedupe_sorted(a, uniq, n)
        integer(int64), intent(in) :: a(:)
        integer(int64), intent(out) :: uniq(:)
        integer, intent(out) :: n

        integer :: i

        if (size(a) == 0) then
            n = 0
            return
        end if

        n = 1
        uniq(1) = a(1)
        do i = 2, size(a)
            if (a(i) /= uniq(n)) then
                n = n + 1
                uniq(n) = a(i)
            end if
        end do
    end subroutine dedupe_sorted

    subroutine serial_matvec(N, lb, ub, offsets, x_full, empty_pattern, y)
        integer(int64), intent(in) :: N, lb, ub
        integer, intent(in) :: offsets(:)
        complex(real64), intent(in) :: x_full(:)
        integer, intent(in) :: empty_pattern
        complex(real64), intent(out) :: y(:)

        integer(int64) :: i_glob
        integer :: k, m, ncols
        integer(int64), allocatable :: cols(:), uniq(:)
        complex(real64) :: acc

        allocate (cols(size(offsets)), uniq(size(offsets)))
        do i_glob = lb - 1_int64, ub - 1_int64
            acc = (0.0_real64, 0.0_real64)
            if (.not. is_empty_row(i_glob, empty_pattern)) then
                do k = 1, size(offsets)
                    cols(k) = modulo(i_glob + int(offsets(k), int64), N)
                end do
                call sort_int64_small(cols)
                call dedupe_sorted(cols, uniq, ncols)
                do m = 1, ncols
                    acc = acc + entry_value(i_glob, uniq(m)) * x_full(uniq(m) + 1_int64)
                end do
            end if
            y(i_glob - (lb - 1_int64) + 1_int64) = acc
        end do
        deallocate (cols, uniq)
    end subroutine serial_matvec

end program test_sparse_cpu_spmv
