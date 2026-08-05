! test_sparse_gpu_chebyshev.f90
!
! GPU-vs-CPU regression test for `chebyshev_multiply` on the sparse
! propagator's unit-valued path (the path used by QAOASparse / hypercube
! mixers).
!
! Both back-ends share the same Bessel-coefficient maths and the same
! recurrence T_{k+1} = 2*X*T_k - T_{k-1}; only the per-iteration kernels
! differ.  This test runs CPU and GPU against the *same* CSR with the
! *same* spectral radius and the *same* time, then compares the resulting
! C vector element-wise.
!
! Coverage:
!   * unit_short        - 1D banded unit-valued CSR (5 offsets), small t
!   * unit_long         - same matrix, longer t (more Chebyshev terms)
!   * weighted_short    - same offsets but explicit complex weights
!
! For systems with a single GPU the test honours QUOP_RANKS_PER_GPU.

program test_sparse_gpu_chebyshev

    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, &
                                           c_size_t, c_null_ptr
    use sparse, only: CSR, setup_graph_communications, &
                      cleanup_graph_communications, csr_to_device, &
                      csr_free_device
    use chebyshev, only: chebyshev_multiply
    use hipfort, only: hipMalloc, hipFree, hipMemcpy, hipMemcpyHostToDevice, &
                       hipMemcpyDeviceToHost, hipDeviceSynchronize
    use hipfort_check, only: hipCheck
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    use communicators, only: create_NODECOMM
    use MPI

    implicit none

    real(real64), parameter :: tol = 1.0e-10_real64

    integer :: rank, nprocs, ierr
    integer :: failures, local_failures
    integer :: NODECOMM
    type(gpu_topology_t) :: topology

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    call create_NODECOMM(MPI_COMM_WORLD, NODECOMM)
    call init_gpu_topology(NODECOMM, topology, suppress_warnings=.true.)

    if (rank == 0) then
        write (*, '(A,I0)') "test_sparse_gpu_chebyshev: ranks=", nprocs
    end if

    failures = 0

    call run_unit_short(local_failures);     failures = failures + local_failures
    call run_unit_long(local_failures);      failures = failures + local_failures
    call run_weighted_short(local_failures); failures = failures + local_failures
    call run_hypercube(local_failures);      failures = failures + local_failures

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
    ! Run a CPU vs GPU comparison of chebyshev_multiply on a single CSR.
    ! Two independent CSR copies are built so that one can stay host-only
    ! (device_ready=false) while the other is uploaded to the device.
    !--------------------------------------------------------------------------
    subroutine run_subtest(label, N, offsets, has_values, t, fail_count)
        character(len=*), intent(in) :: label
        integer(int64), intent(in) :: N
        integer, intent(in) :: offsets(:)
        logical, intent(in) :: has_values
        real(real64), intent(in) :: t
        integer, intent(out) :: fail_count

        integer(int64), allocatable :: partition_table(:)
        integer(int64) :: lb, ub, n_local, i64, n_alloc
        integer(c_size_t) :: bytes
        type(CSR) :: A_cpu, A_gpu
        complex(real64), allocatable, target :: B_host(:)
        complex(real64), allocatable, target :: C_cpu(:)
        complex(real64), allocatable, target :: C_gpu_host(:)
        type(c_ptr) :: B_dev_ptr, C_dev_ptr
        complex(real64), pointer :: B_dev(:), C_dev(:)
        real(real64) :: M_local, M
        real(real64) :: max_err, global_max_err

        fail_count = 0

        call build_partition_table(N, nprocs, partition_table)
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1
        n_alloc = max(n_local, 1_int64)

        ! Initial state: uniform 1/sqrt(N), real-valued.
        allocate (B_host(n_alloc))
        B_host = cmplx(1.0_real64 / sqrt(real(N, real64)), 0.0_real64, real64)

        allocate (C_cpu(n_alloc))
        allocate (C_gpu_host(n_alloc))
        C_cpu = (0.0_real64, 0.0_real64)
        C_gpu_host = (0.0_real64, 0.0_real64)

        call build_local_csr(A_cpu, lb, ub, N, offsets, has_values)
        call build_local_csr(A_gpu, lb, ub, N, offsets, has_values)

        ! Spectral radius via Gershgorin on the host CSR (identical for both
        ! copies).  For unit-valued, this is just the max nnz per row.
        call host_gershgorin_max(A_cpu, lb, ub, has_values, M_local)
        call MPI_Allreduce(M_local, M, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                           MPI_COMM_WORLD, ierr)
        if (M <= 0.0_real64) M = 1.0_real64

        ! ---- CPU run ----
        if (n_local > 0) then
            call setup_graph_communications(A_cpu, partition_table, MPI_COMM_WORLD)
            call chebyshev_multiply(A_cpu, B_host, t, partition_table, C_cpu, &
                                    MPI_COMM_WORLD, spectral_radius=M)
        end if

        ! ---- GPU run ----
        bytes = int(n_alloc, c_size_t) * 16_c_size_t
        call hipCheck(hipMalloc(B_dev_ptr, bytes))
        call hipCheck(hipMalloc(C_dev_ptr, bytes))
        call c_f_pointer(B_dev_ptr, B_dev, [n_alloc])
        call c_f_pointer(C_dev_ptr, C_dev, [n_alloc])

        call hipCheck(hipMemcpy(B_dev_ptr, c_loc(B_host(1)), bytes, &
                                hipMemcpyHostToDevice))

        if (n_local > 0) then
            call setup_graph_communications(A_gpu, partition_table, MPI_COMM_WORLD)
            call csr_to_device(A_gpu)
            call chebyshev_multiply(A_gpu, B_dev, t, partition_table, C_dev, &
                                    MPI_COMM_WORLD, spectral_radius=M)
        end if

        call hipCheck(hipMemcpy(c_loc(C_gpu_host(1)), C_dev_ptr, bytes, &
                                hipMemcpyDeviceToHost))
        call hipCheck(hipDeviceSynchronize())

        max_err = 0.0_real64
        do i64 = 1, n_local
            max_err = max(max_err, abs(C_gpu_host(i64) - C_cpu(i64)))
        end do

        call MPI_Allreduce(max_err, global_max_err, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MPI_COMM_WORLD, ierr)

        if (rank == 0) then
            write (*, '(A,A,A,ES10.2,A,ES10.2)') "  [", trim(label), &
                "] M=", M, " max |C_gpu - C_cpu| = ", global_max_err
        end if

        if (global_max_err > tol) fail_count = 1

        ! Cleanup
        call csr_free_device(A_gpu)
        call cleanup_graph_communications(A_gpu)
        call cleanup_graph_communications(A_cpu)
        if (associated(A_cpu%row_starts)) deallocate (A_cpu%row_starts)
        if (associated(A_cpu%col_indexes)) deallocate (A_cpu%col_indexes)
        if (associated(A_cpu%values)) deallocate (A_cpu%values)
        if (associated(A_gpu%row_starts)) deallocate (A_gpu%row_starts)
        if (associated(A_gpu%col_indexes)) deallocate (A_gpu%col_indexes)
        if (associated(A_gpu%values)) deallocate (A_gpu%values)

        call hipCheck(hipFree(B_dev_ptr))
        call hipCheck(hipFree(C_dev_ptr))

        deallocate (B_host, C_cpu, C_gpu_host, partition_table)
    end subroutine run_subtest

    subroutine run_unit_short(fc)
        integer, intent(out) :: fc
        integer :: offsets(5)
        offsets = [-3, -1, 0, 1, 3]
        call run_subtest("unit_short", 31_int64 * int(nprocs, int64) + 5_int64, &
                         offsets, .false., 0.05_real64, fc)
    end subroutine run_unit_short

    subroutine run_unit_long(fc)
        integer, intent(out) :: fc
        integer :: offsets(5)
        offsets = [-3, -1, 0, 1, 3]
        call run_subtest("unit_long", 31_int64 * int(nprocs, int64) + 5_int64, &
                         offsets, .false., 0.5_real64, fc)
    end subroutine run_unit_long

    subroutine run_weighted_short(fc)
        integer, intent(out) :: fc
        integer :: offsets(5)
        offsets = [-3, -1, 0, 1, 3]
        call run_subtest("weighted_short", 31_int64 * int(nprocs, int64) + 5_int64, &
                         offsets, .true., 0.05_real64, fc)
    end subroutine run_weighted_short

    !--------------------------------------------------------------------------
    ! Hypercube subtest -- exercises the XOR-structured CSR used by
    ! QAOASparse hypercube mixer (3-qubit graph, N=8).  Build CPU and GPU
    ! CSR independently and compare chebyshev_multiply outputs.
    !--------------------------------------------------------------------------
    subroutine run_hypercube(fc)
        integer, intent(out) :: fc

        integer(int64), parameter :: nqubits = 3_int64
        integer(int64) :: N
        integer(int64), allocatable :: partition_table(:)
        integer(int64) :: lb, ub, n_local, i64, n_alloc
        integer(c_size_t) :: bytes
        type(CSR) :: A_cpu, A_gpu
        complex(real64), allocatable, target :: B_host(:)
        complex(real64), allocatable, target :: C_cpu(:)
        complex(real64), allocatable, target :: C_gpu_host(:)
        type(c_ptr) :: B_dev_ptr, C_dev_ptr
        complex(real64), pointer :: B_dev(:), C_dev(:)
        real(real64) :: M_local, M
        real(real64) :: max_err, global_max_err
        real(real64) :: t

        fc = 0
        t = 0.5_real64
        N = ishft(1_int64, int(nqubits))    ! 2^nqubits

        call build_partition_table(N, nprocs, partition_table)
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = max(ub - lb + 1, 0_int64)
        n_alloc = max(n_local, 1_int64)

        allocate (B_host(n_alloc))
        B_host = cmplx(1.0_real64 / sqrt(real(N, real64)), 0.0_real64, real64)

        allocate (C_cpu(n_alloc), C_gpu_host(n_alloc))
        C_cpu = (0.0_real64, 0.0_real64); C_gpu_host = (0.0_real64, 0.0_real64)

        call build_local_hypercube_csr(A_cpu, lb, ub, N, nqubits)
        call build_local_hypercube_csr(A_gpu, lb, ub, N, nqubits)

        ! For unit-valued hypercube, every row has nqubits off-diagonals,
        ! diagonal=0; Gershgorin bound = nqubits.
        call host_gershgorin_max(A_cpu, lb, ub, .false., M_local)
        call MPI_Allreduce(M_local, M, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                           MPI_COMM_WORLD, ierr)
        if (M <= 0.0_real64) M = 1.0_real64

        if (n_local > 0) then
            call setup_graph_communications(A_cpu, partition_table, MPI_COMM_WORLD)
            call chebyshev_multiply(A_cpu, B_host, t, partition_table, C_cpu, &
                                    MPI_COMM_WORLD, spectral_radius=M)
        end if

        bytes = int(n_alloc, c_size_t) * 16_c_size_t
        call hipCheck(hipMalloc(B_dev_ptr, bytes))
        call hipCheck(hipMalloc(C_dev_ptr, bytes))
        call c_f_pointer(B_dev_ptr, B_dev, [n_alloc])
        call c_f_pointer(C_dev_ptr, C_dev, [n_alloc])
        call hipCheck(hipMemcpy(B_dev_ptr, c_loc(B_host(1)), bytes, &
                                hipMemcpyHostToDevice))

        if (n_local > 0) then
            call setup_graph_communications(A_gpu, partition_table, MPI_COMM_WORLD)
            call csr_to_device(A_gpu)
            call chebyshev_multiply(A_gpu, B_dev, t, partition_table, C_dev, &
                                    MPI_COMM_WORLD, spectral_radius=M)
        end if

        call hipCheck(hipMemcpy(c_loc(C_gpu_host(1)), C_dev_ptr, bytes, &
                                hipMemcpyDeviceToHost))
        call hipCheck(hipDeviceSynchronize())

        max_err = 0.0_real64
        do i64 = 1, n_local
            max_err = max(max_err, abs(C_gpu_host(i64) - C_cpu(i64)))
        end do
        call MPI_Allreduce(max_err, global_max_err, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MPI_COMM_WORLD, ierr)

        if (rank == 0) then
            write (*, '(A,I0,A,ES10.2,A,ES10.2)') "  [hypercube] N=2^", &
                int(nqubits), " M=", M, " max |C_gpu - C_cpu| = ", global_max_err
        end if

        ! For the QAOA mixer-only zero-quality case, we expect a UNIFORM
        ! superposition to remain uniform under exp(-i t H).  Print a
        ! diagnostic on rank 0 to confirm.
        if (rank == 0) then
            write (*, '(A,2ES12.4,A,2ES12.4)') "  [hypercube] C_cpu(1)=", &
                C_cpu(1), "  C_gpu(1)=", C_gpu_host(1)
        end if

        if (global_max_err > tol) fc = 1

        call csr_free_device(A_gpu)
        call cleanup_graph_communications(A_gpu)
        call cleanup_graph_communications(A_cpu)
        if (associated(A_cpu%row_starts)) deallocate (A_cpu%row_starts)
        if (associated(A_cpu%col_indexes)) deallocate (A_cpu%col_indexes)
        if (associated(A_gpu%row_starts)) deallocate (A_gpu%row_starts)
        if (associated(A_gpu%col_indexes)) deallocate (A_gpu%col_indexes)

        call hipCheck(hipFree(B_dev_ptr))
        call hipCheck(hipFree(C_dev_ptr))
        deallocate (B_host, C_cpu, C_gpu_host, partition_table)
    end subroutine run_hypercube

    !--------------------------------------------------------------------------
    ! Build local CSR for an n-qubit hypercube (XOR mixer): row i has
    ! nonzeros at columns i XOR 2^b for b = 0..n-1, no diagonal.  Columns
    ! emitted 0-based and sorted to honour the kernel precondition.
    !--------------------------------------------------------------------------
    subroutine build_local_hypercube_csr(A, lb, ub, N, nqubits)
        type(CSR), intent(out) :: A
        integer(int64), intent(in) :: lb, ub, N, nqubits

        integer(int64) :: nrows, i_glob, idx, mask, c
        integer :: b
        integer(int64), allocatable :: cols(:)

        nrows = max(ub - lb + 1, 0_int64)

        A%rows = int(N, int32)
        A%columns = int(N, int32)
        A%structure = "GE"
        A%has_values = .false.

        allocate (A%row_starts(nrows + 1))
        allocate (A%col_indexes(max(nrows * nqubits, 1_int64)))
        allocate (cols(nqubits))

        idx = 0_int64
        do i_glob = lb - 1_int64, ub - 1_int64
            A%row_starts(i_glob - (lb - 1_int64) + 1_int64) = idx
            do b = 1, int(nqubits)
                mask = ishft(1_int64, b - 1)
                cols(b) = ieor(i_glob, mask)
            end do
            call sort_int64_small(cols)
            do b = 1, int(nqubits)
                idx = idx + 1_int64
                A%col_indexes(idx) = cols(b)
            end do
        end do
        A%row_starts(nrows + 1) = idx

        deallocate (cols)
    end subroutine build_local_hypercube_csr

    !--------------------------------------------------------------------------
    ! Build a 1-based partition table of size nprocs+1.
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

    !--------------------------------------------------------------------------
    ! Hermitian off-diagonal weights.  We pair offset (+d) with offset (-d)
    ! and assign A_{i, i+d} = w(d) and A_{i, i-d} = conj(w(d)) so that the
    ! resulting matrix is Hermitian (a sane operator for the Chebyshev
    ! expansion of exp(-i*H*t)).  Diagonal entries (d=0) are real.
    !--------------------------------------------------------------------------
    pure function entry_value(i_glob, j_glob) result(v)
        integer(int64), intent(in) :: i_glob, j_glob
        complex(real64) :: v
        integer(int64) :: d

        d = j_glob - i_glob
        if (d == 0_int64) then
            v = cmplx(0.3_real64, 0.0_real64, real64)
        else if (d > 0_int64) then
            v = cmplx(0.1_real64 * real(d, real64), &
                      0.05_real64 * real(d, real64), real64)
        else
            v = conjg(entry_value(j_glob, i_glob))
        end if
    end function entry_value

    !--------------------------------------------------------------------------
    ! Build local rows.  Columns are produced as 0-based, sorted, deduped.
    ! has_values=.false. leaves A%values null and sets A%has_values=.false.
    !--------------------------------------------------------------------------
    subroutine build_local_csr(A, lb, ub, N, offsets, has_values)
        type(CSR), intent(out) :: A
        integer(int64), intent(in) :: lb, ub, N
        integer, intent(in) :: offsets(:)
        logical, intent(in) :: has_values

        integer(int64) :: nrows, nnz, i_glob, idx
        integer :: k, m, ncols, ncols_max
        integer(int64), allocatable :: cols(:), uniq(:)

        nrows = ub - lb + 1
        ncols_max = size(offsets)

        allocate (cols(ncols_max), uniq(ncols_max))
        nnz = 0_int64
        do i_glob = lb - 1_int64, ub - 1_int64
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
        A%has_values = has_values

        allocate (A%row_starts(nrows + 1))
        allocate (A%col_indexes(max(nnz, 1_int64)))
        if (has_values) then
            allocate (A%values(max(nnz, 1_int64)))
        end if

        idx = 0_int64
        do i_glob = lb - 1_int64, ub - 1_int64
            A%row_starts(i_glob - (lb - 1_int64) + 1_int64) = idx
            do k = 1, ncols_max
                cols(k) = modulo(i_glob + int(offsets(k), int64), N)
            end do
            call sort_int64_small(cols)
            call dedupe_sorted(cols, uniq, ncols)
            do m = 1, ncols
                idx = idx + 1_int64
                A%col_indexes(idx) = uniq(m)
                if (has_values) A%values(idx) = entry_value(i_glob, uniq(m))
            end do
        end do
        A%row_starts(nrows + 1) = idx

        deallocate (cols, uniq)
    end subroutine build_local_csr

    !--------------------------------------------------------------------------
    ! Host Gershgorin upper bound on rho(H), assuming H = (anti-)Hermitian
    ! generator stored in A.  For has_values=.false. this reduces to the
    ! max nnz per row.  Mirrors the logic in estimate_spectral_radius.
    !--------------------------------------------------------------------------
    subroutine host_gershgorin_max(A, lb, ub, has_values, M_local)
        type(CSR), intent(in) :: A
        integer(int64), intent(in) :: lb, ub
        logical, intent(in) :: has_values
        real(real64), intent(out) :: M_local

        integer(int64) :: i, j, start_j, end_j, global_row
        real(real64) :: diag_element, row_sum, local_bound

        M_local = 0.0_real64
        do i = 1, ub - lb + 1
            start_j = A%row_starts(i) + 1
            end_j = A%row_starts(i + 1)
            global_row = lb + i - 2

            diag_element = 0.0_real64
            row_sum = 0.0_real64

            if (has_values) then
                do j = start_j, end_j
                    if (A%col_indexes(j) == global_row) then
                        diag_element = abs(A%values(j))
                    else
                        row_sum = row_sum + abs(A%values(j))
                    end if
                end do
            else
                do j = start_j, end_j
                    if (A%col_indexes(j) == global_row) then
                        diag_element = 1.0_real64
                    else
                        row_sum = row_sum + 1.0_real64
                    end if
                end do
            end if

            local_bound = diag_element + row_sum
            if (local_bound > M_local) M_local = local_bound
        end do
    end subroutine host_gershgorin_max

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

end program test_sparse_gpu_chebyshev
