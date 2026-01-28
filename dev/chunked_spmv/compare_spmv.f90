!------------------------------------------------------------------------------
! Comparison test: New chunked SpMV vs QuOp_MPI's SpMV_Series
!
! Each method is tested independently in its own subroutine with properly
! scoped imports to avoid any confusion about index conventions.
!
! Compile (from dev/chunked_spmv directory):
!   mpif90 -O3 -fopenmp ../../src/mpi/sparse/sparse.f90 \
!       chunked_spmv_mod.f90 compare_spmv.f90 -o compare_spmv
!
! Run: mpirun -np 4 ./compare_spmv 20
! Run with chunking: mpirun -np 4 ./compare_spmv 20 65536
!------------------------------------------------------------------------------

program compare_spmv
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    implicit none

    integer :: ierr, rank, nprocs
    integer :: n_qubits, max_recv_chunk
    character(len=32) :: arg
    real(dp) :: t_setup_quop, t_spmv_quop, mem_quop
    real(dp) :: t_setup_new, t_spmv_new, mem_new
    integer :: n_iters
    logical :: quop_passed, new_passed
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    ! Parse command line
    n_qubits = 16
    max_recv_chunk = 0  ! 0 = no chunking
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg)
        read(arg, *) n_qubits
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg)
        read(arg, *) max_recv_chunk
    end if
    
    n_iters = 10
    
    if (rank == 0) then
        print '(A)', '============================================================'
        print '(A)', 'SpMV Comparison: QuOp_MPI vs New Chunked Method'
        print '(A)', '============================================================'
        print '(A,I0)', 'MPI ranks:    ', nprocs
        print '(A,I0)', 'Qubits:       ', n_qubits
        print '(A,I0)', 'System size:  ', 2_int64**n_qubits
        print '(A,I0)', 'Iterations:   ', n_iters
        if (max_recv_chunk > 0) then
            print '(A,I0)', 'Chunk size:   ', max_recv_chunk
        else
            print '(A)', 'Chunking:     disabled'
        end if
        print '(A)', ''
    end if
    
    ! Test each method independently
    call test_quop_spmv(n_qubits, n_iters, t_setup_quop, t_spmv_quop, mem_quop, quop_passed)
    call test_new_spmv(n_qubits, n_iters, max_recv_chunk, t_setup_new, t_spmv_new, mem_new, new_passed)
    
    ! Print comparison
    if (rank == 0) then
        print '(A)', ''
        print '(A)', '============================================================'
        print '(A)', 'COMPARISON SUMMARY'
        print '(A)', '============================================================'
        print '(A)', ''
        print '(A)', '--- Setup Time ---'
        print '(A,F10.2,A)', 'QuOp:       ', t_setup_quop * 1000, ' ms'
        print '(A,F10.2,A)', 'New method: ', t_setup_new * 1000, ' ms'
        if (t_setup_new > 0) print '(A,F10.2,A)', 'Speedup:    ', t_setup_quop / t_setup_new, 'x'
        print '(A)', ''
        print '(A)', '--- SpMV Time (per iteration) ---'
        print '(A,F10.2,A)', 'QuOp:       ', t_spmv_quop * 1000, ' ms'
        print '(A,F10.2,A)', 'New method: ', t_spmv_new * 1000, ' ms'
        if (t_spmv_new > 0) print '(A,F10.2,A)', 'Speedup:    ', t_spmv_quop / t_spmv_new, 'x'
        print '(A)', ''
        print '(A)', '--- Memory Usage (comm data per rank) ---'
        print '(A,F10.4,A)', 'QuOp:       ', mem_quop, ' MB'
        print '(A,F10.4,A)', 'New method: ', mem_new, ' MB'
        print '(A,F10.2,A)', 'Savings:    ', mem_quop - mem_new, ' MB'
        if (mem_quop > 0) print '(A,F10.1,A)', 'Reduction:  ', 100.0 * (1.0 - mem_new / mem_quop), '%'
        print '(A)', ''
        print '(A)', '--- Overall ---'
        if (quop_passed) then
            print '(A)', 'QuOp correctness:   PASS'
        else
            print '(A)', 'QuOp correctness:   FAIL'
        end if
        if (new_passed) then
            print '(A)', 'New correctness:    PASS'
        else
            print '(A)', 'New correctness:    FAIL'
        end if
    end if
    
    call MPI_Finalize(ierr)

contains

    !--------------------------------------------------------------------------
    ! Test QuOp's SpMV implementation
    !--------------------------------------------------------------------------
    subroutine test_quop_spmv(n_qubits, n_iters, t_setup, t_spmv, mem_mb, passed)
        use mpi
        use iso_fortran_env, only: dp => real64, int64
        use Sparse, only: CSR, SpMV_Series, Reconcile_Communications, Generate_Partition_Table
        use csr_generators, only: hypercube
        
        integer, intent(in) :: n_qubits, n_iters
        real(dp), intent(out) :: t_setup, t_spmv, mem_mb
        logical, intent(out) :: passed
        
        integer :: rank, nprocs, ierr, iter
        integer :: system_size, n_local, lb, ub
        integer, allocatable :: partition_table(:)
        type(CSR) :: A
        complex(dp), allocatable :: u(:), v(:)
        real(dp) :: t_start, max_err, global_max_err
        integer(dp) :: i, local_nnz
        integer(dp) :: lb_dp, ub_dp
        
        ! Temporary arrays for hypercube (which uses assumed-size arrays)
        integer(dp), allocatable :: row_starts_temp(:)
        integer(dp), allocatable :: col_indexes_temp(:)
        complex(dp), allocatable :: values_temp(:)
        integer(dp) :: elem_lb, elem_ub
        integer :: total_send, total_recv
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        system_size = 2**n_qubits
        
        if (rank == 0) print '(A)', '--- Testing QuOp SpMV ---'
        
        ! Setup timing
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        ! Generate partition table (1-based)
        call Generate_Partition_Table(system_size, partition_table, MPI_COMM_WORLD)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1
        local_nnz = int(n_local, dp) * n_qubits
        lb_dp = lb
        ub_dp = ub
        
        ! Element index bounds (global, 1-based)
        elem_lb = n_qubits * (lb_dp - 1) + 1
        elem_ub = n_qubits * ub_dp
        
        ! Allocate temporary arrays for hypercube subroutine
        allocate(row_starts_temp(n_local + 1))
        allocate(col_indexes_temp(elem_lb:elem_ub))
        allocate(values_temp(elem_lb:elem_ub))
        
        ! Use QuOp's hypercube generator
        call hypercube(int(n_qubits, dp), lb_dp, ub_dp, row_starts_temp, col_indexes_temp, values_temp)
        
        ! Build CSR matrix - copy to pointer arrays with correct bounds
        A%rows = system_size
        A%columns = system_size
        A%structure = 'SY'
        
        allocate(A%row_starts(lb:ub+1))
        allocate(A%col_indexes(elem_lb:elem_ub))
        allocate(A%values(elem_lb:elem_ub))
        
        ! Copy row_starts with correct indexing
        do i = 1, n_local + 1
            A%row_starts(lb + i - 1) = row_starts_temp(i)
        end do
        A%col_indexes = col_indexes_temp
        A%values = values_temp
        
        deallocate(row_starts_temp, col_indexes_temp, values_temp)
        
        ! Setup communications
        call Reconcile_Communications(A, partition_table, MPI_COMM_WORLD)
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_setup = MPI_Wtime() - t_start
        
        ! Allocate vectors
        allocate(u(lb:ub), v(lb:ub))
        
        ! Test correctness: all-ones vector
        u = (1.0_dp, 0.0_dp)
        call SpMV_Series(A, u, partition_table, 1, 1, 1, rank, v, MPI_COMM_WORLD)
        
        max_err = 0.0_dp
        do i = lb, ub
            max_err = max(max_err, abs(v(i) - cmplx(n_qubits, 0.0_dp, dp)))
        end do
        call MPI_Allreduce(max_err, global_max_err, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
        
        passed = global_max_err < 1.0e-10_dp
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Correctness (all ones): PASS'
            else
                print '(A,ES12.4)', 'Correctness (all ones): FAIL, max_err=', global_max_err
            end if
        end if
        
        ! Warmup
        do iter = 1, 2
            call SpMV_Series(A, u, partition_table, 1, 1, 1, rank, v, MPI_COMM_WORLD)
        end do
        
        ! Timed runs
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        do iter = 1, n_iters
            call SpMV_Series(A, u, partition_table, 1, 1, 1, rank, v, MPI_COMM_WORLD)
        end do
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_spmv = (MPI_Wtime() - t_start) / n_iters
        
        ! Calculate memory usage - comprehensive analysis
        ! QuOp persistent arrays:
        !   - row_starts: (n_local+1) * 8 bytes
        !   - col_indexes: local_nnz * 8 bytes
        !   - values: local_nnz * 16 bytes
        !   - local_col_inds: local_nnz * 8 bytes (remapped columns)
        !   - RHS_send_inds: total_send * 8 bytes
        !   - num_send_inds, send_disps: nprocs * 4 bytes each
        !   - num_rec_inds, rec_disps: nprocs * 4 bytes each
        ! QuOp runtime buffers (allocated inside SpMV_Series):
        !   - u_resize: (n_local + total_recv) * 16 bytes
        !   - send_values: total_send * 16 bytes
        !   - rec_values: total_recv * 16 bytes
        total_send = sum(A%num_send_inds)
        total_recv = sum(A%num_rec_inds)
        
        ! Persistent storage
        mem_mb = real(n_local + 1) * 8 / 1e6  ! row_starts
        mem_mb = mem_mb + real(local_nnz) * 8 / 1e6  ! col_indexes
        mem_mb = mem_mb + real(local_nnz) * 16 / 1e6  ! values
        mem_mb = mem_mb + real(local_nnz) * 8 / 1e6  ! local_col_inds
        mem_mb = mem_mb + real(total_send) * 8 / 1e6  ! RHS_send_inds
        mem_mb = mem_mb + real(nprocs) * 16 / 1e6  ! num_send/rec_inds + disps
        
        ! Runtime buffers
        mem_mb = mem_mb + real(n_local + total_recv) * 16 / 1e6  ! u_resize
        mem_mb = mem_mb + real(total_send) * 16 / 1e6  ! send_values
        mem_mb = mem_mb + real(total_recv) * 16 / 1e6  ! rec_values
        
        if (rank == 0) then
            print '(A,F10.2,A)', 'Setup time:  ', t_setup * 1000, ' ms'
            print '(A,F10.2,A)', 'SpMV time:   ', t_spmv * 1000, ' ms/iter'
            print '(A,I0)', 'Total recv:  ', total_recv
            print '(A,I0)', 'Total send:  ', total_send
        end if
        
        ! Cleanup
        deallocate(u, v, partition_table)
        if (associated(A%row_starts)) deallocate(A%row_starts)
        if (associated(A%col_indexes)) deallocate(A%col_indexes)
        if (associated(A%values)) deallocate(A%values)
        if (associated(A%local_col_inds)) deallocate(A%local_col_inds)
        if (associated(A%RHS_send_inds)) deallocate(A%RHS_send_inds)
        if (associated(A%num_send_inds)) deallocate(A%num_send_inds)
        if (associated(A%send_disps)) deallocate(A%send_disps)
        if (associated(A%num_rec_inds)) deallocate(A%num_rec_inds)
        if (associated(A%rec_disps)) deallocate(A%rec_disps)
        
    end subroutine test_quop_spmv

    !--------------------------------------------------------------------------
    ! Test new chunked SpMV implementation
    !--------------------------------------------------------------------------
    subroutine test_new_spmv(n_qubits, n_iters, max_recv_chunk, t_setup, t_spmv, mem_mb, passed)
        use mpi
        use iso_fortran_env, only: dp => real64, int64
        use chunked_spmv_mod, only: generate_partition_table, build_hypercube_csr, &
                                    setup_graph_comm, build_hash_table, &
                                    spmv_sorted_rows, cleanup_graph_comm
        
        integer, intent(in) :: n_qubits, n_iters, max_recv_chunk
        real(dp), intent(out) :: t_setup, t_spmv, mem_mb
        logical, intent(out) :: passed
        
        integer :: rank, nprocs, ierr, iter
        integer(int64) :: system_size, n_local, local_nnz, lb, ub, i
        integer(int64), allocatable :: partition_table(:)
        integer(int64), allocatable :: row_starts(:), col_indexes(:)
        complex(dp), allocatable :: u(:), v(:)
        complex(dp), allocatable :: send_buf(:), recv_buf(:)
        integer :: graph_comm
        integer(int64) :: total_recv, total_send
        integer(int64), allocatable :: recv_indices_sorted(:), send_offsets(:)
        integer(int64), allocatable :: sort_perm(:)
        integer, allocatable :: recv_counts(:), recv_disps(:)
        integer, allocatable :: send_counts(:), send_disps(:)
        integer, allocatable :: in_neighbors(:), out_neighbors(:)
        ! Hash table for O(1) lookup
        integer(int64), allocatable :: hash_keys(:)
        integer(int64), allocatable :: hash_vals(:)
        integer(int64) :: hash_size, max_chunk_64
        real(dp) :: t_start, max_err, global_max_err
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        system_size = 2_int64**n_qubits
        
        if (rank == 0) print '(A)', ''
        if (rank == 0) print '(A)', '--- Testing New Chunked SpMV ---'
        
        ! Setup timing
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        ! Generate partition table (0-based)
        call generate_partition_table(system_size, partition_table)
        
        ! Build hypercube CSR (0-based columns, sorted rows)
        call build_hypercube_csr(n_qubits, partition_table, row_starts, col_indexes, n_local, local_nnz)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        
        ! Setup graph communicator
        call setup_graph_comm(row_starts, col_indexes, partition_table, &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              in_neighbors, out_neighbors, total_recv, total_send, lb, ub)
        
        ! Build hash table for O(1) average lookup (done once at setup)
        call build_hash_table(recv_indices_sorted, sort_perm, total_recv, &
                              hash_keys, hash_vals, hash_size)
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_setup = MPI_Wtime() - t_start
        
        ! Allocate vectors and buffers
        allocate(u(n_local), v(n_local))
        allocate(send_buf(max(total_send, 1_int64)), recv_buf(max(total_recv, 1_int64)))
        
        ! Convert max_recv_chunk to int64
        max_chunk_64 = int(max_recv_chunk, int64)
        
        ! Test correctness: all-ones vector
        u = (1.0_dp, 0.0_dp)
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf, &
                              hash_keys, hash_vals, hash_size, max_chunk_64)
        
        max_err = 0.0_dp
        do i = 1, n_local
            max_err = max(max_err, abs(v(i) - cmplx(n_qubits, 0.0_dp, dp)))
        end do
        call MPI_Allreduce(max_err, global_max_err, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
        
        passed = global_max_err < 1.0e-10_dp
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Correctness (all ones): PASS'
            else
                print '(A,ES12.4)', 'Correctness (all ones): FAIL, max_err=', global_max_err
            end if
        end if
        
        ! Warmup
        do iter = 1, 2
            call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                                  graph_comm, recv_indices_sorted, sort_perm, &
                                  recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                                  total_recv, total_send, lb, ub, send_buf, recv_buf, &
                                  hash_keys, hash_vals, hash_size, max_chunk_64)
        end do
        
        ! Timed runs
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        do iter = 1, n_iters
            call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                                  graph_comm, recv_indices_sorted, sort_perm, &
                                  recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                                  total_recv, total_send, lb, ub, send_buf, recv_buf, &
                                  hash_keys, hash_vals, hash_size, max_chunk_64)
        end do
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_spmv = (MPI_Wtime() - t_start) / n_iters
        
        ! Calculate memory usage - comprehensive analysis
        ! New method persistent arrays:
        !   - row_starts: (n_local+1) * 8 bytes
        !   - col_indexes: local_nnz * 8 bytes (NO values - unit matrix)
        !   - recv_indices_sorted: total_recv * 8 bytes
        !   - sort_perm: total_recv * 8 bytes (int64)
        !   - send_offsets: total_send * 8 bytes
        !   - hash_keys: hash_size * 8 bytes
        !   - hash_vals: hash_size * 8 bytes (int64)
        !   - recv/send_counts, disps: ~4 * n_neighbors * 4 bytes (small)
        ! New method runtime buffers:
        !   - send_buf: total_send * 16 bytes
        !   - recv_buf: total_recv * 16 bytes
        !   - recv_buf_sorted: total_recv (or chunk_size) * 16 bytes
        
        ! Persistent storage
        mem_mb = real(n_local + 1) * 8 / 1e6  ! row_starts
        mem_mb = mem_mb + real(local_nnz) * 8 / 1e6  ! col_indexes (no values!)
        mem_mb = mem_mb + real(total_recv) * 8 / 1e6  ! recv_indices_sorted
        mem_mb = mem_mb + real(total_recv) * 8 / 1e6  ! sort_perm (int64)
        mem_mb = mem_mb + real(total_send) * 8 / 1e6  ! send_offsets
        mem_mb = mem_mb + real(hash_size) * 8 / 1e6   ! hash_keys
        mem_mb = mem_mb + real(hash_size) * 8 / 1e6   ! hash_vals (int64)
        
        ! Runtime buffers
        mem_mb = mem_mb + real(total_send) * 16 / 1e6  ! send_buf
        mem_mb = mem_mb + real(total_recv) * 16 / 1e6  ! recv_buf
        if (max_chunk_64 > 0 .and. total_recv > max_chunk_64) then
            mem_mb = mem_mb + real(max_chunk_64) * 16 / 1e6  ! recv_buf_sorted (chunked)
        else
            mem_mb = mem_mb + real(total_recv) * 16 / 1e6  ! recv_buf_sorted (full)
        end if
        
        if (rank == 0) then
            print '(A,F10.2,A)', 'Setup time:  ', t_setup * 1000, ' ms'
            print '(A,F10.2,A)', 'SpMV time:   ', t_spmv * 1000, ' ms/iter'
            print '(A,I0)', 'Total recv:  ', total_recv
            print '(A,I0)', 'Total send:  ', total_send
            if (max_chunk_64 > 0 .and. total_recv > max_chunk_64) then
                print '(A,I0)', 'Chunks used: ', (total_recv + max_chunk_64 - 1) / max_chunk_64
            end if
        end if
        
        ! Cleanup
        deallocate(u, v, send_buf, recv_buf)
        deallocate(partition_table, row_starts, col_indexes)
        deallocate(hash_keys, hash_vals)
        call cleanup_graph_comm(graph_comm, recv_indices_sorted, sort_perm, &
                                recv_counts, recv_disps, send_offsets, &
                                send_counts, send_disps, in_neighbors, out_neighbors)
        
    end subroutine test_new_spmv

end program compare_spmv
