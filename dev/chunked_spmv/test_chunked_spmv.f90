!------------------------------------------------------------------------------
! Test program for memory-efficient SpMV module
!
! Compile: mpif90 -O3 -fopenmp -o test_chunked_spmv chunked_spmv_mod.f90 test_chunked_spmv.f90
! Run: mpirun -np 4 ./test_chunked_spmv 16
!------------------------------------------------------------------------------

program test_chunked_spmv
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use chunked_spmv_mod
    implicit none

    integer :: ierr, rank, nprocs
    integer :: n_qubits
    character(len=32) :: arg
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    
    ! Parse command line
    n_qubits = 14
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg)
        read(arg, *) n_qubits
    end if
    
    if (rank == 0) then
        print '(A)', '================================================='
        print '(A)', 'Memory-Efficient SpMV for Unit-Valued Matrices'
        print '(A)', '================================================='
        print '(A,I0)', 'MPI ranks: ', nprocs
        print '(A,I0)', 'Qubits: ', n_qubits
        print '(A,I0)', 'System size: ', 2_int64**n_qubits
        print '(A)', ''
    end if
    
    call test_correctness(n_qubits)
    call benchmark(n_qubits, 10)
    
    call MPI_Finalize(ierr)

contains

    !--------------------------------------------------------------------------
    ! Test correctness
    !--------------------------------------------------------------------------
    subroutine test_correctness(n_qubits)
        integer, intent(in) :: n_qubits
        
        integer :: rank, nprocs, ierr, k
        integer(int64) :: system_size, n_local, local_nnz, lb, ub, i
        integer(int64), allocatable :: partition_table(:), row_starts(:), col_indexes(:)
        complex(dp), allocatable :: u(:), v(:), expected(:)
        complex(dp), allocatable :: send_buf(:), recv_buf(:)
        integer :: graph_comm, total_recv, total_send
        integer(int64), allocatable :: recv_indices_sorted(:), send_offsets(:)
        integer, allocatable :: sort_perm(:), recv_counts(:), recv_disps(:)
        integer, allocatable :: send_counts(:), send_disps(:)
        integer, allocatable :: in_neighbors(:), out_neighbors(:)
        logical :: passed
        real(dp) :: max_err
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        system_size = 2_int64**n_qubits
        
        ! Generate partition and CSR
        call generate_partition_table(system_size, partition_table)
        call build_hypercube_csr(n_qubits, partition_table, row_starts, col_indexes, n_local, local_nnz)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        
        ! Setup graph communicator
        call setup_graph_comm(row_starts, col_indexes, partition_table, &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              in_neighbors, out_neighbors, total_recv, total_send, lb, ub)
        
        ! Preallocate buffers
        allocate(u(n_local), v(n_local), expected(n_local))
        allocate(send_buf(max(total_send, 1)), recv_buf(max(total_recv, 1)))
        
        ! Test 1: All-ones vector
        u = (1.0_dp, 0.0_dp)
        expected = cmplx(n_qubits, 0.0_dp, dp)
        
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf)
        
        max_err = maxval(abs(v - expected))
        passed = max_err < 1.0e-10_dp
        
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Test 1 (all ones): PASS'
            else
                print '(A,ES12.4)', 'Test 1 (all ones): FAIL, max_err=', max_err
            end if
        end if
        
        ! Test 2: Index vector
        do i = 1, n_local
            u(i) = cmplx(lb + i - 1, 0.0_dp, dp)
        end do
        
        expected = (0.0_dp, 0.0_dp)
        do i = 1, n_local
            do k = 1, n_qubits
                expected(i) = expected(i) + cmplx(ieor(lb + i - 1, ishft(1_int64, k - 1)), 0.0_dp, dp)
            end do
        end do
        
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf)
        
        max_err = maxval(abs(v - expected))
        passed = max_err < 1.0e-10_dp
        
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Test 2 (index vector): PASS'
            else
                print '(A,ES12.4)', 'Test 2 (index vector): FAIL, max_err=', max_err
            end if
        end if
        
        ! Test 3: Scalar multiplier
        u = (1.0_dp, 0.0_dp)
        expected = cmplx(0.0_dp, -real(n_qubits, dp), dp)  ! -i * n_qubits
        
        call spmv_sorted_rows(row_starts, col_indexes, u, v, (0.0_dp, -1.0_dp), &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              total_recv, total_send, lb, ub, send_buf, recv_buf)
        
        max_err = maxval(abs(v - expected))
        passed = max_err < 1.0e-10_dp
        
        if (rank == 0) then
            if (passed) then
                print '(A)', 'Test 3 (scalar -i): PASS'
            else
                print '(A,ES12.4)', 'Test 3 (scalar -i): FAIL, max_err=', max_err
            end if
        end if
        
        ! Cleanup
        deallocate(u, v, expected, send_buf, recv_buf)
        deallocate(partition_table, row_starts, col_indexes)
        call cleanup_graph_comm(graph_comm, recv_indices_sorted, sort_perm, &
                                recv_counts, recv_disps, send_offsets, &
                                send_counts, send_disps, in_neighbors, out_neighbors)
    end subroutine test_correctness

    !--------------------------------------------------------------------------
    ! Benchmark
    !--------------------------------------------------------------------------
    subroutine benchmark(n_qubits, n_iters)
        integer, intent(in) :: n_qubits, n_iters
        
        integer :: rank, nprocs, ierr, iter
        integer(int64) :: system_size, n_local, local_nnz, lb, ub
        integer(int64), allocatable :: partition_table(:), row_starts(:), col_indexes(:)
        complex(dp), allocatable :: u(:), v(:)
        complex(dp), allocatable :: send_buf(:), recv_buf(:)
        integer :: graph_comm, total_recv, total_send
        integer(int64), allocatable :: recv_indices_sorted(:), send_offsets(:)
        integer, allocatable :: sort_perm(:), recv_counts(:), recv_disps(:)
        integer, allocatable :: send_counts(:), send_disps(:)
        integer, allocatable :: in_neighbors(:), out_neighbors(:)
        real(dp) :: t_start, t_end, t_setup, t_spmv
        
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        
        system_size = 2_int64**n_qubits
        
        ! Setup timing
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        call generate_partition_table(system_size, partition_table)
        call build_hypercube_csr(n_qubits, partition_table, row_starts, col_indexes, n_local, local_nnz)
        
        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        
        call setup_graph_comm(row_starts, col_indexes, partition_table, &
                              graph_comm, recv_indices_sorted, sort_perm, &
                              recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                              in_neighbors, out_neighbors, total_recv, total_send, lb, ub)
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_setup = MPI_Wtime() - t_start
        
        ! Allocate and initialize (buffers allocated once, reused)
        allocate(u(n_local), v(n_local))
        allocate(send_buf(max(total_send, 1)), recv_buf(max(total_recv, 1)))
        u = (1.0_dp, 0.0_dp)
        
        ! Warmup
        do iter = 1, 2
            call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                                  graph_comm, recv_indices_sorted, sort_perm, &
                                  recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                                  total_recv, total_send, lb, ub, send_buf, recv_buf)
        end do
        
        ! Timed runs
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_start = MPI_Wtime()
        
        do iter = 1, n_iters
            call spmv_sorted_rows(row_starts, col_indexes, u, v, (1.0_dp, 0.0_dp), &
                                  graph_comm, recv_indices_sorted, sort_perm, &
                                  recv_counts, recv_disps, send_offsets, send_counts, send_disps, &
                                  total_recv, total_send, lb, ub, send_buf, recv_buf)
        end do
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        t_spmv = MPI_Wtime() - t_start
        
        if (rank == 0) then
            print '(A)', ''
            print '(A,I0,A,I0,A,I0,A)', '=== Benchmark: ', n_qubits, ' qubits, ', nprocs, ' ranks, ', n_iters, ' iterations ==='
            print '(A,F10.2,A)', 'Setup time:     ', t_setup * 1000, ' ms'
            print '(A,F10.2,A)', 'SpMV time:      ', (t_spmv / n_iters) * 1000, ' ms/iter'
            print '(A)', ''
            print '(A,I0)', 'Local rows:     ', n_local
            print '(A,I0)', 'Local NNZ:      ', local_nnz
            print '(A,I0)', 'Total recv:     ', total_recv
            print '(A,I0)', 'Total send:     ', total_send
            print '(A,I0)', 'In neighbors:   ', size(in_neighbors)
            print '(A,I0)', 'Out neighbors:  ', size(out_neighbors)
            print '(A)', ''
            print '(A,F10.4,A)', 'Recv buffer:    ', real(total_recv) * 16 / 1e6, ' MB'
            print '(A,F10.4,A)', 'Send buffer:    ', real(total_send) * 16 / 1e6, ' MB'
            print '(A,F10.4,A)', 'Comm data:      ', real(total_recv) * 12 / 1e6, ' MB (indices + perm)'
        end if
        
        ! Cleanup
        deallocate(u, v, send_buf, recv_buf)
        deallocate(partition_table, row_starts, col_indexes)
        call cleanup_graph_comm(graph_comm, recv_indices_sorted, sort_perm, &
                                recv_counts, recv_disps, send_offsets, &
                                send_counts, send_disps, in_neighbors, out_neighbors)
    end subroutine benchmark

end program test_chunked_spmv
