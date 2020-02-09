subroutine rec_a(   M_rows, &
                    M_n_row_starts, &
                    M_n_col_indexes, &
                    M_row_starts, &
                    M_col_indexes, &
                    flock, &
                    partition_table, &
                    MPI_communicator, &
                    M_num_rec_inds, &
                    M_rec_disps, &
                    M_num_send_inds, &
                    M_send_disps)

        use :: Sparse

        implicit none

        integer, intent(in) :: M_rows
        integer, intent(in) :: M_n_row_starts
        integer, intent(in) :: M_n_col_indexes
        integer, dimension(M_n_row_starts), target, intent(in) :: M_row_starts
        integer, dimension(M_n_col_indexes), target, intent(in) :: M_col_indexes
        integer, intent(in) :: flock
        integer, dimension(flock + 1), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator
        integer, dimension(flock), target, intent(out) :: M_num_rec_inds
        integer, dimension(flock), target, intent(out) :: M_rec_disps
        integer, dimension(flock), target, intent(out) :: M_num_send_inds
        integer, dimension(flock), target, intent(out) :: M_send_disps

        type(CSR) :: M

        integer :: rank, ierr

        integer :: lb, ub
        integer :: lb_elements, ub_elements

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        m%row_starts(lb:ub + 1) => m_row_starts

        lb_elements = m%row_starts(lb)
        ub_elements = m%row_starts(ub + 1) - 1

        M%rows = M_rows
        M%columns = M_rows
        M%col_indexes(lb_elements:ub_elements) => M_col_indexes
        M%num_rec_inds(1:size(M_num_rec_inds)) => M_num_rec_inds
        M%rec_disps(1:size(M_rec_disps)) => M_rec_disps
        M%num_send_inds(1:size(M_num_send_inds)) => M_num_send_inds
        M%send_disps(1:size(M_send_disps)) => M_send_disps

        call Reconcile_Communications_A(M, partition_table, MPI_communicator)

end subroutine rec_a

subroutine rec_b(   M_rows, &
                    M_nnz, &
                    M_n_row_starts, &
                    num_send, &
                    M_row_starts, &
                    M_col_indexes, &
                    M_num_rec_inds, &
                    M_rec_disps, &
                    M_num_send_inds, &
                    M_send_disps, &
                    flock, &
                    partition_table, &
                    MPI_communicator, &
                    M_local_col_inds, &
                    M_RHS_send_inds)

        use :: Sparse

        implicit none

        integer, intent(in) :: M_rows
        integer, intent(in) :: M_nnz
        integer, intent(in) :: M_n_row_starts
        integer, intent(in) :: num_send
        integer, dimension(M_n_row_starts), target, intent(in) :: M_row_starts
        integer, dimension(M_nnz), target, intent(in) :: M_col_indexes
        integer, dimension(flock), target, intent(in) :: M_rec_disps
        integer, dimension(flock), target, intent(in) :: M_num_send_inds
        integer, dimension(flock), target, intent(in) :: M_num_rec_inds
        integer, dimension(flock), target, intent(in) :: M_send_disps
        integer, intent(in) :: flock
        integer, dimension(flock + 1), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator
        integer, dimension(M_nnz), target, intent(out) :: M_local_col_inds
        integer, dimension(num_send), target, intent(out) :: M_RHS_send_inds

        type(CSR) :: M

        integer :: rank, ierr

        integer :: lb, ub
        integer :: lb_elements, ub_elements

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        M%row_starts(lb:ub + 1) => m_row_starts

        lb_elements = M%row_starts(lb)
        ub_elements = M%row_starts(ub + 1) - 1

        M%rows = M_rows
        M%columns = M_rows
        M%col_indexes(lb_elements:ub_elements) => M_col_indexes
        M%local_col_inds(lb_elements:ub_elements) => M_local_col_inds
        M%num_rec_inds(1:size(M_num_rec_inds)) => M_num_rec_inds
        M%rec_disps(1:size(M_rec_disps)) => M_rec_disps
        M%num_send_inds(1:size(M_num_send_inds)) => M_num_send_inds
        M%send_disps(1:size(M_send_disps)) => M_send_disps
        M%RHS_send_inds(1:size(M_RHS_send_inds)) => M_RHS_send_inds

        call Reconcile_Communications_B(M, partition_table, MPI_communicator)

end subroutine rec_b

subroutine one_norm_series( M_rows, &
                            M_n_col_indexes, &
                            M_n_values, &
                            M_n_local_col_indexes, &
                            M_n_row_starts, &
                            M_sends, &
                            M_row_starts, &
                            M_col_indexes, &
                            M_values, &
                            M_num_rec_inds, &
                            M_rec_disps, &
                            M_num_send_inds, &
                            M_send_disps, &
                            M_local_col_inds, &
                            M_RHS_send_inds, &
                            flock, &
                            partition_table, &
                            MPI_communicator, &
                            one_norm_array, &
                            p)

        use :: Sparse
        use :: One_Norms
        use :: MPI
        use :: Expm

        implicit none

        integer, intent(in) :: M_rows
        integer, intent(in) :: M_n_col_indexes
        integer, intent(in) :: M_n_values
        integer, intent(in) :: M_n_local_col_indexes
        integer, intent(in) :: M_n_row_starts
        integer, intent(in) :: M_sends
        integer, dimension(M_n_row_starts), target, intent(in) :: M_row_starts
        integer, dimension(M_n_col_indexes), target, intent(in) :: M_col_indexes
        complex(8), dimension(M_n_values), target, intent(in) :: M_values
        integer, dimension(flock), target, intent(in) :: M_num_rec_inds
        integer, dimension(flock), target, intent(in) :: M_rec_disps
        integer, dimension(flock), target, intent(in) :: M_num_send_inds
        integer, dimension(flock), target, intent(in) :: M_send_disps
        integer, dimension(M_n_local_col_indexes), target, intent(in) :: M_local_col_inds
        integer, dimension(M_sends), target, intent(in) :: M_RHS_send_inds
        integer, intent(in) :: flock
        integer, dimension(flock + 1), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator
        real(8), dimension(9), intent(out) :: one_norm_array
        integer, intent(out) :: p

        type(CSR) :: M
        type(CSR) :: M_T

        integer :: itmax

        integer :: i

        integer :: rank, ierr

        integer :: lb, ub
        integer :: lb_elements, ub_elements

        integer, parameter :: l = 3
        integer, parameter :: pmax = 8

        real(8), dimension(9) :: alphas

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        M%rows = M_rows
        M%columns = M_rows
        M%row_starts(lb:ub + 1) => M_row_starts

        lb_elements = M%row_starts(lb)
        ub_elements = M%row_starts(ub + 1) - 1

        M%col_indexes(lb_elements:ub_elements) => M_col_indexes
        M%values(lb_elements:ub_elements) => M_values
        M%local_col_inds(lb_elements:ub_elements) => M_local_col_inds
        M%num_rec_inds(1:size(M_num_rec_inds)) => M_num_rec_inds
        M%rec_disps(1:size(M_rec_disps)) => M_rec_disps
        M%num_send_inds(1:size(M_num_send_inds)) => M_num_send_inds
        M%send_disps(1:size(M_send_disps)) => M_send_disps
        M%RHS_send_inds(1:size(M_RHS_send_inds)) => M_RHS_send_inds

        call CSR_Dagger(M, partition_table, M_T, MPI_communicator)

        call Reconcile_Communications(  M_T, &
                                        partition_table, &
                                        MPI_communicator)

        itmax = M%columns/l

        one_norm_array = 0

        call One_Norm(  M, &
                        one_norm_array(1), &
                        partition_table, &
                        MPI_communicator)

        one_norm_array(1) = one_norm_array(1)

        p = pmax

        do i = 2, pmax + 1

            call One_Norm_Estimation(   M, &
                                        M_T, &
                                        i, &
                                        l, &
                                        itmax, &
                                        partition_table, &
                                        one_norm_array(i), &
                                        mpi_communicator)

            alphas(i) = one_norm_array(i)**(1_dp/real(i,8))

            if (i >= 3) then
                if((abs((alphas(i - 1) - alphas(i))/alphas(i))/alphas(i) < 0.05)) then
                    p = i - 1
                    exit
                endif
            endif

        enddo

        deallocate(M_T%values, M_T%row_starts, M_T%local_col_inds, M_T%col_indexes, &
            M_T%num_rec_inds, M_T%rec_disps, M_T%num_send_inds, M_T%send_disps, M_T%RHS_send_inds)

end subroutine one_norm_series

subroutine step(M_rows, &
                M_n_col_indexes, &
                M_n_values, &
                M_n_local_col_indexes, &
                n_rho0_v, &
                n_rhot_v, &
                M_sends, &
                M_row_starts, &
                M_col_indexes, &
                M_values, &
                M_num_rec_inds, &
                M_rec_disps, &
                M_num_send_inds, &
                M_send_disps, &
                M_local_col_inds, &
                M_RHS_send_inds, &
                t, &
                rho0_v, &
                flock, &
                partition_table, &
                p, &
                one_norm_array, &
                MPI_communicator, &
                rhot_v, &
                target_precision, &
                M_n_row_starts)

        use :: Sparse
        use :: Expm

        implicit none

        integer, intent(in) :: M_rows
        integer, intent(in) :: M_n_row_starts
        integer, intent(in) :: M_n_col_indexes
        integer, intent(in) :: M_n_values
        integer, intent(in) :: M_n_local_col_indexes
        integer, intent(in) :: n_rho0_v
        integer, intent(in) :: n_rhot_v
        integer, intent(in) :: M_sends
        integer, dimension(M_n_row_starts), target, intent(in) :: M_row_starts
        integer, dimension(M_n_col_indexes), target, intent(in) :: M_col_indexes
        complex(8), dimension(M_n_values), target, intent(in) :: M_values
        integer, dimension(flock), target, intent(in) :: M_num_rec_inds
        integer, dimension(flock), target, intent(in) :: M_rec_disps
        integer, dimension(flock), target, intent(in) :: M_num_send_inds
        integer, dimension(flock), target, intent(in) :: M_send_disps
        integer, dimension(M_n_local_col_indexes), target, intent(in) :: M_local_col_inds
        integer, dimension(M_sends), target, intent(in) :: M_RHS_send_inds
        real(8), intent(in) :: t
        complex(8), dimension(n_rho0_v), intent(in) :: rho0_v
        integer, intent(in) :: flock
        integer, dimension(flock + 1), intent(in) :: partition_table
        integer, intent(inout) :: p
        real(8), dimension(9), intent(inout) :: one_norm_array
        integer, intent(in) :: MPI_communicator
        complex(8), dimension(n_rhot_v), intent(out) :: rhot_v
        character(len=2), intent(in) :: target_precision

        type(CSR) :: M

        integer :: rank, ierr

        integer :: lb, ub
        integer :: lb_elements, ub_elements

        real(8) :: start, finish

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        m%rows = m_rows
        m%columns = m_rows
        m%row_starts(lb:ub + 1) => m_row_starts

        lb_elements = m%row_starts(lb)
        ub_elements = m%row_starts(ub + 1) - 1

        M%col_indexes(lb_elements:ub_elements) => M_col_indexes
        M%values(lb_elements:ub_elements) => M_values
        M%local_col_inds(lb_elements:ub_elements) => M_local_col_inds
        M%num_rec_inds(1:size(M_num_rec_inds)) => M_num_rec_inds
        M%rec_disps(1:size(M_rec_disps)) => M_rec_disps
        M%num_send_inds(1:size(M_num_send_inds)) => M_num_send_inds
        M%send_disps(1:size(M_send_disps)) => M_send_disps
        M%RHS_send_inds(1:size(M_RHS_send_inds)) => M_RHS_send_inds

        start = MPI_wtime()

        call Expm_Multiply( M, &
                            rho0_v, &
                            t, &
                            partition_table, &
                            rhot_v, &
                            MPI_communicator, &
                            one_norm_series = one_norm_array, &
                            p = p, &
                            target_precision = target_precision)
        finish = MPI_wtime()

end subroutine step
