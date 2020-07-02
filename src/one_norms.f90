!   QSW_MPI -  A package for parallel Quantum Stochastic Walk simulation.
!   Copyright (C) 2019 Edric Matwiejew
!
!   This program is free software: you can redistribute it and/or modify
!   it under the terms of the GNU General Public License as published by
!   the Free Software Foundation, either version 3 of the License, or
!   (at your option) any later version.
!
!   This program is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!   GNU General Public License for more details.
!
!   You should have received a copy of the GNU General Public License
!   along with this program.  If not, see <https://www.gnu.org/licenses/>.

! Module Linalg_One_Norm
!> @brief MPI parallel CSR 1-norm estimation.
!
module One_Norms

    use ISO_Precisions
    use Sparse
    use MPI

    implicit none

    contains

    function  Random_Choice(array)

        real(dp) :: Random_Choice
        real(dp), dimension(:) :: array

        real(dp) :: r

        call random_number(r)

        Random_Choice = array(int(r*size(array)) + 1)

    end function Random_Choice

    subroutine  Seed_Random_Number(seed)

        integer, intent(in) :: seed

        integer :: seed_length
        integer, dimension(:), allocatable :: seed_array

        call random_seed(size=seed_length)

        allocate(seed_array(seed_length))
        seed_array = seed

        call random_seed(put=seed_array)

        deallocate(seed_array)

    end subroutine Seed_Random_Number

    subroutine reversed_insertion_sort_indexed(array, indices)

        real(dp), intent(inout), dimension(:) :: array
        integer, intent(inout), dimension(:) :: indices

        real(dp) :: temp
        integer :: temp_indx
        integer :: i, j

        do i = 2, size(array)

            temp = array(i)
            temp_indx = indices(i)
            j = i - 1

            do while (j >= 1)
                if (array(j) >= temp) exit
                    array(j + 1) = array(j)
                    indices(j + 1) = indices(j)
                    j = j - 1
            enddo
            array(j + 1) = temp
            indices(j + 1) = temp_indx
        enddo

    end subroutine reversed_insertion_sort_indexed

    !
    !   subroutine: One_Norm_Estimation
    !
    !> @brief 1-norm estimation.
    !
    !> @details Estimates the matrix 1-norm of A^n where n is an integer and
    !> A is a CSR sparse complex matrix. The estimated 1-norm is returned to all
    !> nodes in the given MPI communicator.
    !>
    !> *CSR* matrix *A* must be paritioned as given by
    !> *partition_table*, see @ref sparse_parts::generate_partition_table for more.

    subroutine One_Norm_Estimation( A, &
                                    A_T, &
                                    n, &
                                    t, &
                                    itmax, &
                                    partition_table, &
                                    est, &
                                    mpi_communicator)

        type(CSR), intent(inout) :: A !< @param Local CSR array partition.
        type(CSR), intent(inout) :: A_T
        integer, intent(in) :: n !< @param Exponent on A.
        integer, intent(in) :: t !< @param Right hand side matrix columns.
        integer, intent(in) :: itmax !< @param Maximum permitted iterations.
        integer, intent(in), dimension(:) :: partition_table !< @param MPI communicator partition scheme.
        real(dp), intent(out) :: est !< @param Estimated 1-norm.
        integer, intent(in) :: mpi_communicator !< @param MPI communiator handel.

        complex(dp), allocatable, dimension(:,:) :: X, Y, S, Z

        integer :: lb, ub

        integer :: sys_clock
        real(dp), dimension(2) :: plus_minus = [-1_dp, 1_dp]

        integer :: k ! current algorithm iteration

        real(dp), dimension(t) :: Y_norms_local, Y_norms
        real(dp), dimension(:), allocatable :: Z_norms_local

        real(dp) :: est_old

        integer :: ind_best

        real(dp), dimension(t) :: h_maxes_local
        integer, dimension(t) :: h_inds_local

        real(dp), dimension(:), allocatable :: h_maxes
        integer, dimension(:), allocatable :: h_inds
        real(dp) :: h_max

        integer, dimension(:), allocatable :: h_disps, h_blocks

        integer, dimension(:), allocatable :: h_inds_hist, h_inds_hist_temp

        logical :: complete

        integer, dimension(t) :: e_i
        integer :: indx

        integer :: i, j

        !MPI environment
        integer :: flock
        integer :: rank
        integer :: ierr
        integer :: MASTER = 0

        ind_best = 0

        call mpi_comm_size(mpi_communicator, flock, ierr)
        call mpi_comm_rank(mpi_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        allocate(X(lb:ub, t))
        allocate(Y(lb:ub, t))
        allocate(S(lb:ub, t))

        allocate(Z(lb:ub, t))
        allocate(Z_norms_local(lb:ub))

        allocate(h_inds_hist(0))

        if (rank == MASTER) then

            allocate(h_maxes(t*flock))
            allocate(h_inds(t*flock))
            allocate(h_disps(flock))
            allocate(h_blocks(flock))

            do i = 1, flock
                h_disps(i) = (i - 1)*t
            enddo

            h_blocks = t

        else

            allocate(h_maxes(0))
            allocate(h_inds(0))
            allocate(h_disps(0))
            allocate(h_blocks(0))

        endif

        X(:,1) = 1_dp/real(A%columns, dp)

        if (t > 1) then

            call system_clock(sys_clock)
            call Seed_Random_Number(sys_clock)

            do j = 2, t
                do i = lb, ub
                    X(i, j) = Random_Choice(plus_minus)/real(A%columns, dp)
                enddo
            enddo
        endif

        complete = .false.
        k = 1

        est_old = 0

        do

            call SpMM(  A, &
                        n, &
                        X, &
                        partition_table, &
                        rank, &
                        Y, &
                        MPI_communicator)

            Y_norms_local = 0

            do j = 1, t
                do i = lb, ub
                    Y_norms_local(j) = Y_norms_local(j) + abs(Y(i, j))
                enddo
            enddo

            call mpi_reduce(    Y_norms_local, &
                                Y_norms, &
                                t, &
                                mpi_double, &
                                mpi_sum, &
                                MASTER, &
                                mpi_communicator, &
                                ierr)

            if (rank == MASTER) then
                est = maxval(Y_norms)
            endif

            call mpi_bcast( est, &
                            1, &
                            mpi_double, &
                            MASTER, &
                            mpi_communicator, &
                            ierr)

            if (abs(est) < epsilon(est)) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            endif

            if (rank == MASTER) then
                if ((est > est_old) .or. (k == 2)) then
                    ind_best = maxloc(Y_norms, 1)
                endif
            endif

            if ((k >= 2) .and. (est <= est_old)) then
                est = est_old
                call mpi_barrier(mpi_communicator, ierr)
                exit
            endif

            if (k > itmax) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            endif

            est_old = est

            do j = 1, t
                do i = lb, ub
                    if (abs(Y(i, j)) < epsilon(est)) then
                        S(i, j) = 1
                    else
                       S(i,j) = Y(i, j)/abs(Y(i, j))
                    endif
                enddo
            enddo

            call SpMM(  A_T, &
                        n, &
                        S, &
                        partition_table, &
                        rank, &
                        Z, &
                        MPI_communicator)

            Z_norms_local = 0

            do j = 1, t
                do i = lb, ub
                    Z_norms_local(i) = Z_norms_local(i) + abs(Z(i, j))
                enddo
            enddo

            h_maxes_local = 0
            h_inds_local = 0

            do i = 1, t
                do j = lb, ub

                    if (Z_norms_local(j) > h_maxes_local(i)) then
                        h_maxes_local(i) = Z_norms_local(j)
                        h_inds_local(i) = j
                    endif

                enddo

                if (h_inds_local(i) == 0) exit

                Z_norms_local(h_inds_local(i)) = 0

            enddo

            call mpi_gatherv(   h_inds_local, &
                                t, &
                                mpi_integer, &
                                h_inds, &
                                h_blocks, &
                                h_disps, &
                                mpi_integer, &
                                MASTER, &
                                mpi_communicator, &
                                ierr)

            call mpi_gatherv(   h_maxes_local, &
                                t, &
                                mpi_double, &
                                h_maxes, &
                                h_blocks, &
                                h_disps, &
                                mpi_double, &
                                MASTER, &
                                mpi_communicator, &
                                ierr)

            if (rank == MASTER) then

                call reversed_insertion_sort_indexed(h_maxes, h_inds)

                h_max = h_maxes(1)

                if ((k >= 2) .and. (h_inds(1) == ind_best)) then
                        complete = .true.
                endif

            endif

            call mpi_bcast( complete, &
                            1, &
                            mpi_logical, &
                            MASTER, &
                            mpi_communicator, &
                            ierr)


            if (complete) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            endif

            if (rank == MASTER) then

                do i = 1, t
                    do j = 1, size(h_inds_hist)
                        complete = .true.
                        if (h_inds(i) == h_inds_hist(j)) then
                            exit
                        endif
                        complete = .false.
                    enddo
                enddo

            endif

            call mpi_bcast( complete, &
                            1, &
                            mpi_logical, &
                            MASTER, &
                            mpi_communicator, &
                            ierr)

            if (complete) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            endif

            indx = 1
            e_i = 0

            if (rank == MASTER) then

                if (k == 1) then
                    e_i = h_inds(1:t)
                else

                    indx = 1
                    do i = 1, size(h_inds)
                        if (any(h_inds_hist == h_inds(i))) cycle
                        e_i(indx) = h_inds(i)
                        indx = indx + 1
                        if (indx == t) exit
                    enddo

                endif


                    allocate(h_inds_hist_temp(size(h_inds_hist)))

                    h_inds_hist_temp = h_inds_hist

                    deallocate(h_inds_hist)
                    allocate(h_inds_hist(size(h_inds_hist_temp) + t))

                    h_inds_hist(1:t) = e_i
                    h_inds_hist(1 + t:size(h_inds_hist)) = h_inds_hist_temp

                    deallocate(h_inds_hist_temp)

            endif

            call mpi_bcast( e_i, &
                            2, &
                            mpi_integer, &
                            MASTER, &
                            mpi_communicator, &
                            ierr)

            X = 0

            do i = 1, t
                if ((lb <= e_i(i)) .and. (e_i(i) <= ub)) then
                    X(e_i(i), i) = 1.0_dp
                endif
            enddo

            k = k + 1

        enddo

    end subroutine One_Norm_Estimation

    subroutine One_Norm(A, &
                        norm, &
                        partition_table, &
                        MPI_communicator)

        type(CSR), intent(in) :: A
        real(dp) :: norm
        integer, dimension(:), intent(in) :: partition_table
        integer :: MPI_communicator

        real(dp), dimension(:), allocatable :: one_norms_local, one_norms

        integer :: lb_elements, ub_elements
        integer :: i, j

        !MPI ENVIRONMENT
        integer :: rank
        integer :: ierr
        integer :: MASTER = 0

        call MPI_comm_rank(MPI_comm_world, rank, ierr)

        lb_elements = A%row_starts(partition_table(rank + 1))
        ub_elements = A%row_starts(partition_table(rank + 2)) - 1

        allocate(one_norms_local(A%columns))

        if (rank == 0) then
            allocate(one_norms(A%columns))
        else
            allocate(one_norms(0))
        endif

        one_norms_local = 0

        do j = lb_elements, ub_elements
            one_norms_local(A%col_indexes(j)) = abs(A%values(j)) + &
                one_norms_local(A%col_indexes(j))
        enddo

        call mpi_reduce(one_norms_local, &
                        one_norms, &
                        A%columns, &
                        mpi_double, &
                        mpi_sum, &
                        MASTER, &
                        MPI_communicator, &
                        ierr)

        if (rank == 0) then

            norm = 0
            do i = 1, A%columns
                if (one_norms(i) > norm) then
                    norm = one_norms(i)
                endif
            enddo

        endif

        call MPI_bcast(norm, &
                        1, &
                        mpi_double, &
                        MASTER, &
                        MPI_communicator, &
                        ierr)

    end subroutine


end module One_Norms
