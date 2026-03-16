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
module one_norms

    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64
    use sparse, only: csr, spmm_graph
    use MPI

    implicit none

    private
    public :: one_norm_estimation, one_norm

contains

    function random_choice(array)

        real(real64) :: random_choice
        real(real64), dimension(:), intent(in) :: array

        real(real64) :: r

        call random_number(r)

        random_choice = array(int(r * size(array)) + 1)

    end function random_choice

    subroutine seed_random_number(seed)

        integer, intent(in) :: seed

        integer(int32) :: seed_length
        integer, dimension(:), allocatable :: seed_array

        call random_seed(size=seed_length)

        allocate (seed_array(seed_length))
        seed_array = seed

        call random_seed(put=seed_array)

        deallocate (seed_array)

    end subroutine seed_random_number

    subroutine reversed_insertion_sort_indexed(array, indices)

        real(real64), intent(inout), dimension(:) :: array
        integer, intent(inout), dimension(:) :: indices

        real(real64) :: temp
        integer(int32) :: temp_indx
        integer(int32) :: i, j

        do i = 2, size(array)

            temp = array(i)
            temp_indx = indices(i)
            j = i - 1

            do while (j >= 1)
                if (array(j) >= temp) exit
                array(j + 1) = array(j)
                indices(j + 1) = indices(j)
                j = j - 1
            end do
            array(j + 1) = temp
            indices(j + 1) = temp_indx
        end do

    end subroutine reversed_insertion_sort_indexed

    !
    !   subroutine: one_norm_estimation
    !
    !> @brief 1-norm estimation.
    !
    !> @details Estimates the matrix 1-norm of A^n where n is an integer and
    !> A is a CSR sparse complex matrix. The estimated 1-norm is returned to all
    !> nodes in the given MPI communicator.
    !>
    !> *CSR* matrix *A* must be paritioned as given by
    !> *partition_table*, see @ref sparse_parts::generate_partition_table for more.

    subroutine one_norm_estimation(A, &
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
        integer(int64), intent(in), dimension(:) :: partition_table !< @param MPI communicator partition scheme.
        real(real64), intent(out) :: est !< @param Estimated 1-norm.
        integer, intent(in) :: mpi_communicator !< @param MPI communiator handel.

        complex(real64), allocatable, dimension(:, :) :: X, Y, S, Z

        integer(int64) :: lb, ub

        integer(int32) :: sys_clock
        real(real64), dimension(2) :: plus_minus = [-1.0_real64, 1.0_real64]

        integer(int32) :: k ! current algorithm iteration

        real(real64), dimension(t) :: Y_norms_local, Y_norms
        real(real64), dimension(:), allocatable :: Z_norms_local

        real(real64) :: est_old

        integer(int32) :: ind_best

        real(real64), dimension(t) :: h_maxes_local
        integer, dimension(t) :: h_inds_local

        real(real64), dimension(:), allocatable :: h_maxes
        integer, dimension(:), allocatable :: h_inds
        real(real64) :: h_max

        integer, dimension(:), allocatable :: h_disps, h_blocks

        integer, dimension(:), allocatable :: h_inds_hist, h_inds_hist_temp

        logical :: complete

        integer, dimension(t) :: e_i
        integer(int32) :: indx

        integer(int64) :: i, j

        !MPI environment
        integer(int32) :: flock
        integer(int32) :: rank
        integer(int32) :: ierr
        integer(int32) :: MASTER = 0

        ind_best = 0

        call mpi_comm_size(mpi_communicator, flock, ierr)
        call mpi_comm_rank(mpi_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        allocate (X(lb:ub, t))
        allocate (Y(lb:ub, t))
        allocate (S(lb:ub, t))

        allocate (Z(lb:ub, t))
        allocate (Z_norms_local(lb:ub))

        allocate (h_inds_hist(0))

        if (rank == MASTER) then

            allocate (h_maxes(t * flock))
            allocate (h_inds(t * flock))
            allocate (h_disps(flock))
            allocate (h_blocks(flock))

            do i = 1, flock
                h_disps(i) = (i - 1) * t
            end do

            h_blocks = t

        else

            allocate (h_maxes(0))
            allocate (h_inds(0))
            allocate (h_disps(0))
            allocate (h_blocks(0))

        end if

        X(:, 1) = 1.0_real64 / real(A%columns, real64)

        if (t > 1) then

            call system_clock(sys_clock)
            call seed_random_number(sys_clock)

            do j = 2, t
                do i = lb, ub
                    X(i, j) = random_choice(plus_minus) / real(A%columns, real64)
                end do
            end do
        end if

        complete = .false.
        k = 1

        est_old = 0

        do

            call spmm_graph(A, &
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
                end do
            end do

            call mpi_reduce(Y_norms_local, &
                            Y_norms, &
                            t, &
                            mpi_double_precision, &
                            mpi_sum, &
                            MASTER, &
                            mpi_communicator, &
                            ierr)

            if (rank == MASTER) then
                est = maxval(Y_norms)
            end if

            call mpi_bcast(est, &
                           1, &
                           mpi_double_precision, &
                           MASTER, &
                           mpi_communicator, &
                           ierr)

            if (abs(est) < epsilon(est)) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            end if

            if (rank == MASTER) then
                if ((est > est_old) .or. (k == 2)) then
                    ind_best = maxloc(Y_norms, 1)
                end if
            end if

            if ((k >= 2) .and. (est <= est_old)) then
                est = est_old
                call mpi_barrier(mpi_communicator, ierr)
                exit
            end if

            if (k > itmax) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            end if

            est_old = est

            do j = 1, t
                do i = lb, ub
                    if (abs(Y(i, j)) < epsilon(est)) then
                        S(i, j) = 1
                    else
                        S(i, j) = Y(i, j) / abs(Y(i, j))
                    end if
                end do
            end do

            call spmm_graph(A_T, &
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
                end do
            end do

            h_maxes_local = 0
            h_inds_local = 0

            do i = 1, t
                do j = lb, ub

                    if (Z_norms_local(j) > h_maxes_local(i)) then
                        h_maxes_local(i) = Z_norms_local(j)
                        h_inds_local(i) = j
                    end if

                end do

                if (h_inds_local(i) == 0) exit

                Z_norms_local(h_inds_local(i)) = 0

            end do

            call mpi_gatherv(h_inds_local, &
                             t, &
                             mpi_integer, &
                             h_inds, &
                             h_blocks, &
                             h_disps, &
                             mpi_integer, &
                             MASTER, &
                             mpi_communicator, &
                             ierr)

            call mpi_gatherv(h_maxes_local, &
                             t, &
                             mpi_double_precision, &
                             h_maxes, &
                             h_blocks, &
                             h_disps, &
                             mpi_double_precision, &
                             MASTER, &
                             mpi_communicator, &
                             ierr)

            if (rank == MASTER) then

                call reversed_insertion_sort_indexed(h_maxes, h_inds)

                h_max = h_maxes(1)

                if ((k >= 2) .and. (h_inds(1) == ind_best)) then
                    complete = .true.
                end if

            end if

            call mpi_bcast(complete, &
                           1, &
                           mpi_logical, &
                           MASTER, &
                           mpi_communicator, &
                           ierr)

            if (complete) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            end if

            if (rank == MASTER) then

                do i = 1, t
                    do j = 1, size(h_inds_hist)
                        complete = .true.
                        if (h_inds(i) == h_inds_hist(j)) then
                            exit
                        end if
                        complete = .false.
                    end do
                end do

            end if

            call mpi_bcast(complete, &
                           1, &
                           mpi_logical, &
                           MASTER, &
                           mpi_communicator, &
                           ierr)

            if (complete) then
                call mpi_barrier(mpi_communicator, ierr)
                exit
            end if

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
                    end do

                end if

                allocate (h_inds_hist_temp(size(h_inds_hist)))

                h_inds_hist_temp = h_inds_hist

                deallocate (h_inds_hist)
                allocate (h_inds_hist(size(h_inds_hist_temp) + t))

                h_inds_hist(1:t) = e_i
                h_inds_hist(1 + t:size(h_inds_hist)) = h_inds_hist_temp

                deallocate (h_inds_hist_temp)

            end if

            call mpi_bcast(e_i, &
                           2, &
                           mpi_integer, &
                           MASTER, &
                           mpi_communicator, &
                           ierr)

            X = 0

            do i = 1, t
                if ((lb <= e_i(i)) .and. (e_i(i) <= ub)) then
                    X(e_i(i), i) = 1.0_real64
                end if
            end do

            k = k + 1

        end do

    end subroutine one_norm_estimation

    subroutine one_norm(A, &
                        norm, &
                        partition_table, &
                        MPI_communicator)

        type(CSR), intent(in) :: A
        real(real64), intent(out) :: norm
        integer(int64), dimension(:), intent(in) :: partition_table
        integer(int32), intent(in) :: MPI_communicator

        real(real64), dimension(:), allocatable :: one_norms_local, one_norms

        integer(int64) :: lb_elements, ub_elements
        integer(int64) :: i, j

        !MPI ENVIRONMENT
        integer(int32) :: rank
        integer(int32) :: ierr
        integer(int32) :: MASTER = 0

        call MPI_comm_rank(MPI_communicator, rank, ierr)

        lb_elements = A%row_starts(partition_table(rank + 1))
        ub_elements = A%row_starts(partition_table(rank + 2)) - 1

        allocate (one_norms_local(A%columns))

        if (rank == 0) then
            allocate (one_norms(A%columns))
        else
            allocate (one_norms(0))
        end if

        one_norms_local = 0

        do j = lb_elements, ub_elements
            one_norms_local(A%col_indexes(j)) = abs(A%values(j)) + &
                                                one_norms_local(A%col_indexes(j))
        end do

        call mpi_reduce(one_norms_local, &
                        one_norms, &
                        A%columns, &
                        mpi_double_precision, &
                        mpi_sum, &
                        MASTER, &
                        MPI_communicator, &
                        ierr)

        if (rank == 0) then

            norm = 0
            do i = 1, A%columns
                if (one_norms(i) > norm) then
                    norm = one_norms(i)
                end if
            end do

        end if

        call MPI_bcast(norm, &
                       1, &
                       mpi_double_precision, &
                       MASTER, &
                       MPI_communicator, &
                       ierr)

    end subroutine one_norm

end module one_norms
