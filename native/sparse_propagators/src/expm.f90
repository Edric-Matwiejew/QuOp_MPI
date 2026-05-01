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

!   Module: Expm
!
!>  @brief Action of the complex matrix exponential on a vector parallalized
!>  using MPI.
!
!>  @deatils This module implements Algorithms 3.2 and 5.2 as described in
!>  "Computing the action of the matrix exponential with an application to
!>  exponential integrators" by Awad H. Al-Mohy and Nicholas J, Higham,
!>  DOI: 10.1137/100788860.

!>  @note There are two omissions. Firstly, optional balancing of the input
!>  matrix is not included. Secondly, the norm of the input matrix is not
!>  minimised via reduction of the Frobenius norm.

module expm

    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, int32, int64
    use sparse, only: cleanup_graph_communications, csr, csr_dagger, setup_graph_communications, spmv_graph
    use one_norms, only: one_norm, one_norm_estimation
    use :: MPI

    implicit none

    private

    public :: expm_multiply, expm_multiply_series, parameters

    integer, parameter :: RHV = 2, m_max = 55, p_max = 8
    integer, parameter :: int128 = selected_int_kind(38)

    real(real128), parameter :: tol_real64 = 2.0_real128**(-53.0_real128)
    real(real128), parameter :: epsilon_real64 = epsilon(real(0, real64))

    real(real64), dimension(100), target :: theta_real64 = [2.220446049250313D-16, &
                                                    2.5809568029717673D-8, 0.000013863478661191213_real64, &
                                           0.0003397168839976962_real64, 0.002400876357887274_real64, 0.009065656407595102_real64, &
                                             0.023844555325002736_real64, 0.049912288711153226_real64, 0.08957760203223342_real64, &
                                                0.14418297616143777_real64, 0.21423580684517107_real64, 0.2996158913811581_real64, &
                                                  0.3997775336316795_real64, 0.5139146936124294_real64, 0.6410835233041199_real64, &
                                                  0.7802874256626575_real64, 0.9305328460786568_real64, 1.0908637192900361_real64, &
                                                   1.2603810606426387_real64, 1.438252596804337_real64, 1.6237159502358214_real64, &
                                                    1.8160778162150857_real64, 2.014710780944616_real64, 2.21904886936509_real64, &
                                                   2.4285825244428265_real64, 2.6428534574594353_real64, 2.861449633934264_real64, &
                                                    3.084000544989162_real64, 3.310172839890271_real64, 3.5396663487436895_real64, &
                                                    3.772210495681751_real64, 4.00756108611804_real64, 4.245497442579696_real64, &
                                                    4.485819859447368_real64, 4.728347345793539_real64, 4.972915626191982_real64, &
                                                    5.219375371084059_real64, 5.467590630524544_real64, 5.717437447572013_real64, &
                                                    5.968802630041848_real64, 6.221582661689891_real64, 6.475682736079984_real64, &
                                                    6.731015898381024_real64, 6.98750228213063_real64, 7.245068429597952_real64, &
                                                    7.503646685788864_real64, 7.763174657377988_real64, 8.02359472893998_real64, &
                                                    8.284853629803916_real64, 8.546902045684934_real64, 8.809694269971322_real64, &
                                                    9.073187890176143_real64, 9.337343505612015_real64, 9.602124472826556_real64, &
                                                    9.8674966757534_real64, 10.133428317897478_real64, 10.399889734191031_real64, &
                                                  10.666853220434106_real64, 10.934292878475777_real64, 11.202184475504577_real64, &
                                                  11.470505316002537_real64, 11.739234125080184_real64, 12.008350942053166_real64, &
                                                  12.277837023246892_real64, 12.547674753126437_real64, 12.817847562946628_real64, &
                                                  13.088339856203294_real64, 13.359136940242902_real64, 13.630224963455024_real64, &
                                                   13.90159085753186_real64, 14.173222284331821_real64, 14.445107586931254_real64, &
                                                  14.717235744490083_real64, 14.989596330594331_real64, 15.262179474771681_real64, &
                                                   15.534975826905704_real64, 15.80797652430087_real64, 16.081173161174043_real64, &
                                                   16.35455776036932_real64, 16.628122747112073_real64, 16.901860924634942_real64, &
                                                   17.175765451524093_real64, 17.449829820647437_real64, 17.72404783953921_real64, &
                                                  17.998413612126303_real64, 18.272921521691813_real64, 18.547566214980513_real64, &
                                                  18.822342587358953_real64, 19.097245768950543_real64, 19.372271111672568_real64, &
                                                   19.647414177108324_real64, 19.922670725152962_real64, 20.19803670337687_real64, &
                                                   20.473508237054766_real64, 20.749081619813076_real64, 21.02475330485181_real64, &
                                                  21.300519896700695_real64, 21.576378143472468_real64, 21.852324929579098_real64, &
                                                    22.128357268879363_real64]

    real(real64), parameter :: tol_real32 = 2.0_real64**(-24.0_real64)
    real(real64), parameter :: epsilon_real32 = epsilon(real(0, real32))

    real(real64), dimension(55), target :: theta_real32 = [ &
        1.19209D-7, 0.000597886_real64, 0.0112339_real64, 0.0511662_real64, 0.130849_real64, 0.249529_real64, &
        0.401458_real64, 0.580052_real64, 0.779511_real64, 0.995184_real64, 1.22348_real64, 1.46166_real64, &
        1.70765_real64, 1.95985_real64, 2.21704_real64, 2.47828_real64, 2.74282_real64, 3.01007_real64, &
        3.27956_real64, 3.55093_real64, 3.82386_real64, 4.09811_real64, 4.37347_real64, 4.64978_real64, &
        4.9269_real64, 5.20471_real64, 5.48311_real64, 5.76201_real64, 6.04136_real64, 6.32108_real64, &
        6.60113_real64, 6.88146_real64, 7.16204_real64, 7.44283_real64, 7.7238_real64, 8.00493_real64, &
        8.2862_real64, 8.56759_real64, 8.84908_real64, 9.13065_real64, 9.4123_real64, 9.69402_real64, &
        9.97579_real64, 10.2576_real64, 10.5394_real64, 10.8213_real64, 11.1032_real64, 11.3852_real64, &
        11.6671_real64, 11.949_real64, 12.231_real64, 12.5129_real64, 12.7949_real64, 13.0769_real64, &
        13.3588_real64]

contains

    function infinity_norm(B, MPI_communicator)

        real(real64) :: infinity_norm
        complex(real64), intent(in), dimension(:) :: B
        integer, intent(in) :: MPI_communicator

        real(real64) :: inf_reduce

        integer(int32) :: i

        ! MPI ENVIRONMENT
        integer(int32) :: ierr

        if (size(B) > 0) then
            infinity_norm = abs(B(1))
        else
            infinity_norm = 0
        end if

        !$omp parallel do reduction(max:infinity_norm)
        do i = 2, size(B)

            if (abs(B(i)) > infinity_norm) then

                infinity_norm = abs(B(i))

            end if

        end do
        !$omp end parallel do

        call MPI_allreduce(infinity_norm, &
                           inf_reduce, &
                           1, &
                           mpi_double_precision, &
                           MPI_max, &
                           MPI_communicator, &
                           ierr)

        infinity_norm = inf_reduce

    end function infinity_norm

    subroutine c_m(A, &
                   t, &
                   target_precision, &
                   partition_table, &
                   m_star, &
                   s, &
                   mpi_communicator, &
                   one_norm_series, &
                   p_in)

        type(CSR), intent(inout) :: A
        real(real64), intent(in) :: t
        character(len=6), intent(in) :: target_precision
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(out) :: m_star, s
        integer, intent(in) :: mpi_communicator
        real(real64), dimension(p_max + 1), optional, intent(inout) :: one_norm_series
        integer, intent(inout), optional :: p_in

        real(real64), dimension(:), pointer :: theta => null()

        type(CSR) :: A_T

        real(real64), dimension(2:p_max + 1) :: A_norms
        integer(int128), dimension(:), allocatable :: c_array
        real(real64), dimension(2:p_max) :: alpha_array

        integer(int128):: min_c
        integer(int32) :: itmax

        integer(int32) :: num_ms_and_ps
        integer(int64), dimension(:, :), allocatable :: ms_and_ps

        integer(int32) :: p

        integer(int32) :: i, j, indx

        integer(int32) :: l

        ! MPI ENVIRONMENT
        integer(int32) :: rank
        integer(int32) :: ierr

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        if (target_precision == "real32") then
            theta => theta_real32
        elseif (target_precision == "real64") then
            theta => theta_real64
        end if

        if (present(p_in)) then
            p = p_in
        else
            p = 0
        end if

        if ((p == 0) .or. (.not. present(one_norm_series))) then

            if (partition_table(rank + 2) - partition_table(rank + 1) == 1) then
                l = 1
            else
                l = RHV
            end if

            itmax = A%columns / l

            call csr_dagger(A, &
                            partition_table, &
                            A_T, &
                            MPI_communicator)

            call setup_graph_communications(A_T, &
                                            partition_table, &
                                            MPI_communicator)

            A_norms = 0
            p = p_max
            do i = 2, p_max + 1

                call one_norm_estimation(A, &
                                         A_T, &
                                         i, &
                                         l, &
                                         itmax, &
                                         partition_table, &
                                         A_norms(i), &
                                         MPI_communicator)

                if (present(one_norm_series)) then
                    one_norm_series(i) = A_norms(i)
                end if

                A_norms(i) = A_norms(i)**(1.0_real64 / real(i, real64))

                if (i >= 3) then
                    if ((abs((A_norms(i - 1) - A_norms(i)) / A_norms(i)) < 0.5)) then
                        p = i
                        exit
                    end if
                end if

            end do

            ! Cleanup A_T graph communicator resources
            call cleanup_graph_communications(A_T)

        else

            do i = 2, p + 1

                A_norms(i) = one_norm_series(i)**(1.0_real64 / real(i, real64))

            end do

        end if

        do i = 2, p

            if (A_norms(i) > A_norms(i + 1)) then

                alpha_array(i) = t * A_norms(i)

            else

                alpha_array(i) = t * A_norms(i + 1)

            end if

        end do

        num_ms_and_ps = 0
        do i = 2, p
            do j = i * (i - 1) - 1, m_max
                num_ms_and_ps = num_ms_and_ps + 1
            end do
        end do

        allocate (ms_and_ps(num_ms_and_ps, 2))

        indx = 1
        do i = 2, p

            do j = (i * (i - 1) - 1), m_max

                ms_and_ps(indx, 1) = int(j, int64)
                ms_and_ps(indx, 2) = int(i, int64)
                indx = indx + 1

            end do

        end do

        allocate (c_array(size(ms_and_ps, 1)))

        do i = 1, size(ms_and_ps, 1)

            c_array(i) = ms_and_ps(i, 1) * ceiling(alpha_array(ms_and_ps(i, 2)) &
                                                   / theta(ms_and_ps(i, 1)), kind=int128)

        end do

        min_c = minval(c_array)
        m_star = int(ms_and_ps(minloc(c_array, 1), 1), kind=int32)
        s = max(int(min_c / m_star, kind=int32), 1)

    end subroutine c_m

    subroutine parameters(A, &
                          t, &
                          target_precision, &
                          partition_table, &
                          m_star, &
                          s, &
                          mpi_communicator, &
                          one_norm_series, &
                          p)

        type(CSR), intent(inout) :: A
        real(real64), intent(in) :: t
        character(len=6), intent(in) :: target_precision
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(out) :: m_star, s
        integer, intent(in) :: mpi_communicator
        real(real64), optional, dimension(p_max + 1), intent(inout) :: one_norm_series
        integer, optional, intent(inout) :: p

        real(real64), dimension(:), pointer :: theta

        real(real64) :: A_norm

        integer(int64) :: m_temp_1, m_temp_2

        integer(int32) :: m

        ! MPI ENVIRONMENT
        integer(int32) :: rank
        integer(int32) :: ierr

        A_norm = 0

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        if (target_precision == "real32") then
            theta => theta_real32
        elseif (target_precision == "real64") then
            theta => theta_real64
        end if

        if (present(p) .and. present(one_norm_series)) then

            if (p /= 0) then
                A_norm = one_norm_series(1)
            end if

        else

            call one_norm(A, &
                          A_norm, &
                          partition_table, &
                          MPI_communicator)

            if (present(one_norm_series)) then
                one_norm_series(1) = A_norm
            end if

        end if

        if (t * A_norm <= &
            (2 * 1 * (theta(m_max) / real(m_max, real64)) * (p_max + 3) * p_max)) then

            m_temp_1 = ceiling(t * A_norm / theta(1), kind=int64)

            do m = 2, m_max

                m_temp_2 = m * ceiling(t * A_norm / theta(m), kind=int64)

                if (m_temp_2 < m_temp_1) then

                    m_star = m

                end if

                m_temp_1 = m_temp_2

            end do

            s = int(ceiling(t * A_norm / theta(m_star), kind=int64), kind=int32)

        else

            if (present(one_norm_series)) then

                call c_m(A, &
                         t, &
                         target_precision, &
                         partition_table, &
                         m_star, &
                         s, &
                         mpi_communicator, &
                         one_norm_series=one_norm_series, &
                         p_in=p)

            else

                call c_m(A, &
                         t, &
                         target_precision, &
                         partition_table, &
                         m_star, &
                         s, &
                         mpi_communicator)

            end if

        end if

    end subroutine parameters

    subroutine expm_multiply(A, &
                             B, &
                             t, &
                             partition_table, &
                             C, &
                             mpi_communicator, &
                             one_norm_series, &
                             p, &
                             target_precision)

        type(CSR), intent(inout) :: A
        complex(real64), intent(in), dimension(:) :: B
        real(real64), intent(in) :: t
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(in) :: mpi_communicator
        complex(real64), dimension(:), intent(inout) :: C
        real(real64), dimension(p_max + 1), optional, intent(inout) :: one_norm_series
        integer, optional, intent(inout) :: p
        character(len=6), optional, intent(in) :: target_precision

        complex(real64), dimension(:), allocatable :: B_temp_1, B_temp_2

        type(CSR) :: A_temp

        integer(int32) :: m_star, s

        character(len=6) :: set_target_precision
        real(real128) :: tol, epsilon_tol
        real(real128) :: c_1, c_2

        integer(int32) :: i, j, lb, ub

        integer(int32) :: rank
        integer(int32) :: ierr

        tol = 0

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        if (present(target_precision)) then
            if (target_precision == "real32") then
                set_target_precision = target_precision
                tol = tol_real32
                epsilon_tol = epsilon_real32
            elseif (target_precision == "real64") then
                set_target_precision = target_precision
                tol = tol_real64
                epsilon_tol = epsilon_real64
            end if
        else
            set_target_precision = "real64"
            tol = tol_real64
            epsilon_tol = epsilon_real64
        end if

        lb = lbound(C, 1)
        ub = ubound(C, 1)

        allocate (B_temp_1(lb:ub))
        allocate (B_temp_2(lb:ub))

        if (abs(t) < epsilon_tol) then
            m_star = 0
            s = 1
        else

            if (present(one_norm_series) .and. present(p)) then

                call parameters(A, &
                                t, &
                                set_target_precision, &
                                partition_table, &
                                m_star, &
                                s, &
                                mpi_communicator, &
                                one_norm_series=one_norm_series, &
                                p=p)

            else

                call parameters(A, &
                                t, &
                                set_target_precision, &
                                partition_table, &
                                m_star, &
                                s, &
                                mpi_communicator)

            end if

        end if

        A_temp%rows = A%rows
        A_temp%columns = A%columns
        A_temp%row_starts => A%row_starts
        A_temp%col_indexes => A%col_indexes
        ! Share graph communicator data - copy scalars
        A_temp%graph_comm = A%graph_comm
        A_temp%total_recv = A%total_recv
        A_temp%total_send = A%total_send
        A_temp%lb_graph = A%lb_graph
        A_temp%ub_graph = A%ub_graph
        A_temp%graph_comm_ready = A%graph_comm_ready
        A_temp%col_indexes => A%col_indexes

        ! Copy allocatable graph comm arrays (shallow copy for arrays we don't modify)
        if (allocated(A%recv_indices_sorted)) then
            allocate (A_temp%recv_indices_sorted, source=A%recv_indices_sorted)
        end if
        if (associated(A%sort_perm)) then
            allocate (A_temp%sort_perm(size(A%sort_perm)))
            A_temp%sort_perm = A%sort_perm
        end if
        if (allocated(A%graph_recv_counts)) then
            allocate (A_temp%graph_recv_counts, source=A%graph_recv_counts)
        end if
        if (allocated(A%graph_recv_disps)) then
            allocate (A_temp%graph_recv_disps, source=A%graph_recv_disps)
        end if
        if (associated(A%send_offsets)) then
            allocate (A_temp%send_offsets(size(A%send_offsets)))
            A_temp%send_offsets = A%send_offsets
        end if
        if (allocated(A%graph_send_counts)) then
            allocate (A_temp%graph_send_counts, source=A%graph_send_counts)
        end if
        if (allocated(A%graph_send_disps)) then
            allocate (A_temp%graph_send_disps, source=A%graph_send_disps)
        end if
        if (allocated(A%in_neighbors)) then
            allocate (A_temp%in_neighbors, source=A%in_neighbors)
        end if
        if (allocated(A%out_neighbors)) then
            allocate (A_temp%out_neighbors, source=A%out_neighbors)
        end if
        if (associated(A%send_buf)) then
            allocate (A_temp%send_buf(size(A%send_buf)))
            A_temp%send_buf = A%send_buf
        end if
        if (associated(A%recv_buf)) then
            allocate (A_temp%recv_buf(size(A%recv_buf)))
            A_temp%recv_buf = A%recv_buf
        end if
        if (associated(A%row_starts_local)) then
            allocate (A_temp%row_starts_local(size(A%row_starts_local)))
            A_temp%row_starts_local = A%row_starts_local
        end if
        if (associated(A%col_indexes_local)) then
            allocate (A_temp%col_indexes_local(size(A%col_indexes_local)))
            A_temp%col_indexes_local = A%col_indexes_local
            ! col_halo aliases col_indexes_local in the halo-based design
            A_temp%col_halo => A_temp%col_indexes_local
            A_temp%owns_col_halo = .false.
        end if
        if (associated(A%diag_lo)) then
            allocate (A_temp%diag_lo(size(A%diag_lo)))
            A_temp%diag_lo = A%diag_lo
        end if
        if (associated(A%diag_hi)) then
            allocate (A_temp%diag_hi(size(A%diag_hi)))
            A_temp%diag_hi = A%diag_hi
        end if

        allocate (A_temp%values(A%row_starts(partition_table(rank + 1)): &
                                A%row_starts(partition_table(rank + 2)) - 1))

        do i = A%row_starts(lbound(A%row_starts, 1)), &
            A%row_starts(ubound(A%row_starts, 1)) - 1
            A_temp%values(i) = t * A%values(i)
        end do

        ! Create scaled values_local for spmv_graph
        if (associated(A%values_local)) then
            allocate (A_temp%values_local(size(A%values_local)))
            A_temp%values_local = t * A%values_local
        end if

        B_temp_1(lb:ub) = B(lb:ub)
        C(lb:ub) = B(lb:ub)

        do i = 1, s

            c_1 = infinity_norm(B_temp_1, MPI_communicator)

            do j = 1, m_star

                call spmv_graph(A_temp, &
                                B_temp_1(lb:ub), &
                                partition_table, &
                                rank, &
                                B_temp_2(lb:ub), &
                                mpi_communicator=mpi_communicator)

                B_temp_2(lb:ub) = B_temp_2(lb:ub) / real(s * j, real64)

                c_2 = infinity_norm(B_temp_2, MPI_communicator)

                C(lb:ub) = C(lb:ub) + B_temp_2(lb:ub)

                if ((c_1 + c_2) <= &
                    (tol * infinity_norm(C, MPI_communicator))) then
                    exit
                end if

                B_temp_1(lb:ub) = B_temp_2(lb:ub)

                c_1 = c_2

            end do

            B_temp_1(lb:ub) = C(lb:ub)

        end do

        ! Cleanup A_temp graph comm arrays (allocatable and pointer)
        if (allocated(A_temp%recv_indices_sorted)) deallocate (A_temp%recv_indices_sorted)
        if (associated(A_temp%sort_perm)) deallocate (A_temp%sort_perm)
        if (allocated(A_temp%graph_recv_counts)) deallocate (A_temp%graph_recv_counts)
        if (allocated(A_temp%graph_recv_disps)) deallocate (A_temp%graph_recv_disps)
        if (associated(A_temp%send_offsets)) deallocate (A_temp%send_offsets)
        if (allocated(A_temp%graph_send_counts)) deallocate (A_temp%graph_send_counts)
        if (allocated(A_temp%graph_send_disps)) deallocate (A_temp%graph_send_disps)
        if (allocated(A_temp%in_neighbors)) deallocate (A_temp%in_neighbors)
        if (allocated(A_temp%out_neighbors)) deallocate (A_temp%out_neighbors)
        if (associated(A_temp%send_buf)) deallocate (A_temp%send_buf)
        if (associated(A_temp%recv_buf)) deallocate (A_temp%recv_buf)
        if (associated(A_temp%row_starts_local)) deallocate (A_temp%row_starts_local)
        ! col_halo aliases col_indexes_local; only deallocate the underlying buffer.
        nullify (A_temp%col_halo)
        if (associated(A_temp%col_indexes_local)) deallocate (A_temp%col_indexes_local)
        if (associated(A_temp%diag_lo)) deallocate (A_temp%diag_lo)
        if (associated(A_temp%diag_hi)) deallocate (A_temp%diag_hi)
        if (associated(A_temp%values_local)) deallocate (A_temp%values_local)

        deallocate (A_temp%values)

    end subroutine expm_multiply

    subroutine expm_multiply_series(A, &
                                    B, &
                                    t0, &
                                    tq, &
                                    steps, &
                                    partition_table, &
                                    X, &
                                    mpi_communicator, &
                                    one_norm_series_in, &
                                    p_ex, &
                                    target_precision)

        real(real64), intent(in) :: t0, tq
        type(CSR), intent(inout) :: A
        complex(real64), intent(in), dimension(:) :: B
        integer, intent(in) :: steps
        integer(int64), dimension(:), intent(in) :: partition_table
        complex(real64), dimension(:, :), intent(out) :: X
        integer, intent(in) :: mpi_communicator
        real(real64), dimension(p_max + 1), optional, intent(inout) :: one_norm_series_in
        integer, optional, intent(inout) :: p_ex
        character(len=6), optional, intent(in) :: target_precision

        real(real64), dimension(p_max + 1) :: one_norm_series
        integer(int32) :: p_in

        integer(int32) :: q

        complex(real64), dimension(:), allocatable :: Z, F
        complex(real64), dimension(:, :), allocatable :: K
        real(real64) :: h

        character(len=6) :: set_target_precision
        real(real128) :: tol, epsilon_tol
        real(real128) :: c_1, c_2

        integer(int32) :: m_star, s, m_hat
        integer(int32) :: d, j, r, d_tilde, p

        ! MPI ENVIRONMENT
        integer(int32) :: rank
        integer(int32) :: ierr

        integer(int32) :: i, kay, indx
        integer(int64) :: lb, ub

        tol = 0
        indx = 0

        call mpi_comm_rank(MPI_communicator, rank, ierr)

        if (present(target_precision)) then
            if (target_precision == "real32") then
                set_target_precision = target_precision
                tol = tol_real32
                epsilon_tol = epsilon_real32
            elseif (target_precision == "real64") then
                set_target_precision = target_precision
                tol = tol_real64
                epsilon_tol = epsilon_real64
            end if
        else
            set_target_precision = "real64"
            tol = tol_real64
            epsilon_tol = epsilon_real64
        end if

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        allocate (Z(ub - lb + 1))
        allocate (F(ub - lb + 1))

        if (present(one_norm_series_in) .and. present(p_ex)) then
            one_norm_series = one_norm_series_in
            p_in = p_ex
        else
            p_in = 0
        end if

        call expm_multiply(A, &
                           B, &
                           t0, &
                           partition_table, &
                           X(:, 1), &
                           mpi_communicator, &
                           one_norm_series=one_norm_series, &
                           p=p_in, &
                           target_precision=set_target_precision)

        if (steps == 0) return

        q = steps
        h = (tq - t0) / real(q, real64)

        call parameters(A, &
                        tq - t0, &
                        set_target_precision, &
                        partition_table, &
                        m_star, &
                        s, &
                        mpi_communicator, &
                        one_norm_series=one_norm_series, &
                        p=p_in)

        if (q <= s) then

            do kay = 1, q

                call expm_multiply(A, &
                                   X(:, kay), &
                                   h, &
                                   partition_table, &
                                   X(:, kay + 1), &
                                   MPI_communicator, &
                                   one_norm_series=one_norm_series, &
                                   p=p_in, &
                                   target_precision=set_target_precision)

            end do

            return

        end if

        d = floor(real(q) / real(s))
        j = floor(real(q) / real(d))
        r = q - d * j
        d_tilde = d

        call c_m(A, &
                 real(d, real64), &
                 set_target_precision, &
                 partition_table, &
                 m_star, &
                 s, &
                 mpi_communicator, &
                 one_norm_series=one_norm_series, &
                 p_in=p_in)

        Z = X(:, 1)

        allocate (K(ub - lb + 1, m_star + 1))

        K = 0

        do i = 1, j + 1

            if (i > j) then
                d_tilde = r
            end if

            K(:, 1) = Z
            k(:, 2:m_star + 1) = 0

            m_hat = 0

            do kay = 1, d_tilde

                F = Z

                c_1 = infinity_norm(Z, MPI_communicator)

                do p = 1, m_star

                    if (p > m_hat) then

                        call spmv_graph(A, &
                                        K(:, p), &
                                        partition_table, &
                                        rank, &
                                        K(:, p + 1), &
                                        mpi_communicator=MPI_communicator)

                        K(:, p + 1) = h * K(:, p + 1) / real(p, real64)

                    end if

                    F = F + (real(kay, real64)**real(p, real64)) * K(:, p + 1)

                    c_2 = (real(kay, real128)**real(p, real64)) &
                          * infinity_norm(K(:, p + 1), MPI_communicator)

                    indx = p

                    if ((c_1 + c_2) <= &
                        (tol * infinity_norm(F, MPI_communicator))) then

                        exit

                    end if

                    c_1 = c_2

                end do

                m_hat = max(m_hat, indx)

                X(:, kay + (i - 1) * d + 1) = F

            end do

            if (i <= j) then
                Z = X(:, i * d + 1)
            end if

        end do

    end subroutine expm_multiply_series

end module expm
