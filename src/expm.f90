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

module Expm

    use :: ISO_Precisions
    use :: Sparse
    use :: One_Norms
    use :: MPI

    implicit none

    private

    public :: Expm_Multiply!, Expm_Multiply_Series

    integer, parameter :: RHV = 2, m_max = 55, p_max = 8

    real(qp), parameter :: tol_dp = 2.0_qp**(-53.0_qp)

    real(dp), dimension(55), target :: theta_dp = [2.220446049250313D-16, &
        2.5809568029717673D-8, 1.3863478661191213D-5,  3.397168839976962D-4,  &
        2.400876357887274D-3, 9.065656407595102D-3,   2.3844555325002736D-2,  &
        4.9912288711153226D-2, 8.957760203223342D-2,   1.4418297616143777D-1, &
        2.1423580684517107D-1, 2.996158913811581D-1,   3.997775336316795D-1,  &
        5.139146936124294D-1, 6.410835233041199D-1,   7.802874256626575D-1,   &
        9.305328460786568D-1, 1.0908637192900361_dp,  1.2603810606426387_dp,  &
        1.438252596804337_dp, 1.6237159502358214_dp,  1.8160778162150857_dp,  &
        2.014710780944616_dp, 2.21904886936509_dp,    2.4285825244428265_dp,  &
        2.6428534574594353_dp, 2.861449633934264_dp,   3.084000544989162_dp,  &
        3.310172839890271_dp, 3.5396663487436895_dp,  3.772210495681751_dp,   &
        4.00756108611804_dp, 4.245497442579696_dp,   4.485819859447368_dp,    &
        4.728347345793539_dp, 4.972915626191982_dp,   5.219375371084059_dp,   &
        5.467590630524544_dp, 5.717437447572013_dp,   5.968802630041848_dp,   &
        6.221582661689891_dp, 6.475682736079984_dp,   6.731015898381024_dp,   &
        6.98750228213063_dp, 7.245068429597952_dp,   7.503646685788864_dp,    &
        7.763174657377988_dp, 8.02359472893998_dp,    8.284853629803916_dp,   &
        8.546902045684934_dp, 8.809694269971322_dp,   9.073187890176143_dp,   &
        9.337343505612015_dp, 9.602124472826556_dp,   9.8674966757534_dp]

    real(qp), parameter :: tol_sp = 2.0_dp**(-24.0_dp)

    real(dp), dimension(55), target :: theta_sp = [1.19209D-7, 0.000597886_dp, &
        0.0112339_dp, 0.0511662_dp, 0.130849_dp, 0.249529_dp, 0.401458_dp,     &
        0.580052_dp, 0.779511_dp, 0.995184_dp, 1.22348_dp, 1.46166_dp,         &
        1.70765_dp, 1.95985_dp, 2.21704_dp, 2.47828_dp, 2.74282_dp, 3.01007_dp,&
        3.27956_dp, 3.55093_dp, 3.82386_dp, 4.09811_dp, 4.37347_dp, 4.64978_dp,&
        4.9269_dp, 5.20471_dp, 5.48311_dp, 5.76201_dp, 6.04136_dp, 6.32108_dp, &
        6.60113_dp, 6.88146_dp, 7.16204_dp, 7.44283_dp, 7.7238_dp, 8.00493_dp, &
        8.2862_dp, 8.56759_dp, 8.84908_dp, 9.13065_dp, 9.4123_dp, 9.69402_dp,  &
        9.97579_dp, 10.2576_dp, 10.5394_dp, 10.8213_dp, 11.1032_dp, 11.3852_dp,&
        11.6671_dp, 11.949_dp, 12.231_dp, 12.5129_dp, 12.7949_dp, 13.0769_dp,  &
        13.3588_dp]

    contains

    function Infinity_Norm(B, MPI_communicator)

        real(dp) :: Infinity_Norm
        complex(dp), intent(in), dimension(:) :: B
        integer, intent(in) :: MPI_communicator

        real(dp) :: inf_reduce

        integer :: i

        ! MPI ENVIRONMENT
        integer :: ierr

        if (size(B)>0) then
            Infinity_Norm = abs(B(1))
        else
            Infinity_Norm = 0
        endif

        do i = 2, size(B)

            if (abs(B(i)) > Infinity_Norm) then

                Infinity_Norm = abs(B(i))

            endif

        enddo

        call MPI_allreduce( Infinity_Norm, &
                            inf_reduce, &
                            1, &
                            MPI_double, &
                            MPI_max, &
                            MPI_communicator, &
                            ierr)

        Infinity_Norm = inf_reduce

    end function Infinity_Norm

    subroutine C_m( A, &
                    t, &
                    target_precision, &
                    partition_table, &
                    m_star, &
                    s, &
                    mpi_communicator, &
                    one_norm_series, &
                    p_in)

        type(CSR), intent(inout) :: A
        real(dp), intent(in) :: t
        character(len = 2), intent(in) :: target_precision
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(out) :: m_star, s
        integer, intent(in) :: mpi_communicator
        real(dp), dimension(p_max + 1), optional, intent(inout) :: one_norm_series
        integer, intent(inout), optional :: p_in

        real(dp), dimension(:), pointer :: theta

        type(CSR) :: A_T

        real(dp), dimension(2:p_max + 1) :: A_norms
        integer(qp), dimension(:), allocatable :: c_array
        real(dp), dimension(2:p_max) :: alpha_array

        integer :: itmax, min_c

        integer :: num_ms_and_ps
        integer, dimension(:,:), allocatable :: ms_and_ps

        integer :: p

        integer :: i, j , indx

        integer :: l

        ! MPI ENVIRONMENT
        integer :: rank
        integer :: ierr

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        if (target_precision == "sp") then
            theta => theta_sp
        elseif (target_precision == "dp") then
            theta => theta_dp
        endif

        if (present(p_in)) then
            p = p_in
        else
            p = 0
        endif

        if ((p == 0) .or. (.not. present(one_norm_series))) then

            if (partition_table(rank + 2) - partition_table(rank + 1) == 1) then
                l = 1
            else
                l = RHV
            endif

            itmax = A%columns/l


            call CSR_Dagger(A, &
                            partition_table, &
                            A_T, &
                            MPI_communicator)

            call Reconcile_Communications(  A_T, &
                                            partition_table, &
                                            MPI_communicator)

            A_norms = 0
            p = p_max
            do i = 2, p_max + 1

                call One_Norm_Estimation(   A, &
                                            A_T, &
                                            i, &
                                            l, &
                                            itmax, &
                                            partition_table, &
                                            A_norms(i), &
                                            MPI_communicator)

                if (present(one_norm_series)) then
                    one_norm_series(i) = A_norms(i)
                endif

                A_norms(i) = A_norms(i)**(1_dp/real(i,dp))

                if (i >= 3) then
                    if((abs((A_norms(i - 1) - A_norms(i))/A_norms(i)) < 0.5)) then
                        p = i
                        exit
                    endif
                endif

            enddo

        else

            do i = 2, p + 1

                A_norms(i) = one_norm_series(i)**(1_dp/real(i,dp))

            enddo

        endif

        do i = 2, p

            if (A_norms(i) > A_norms(i+1)) then

                alpha_array(i) = t*A_norms(i)

            else

                alpha_array(i) = t*A_norms(i+1)

            endif

        enddo

        num_ms_and_ps = 0
        do i = 2, p
            do j = i*(i- 1) - 1, m_max
                num_ms_and_ps = num_ms_and_ps + 1
            enddo
        enddo

        allocate(ms_and_ps(num_ms_and_ps,2))

        indx = 1
        do i = 2, p

            do j = (i*(i-1)-1), m_max

                ms_and_ps(indx, 1) = int(j,dp)
                ms_and_ps(indx,2) = int(i,dp)
                indx = indx + 1

            enddo

        enddo

        allocate(c_array(size(ms_and_ps, 1)))

        do i = 1, size(ms_and_ps, 1)

            c_array(i) = ms_and_ps(i, 1)*ceiling(alpha_array(ms_and_ps(i,2)) &
                        /theta(ms_and_ps(i,1)), kind=qp)

        enddo

        min_c = minval(c_array)
        m_star = ms_and_ps(minloc(c_array,1),1)
        s = max(min_c/m_star, 1)

    end subroutine C_m

    subroutine Parameters(  A, &
                            t, &
                            target_precision, &
                            partition_table, &
                            m_star, &
                            s, &
                            mpi_communicator, &
                            one_norm_series, &
                            p)

        type(CSR), intent(inout) :: A
        real(dp), intent(in) :: t
        character(len=2), intent(in) :: target_precision
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(out) :: m_star, s
        integer, intent(in) :: mpi_communicator
        real(dp), optional, dimension(p_max + 1), intent(inout) :: one_norm_series
        integer, optional, intent(inout) :: p

        real(dp), dimension(:), pointer :: theta

        real(dp) :: A_norm

        integer(dp) :: m_temp_1, m_temp_2

        integer :: m

        ! MPI ENVIRONMENT
        integer :: rank
        integer :: ierr

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        if (target_precision == "sp") then
            theta => theta_sp
        elseif (target_precision == "dp") then
            theta => theta_dp
        endif

        if (present(p) .and. present(one_norm_series)) then

            if (p /= 0) then
                A_norm = one_norm_series(1)
            endif

        else

            call One_Norm(  A, &
                            A_norm, &
                            partition_table, &
                            MPI_communicator)

            if (present(one_norm_series)) then
                one_norm_series(1) = A_norm
            endif

        endif

        if (t*A_norm <= &
            (2*1*(theta(m_max)/real(m_max,dp))*(p_max + 3)*p_max)) then

            m_temp_1 = ceiling(real(t,qp)*real(A_norm,qp)/theta(1), kind = qp)

            do m = 2, m_max

                m_temp_2 = m*ceiling(t*A_norm/theta(m), kind = qp)


               if (m_temp_2 < m_temp_1) then

                   m_star = m

               endif

               m_temp_1 = m_temp_2

            enddo

            s = ceiling(t*A_norm/theta(m_star), kind = qp)

        else

            if (present(one_norm_series)) then

                call C_m(   A, &
                            t, &
                            target_precision, &
                            partition_table, &
                            m_star, &
                            s, &
                            mpi_communicator, &
                            one_norm_series = one_norm_series, &
                            p_in = p)

            else


                call C_m(   A, &
                            t, &
                            target_precision, &
                            partition_table, &
                            m_star, &
                            s, &
                            mpi_communicator)

            endif

        endif

    end subroutine  Parameters

    subroutine Expm_Multiply(   A, &
                                B, &
                                t, &
                                partition_table, &
                                C, &
                                mpi_communicator, &
                                one_norm_series, &
                                p, &
                                target_precision)

        type(CSR), intent(inout) :: A
        complex(dp), intent(in), dimension(:) :: B
        real(dp), intent(in) :: t
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(in) :: mpi_communicator
        complex(dp), dimension(:), intent(inout) :: C
        real(dp), dimension(p_max + 1), optional, intent(inout) :: one_norm_series
        integer, optional, intent(inout) :: p
        character(len=2), optional, intent(in) :: target_precision

        complex(dp), dimension(:), allocatable :: B_temp_1, B_temp_2

        type(CSR) :: A_temp

        integer :: m_star, s

        character(len=2) :: set_target_precision
        real(qp) :: tol
        real(qp) :: c_1, c_2

        integer :: i, j, k, lb, ub

        integer :: rank
        integer :: ierr

        real(dp) :: start, finish, norm, norm_start, norm_finish

        call mpi_comm_rank(mpi_communicator, rank, ierr)

        if (present(target_precision)) then
            if (target_precision == "sp") then
                set_target_precision = target_precision
                tol = tol_sp
            elseif (target_precision == "dp") then
                set_target_precision = target_precision
                tol = tol_dp
            endif
        else
            set_target_precision = "dp"
            tol = tol_dp
        endif

        lb = lbound(C,1)
        ub = ubound(C,1)

        allocate(B_temp_1(lb:ub))
        allocate(B_temp_2(lb:ub))

        if (abs(t) < epsilon(t)) then
            m_star = 0
            s = 1
        else

            if (present(one_norm_series) .and. present(p)) then

                call Parameters(A, &
                                t, &
                                set_target_precision, &
                                partition_table, &
                                m_star, &
                                s, &
                                mpi_communicator, &
                                one_norm_series = one_norm_series, &
                                p = p)

            else

            call Parameters(A, &
                            t, &
                            set_target_precision, &
                            partition_table, &
                            m_star, &
                            s, &
                            mpi_communicator)

            endif

        endif

        A_temp%rows = A%rows
        A_temp%columns = A%columns
        A_temp%row_starts => A%row_starts
        A_temp%local_col_inds => A%local_col_inds
        A_temp%RHS_send_inds => A%RHS_send_inds
        A_temp%num_send_inds => A%num_send_inds
        A_temp%send_disps => A%send_disps
        A_temp%num_rec_inds => A%num_rec_inds
        A_temp%rec_disps => A%rec_disps

        allocate(A_temp%values(A%row_starts(partition_table(rank + 1)): &
            A%row_starts(partition_table(rank + 2)) - 1))

        do i = A%row_starts(lbound(A%row_starts, 1)), &
                A%row_starts(ubound(A%row_starts, 1)) - 1
            A_temp%values(i) = t*A%values(i)
        enddo

        B_temp_1(lb:ub) = B(lb:ub)
        C(lb:ub) = B(lb:ub)

        start = MPI_wtime()
        do i = 1, s

            norm_start = MPI_wtime()
            c_1 = Infinity_Norm(B_temp_1, MPI_communicator)
            norm_finish = MPI_wtime()

            norm = norm + norm_finish - norm_start

            do j = 1, m_star

                call SpMV_Series(   A_temp, &
                                    B_temp_1, &
                                    partition_table, &
                                    1, &
                                    j, &
                                    m_star, &
                                    rank, &
                                    B_temp_2, &
                                    mpi_communicator)

                B_temp_2(lb:ub) = B_temp_2(lb:ub)/real(s*j, dp)

                norm_start = MPI_wtime()
                c_2 = Infinity_Norm(B_temp_2, MPI_communicator)
                norm_finish = MPI_wtime()
                norm = norm + norm_finish - norm_start


                C(lb:ub) = C(lb:ub) + B_temp_2(lb:ub)

                norm_start = MPI_wtime()
                if ((c_1 + c_2) <= &
                    (tol*Infinity_Norm(C, MPI_communicator))) then
                    norm_finish = MPI_wtime()
                    norm = norm + norm_finish - norm_start
                    exit
                endif

                B_temp_1(lb:ub) = B_temp_2(lb:ub)

                c_1 = c_2

            enddo

            B_temp_1(lb:ub) = C(lb:ub)

        enddo
        finish = MPI_wtime()

        ! Final call to deallocate saved arrays.
        call SpMV_Series(   A_temp, &
                            B_temp_1, &
                            partition_table, &
                            0, &
                            0, &
                            0, &
                            rank, &
                            B_temp_2, &
                            mpi_communicator)

        deallocate(A_temp%values)

    end subroutine Expm_Multiply

    !subroutine Expm_Multiply_Series(    A, &
    !                                    B, &
    !                                    t0, &
    !                                    tq, &
    !                                    steps, &
    !                                    partition_table, &
    !                                    X, &
    !                                    mpi_communicator, &
    !                                    one_norm_series_in, &
    !                                    p_ex, &
    !                                    target_precision)

    !    real(dp), intent(in) :: t0, tq
    !    type(CSR), intent(inout) :: A
    !    complex(dp), intent(in), dimension(:) :: B
    !    integer, intent(in) :: steps
    !    integer, dimension(:), intent(in) :: partition_table
    !    complex(dp), dimension(:,:), intent(out) :: X
    !    integer, intent(in) :: mpi_communicator
    !    real(dp), dimension(p_max + 1), optional, intent(inout) :: one_norm_series_in
    !    integer, optional, intent(inout) :: p_ex
    !    character(len=2), optional, intent(in) :: target_precision

    !    real(dp), dimension(p_max + 1) :: one_norm_series
    !    integer :: p_in

    !    integer :: q

    !    complex(dp), dimension(:), allocatable :: Z, F
    !    complex(dp), dimension(:,:), allocatable :: K
    !    real(dp) :: h

    !    character(len=2) :: set_target_precision
    !    real(qp) :: tol
    !    real(qp) :: c_1, c_2

    !    integer :: m_star, s, m_hat
    !    integer :: d, j, r, d_tilde, p

    !    ! MPI ENVIRONMENT
    !    integer :: rank
    !    integer :: ierr

    !    integer :: i, kay, indx, lb, ub

    !    call mpi_comm_rank(MPI_communicator, rank, ierr)

    !    if (present(target_precision)) then
    !        if (target_precision == "sp") then
    !            set_target_precision = target_precision
    !            tol = tol_sp
    !        elseif (target_precision == "dp") then
    !            set_target_precision = target_precision
    !            tol = tol_dp
    !        endif
    !    else
    !        set_target_precision = "dp"
    !        tol = tol_dp
    !    endif

    !    lb = partition_table(rank + 1)
    !    ub = partition_table(rank + 2) - 1

    !    allocate(Z(ub - lb + 1))
    !    allocate(F(ub - lb + 1))

    !    if (present(one_norm_series_in) .and. present(p_ex)) then
    !        one_norm_series = one_norm_series_in
    !        p_in = p_ex
    !    else
    !        p_in =0
    !    endif

    !    call Expm_Multiply( A, &
    !                        B, &
    !                        t0, &
    !                        partition_table, &
    !                        X(:,1), &
    !                        mpi_communicator, &
    !                        one_norm_series = one_norm_series, &
    !                        p = p_in, &
    !                        target_precision = set_target_precision)

    !    if (steps == 0) return

    !    q = steps
    !    h = (tq - t0)/real(q, dp)

    !    call Parameters(A, &
    !                    tq - t0, &
    !                    set_target_precision, &
    !                    partition_table, &
    !                    m_star, &
    !                    s, &
    !                    mpi_communicator, &
    !                    one_norm_series = one_norm_series, &
    !                    p = p_in)

    !    if (q <= s) then

    !        do kay = 1, q

    !            call Expm_Multiply( A, &
    !                                X(:,kay), &
    !                                h, &
    !                                partition_table, &
    !                                X(:,kay + 1), &
    !                                MPI_communicator, &
    !                                one_norm_series = one_norm_series, &
    !                                p = p_in, &
    !                                target_precision = set_target_precision)

    !        enddo

    !        return

    !    endif

    !    d = floor(real(q)/real(s))
    !    j = floor(real(q)/real(d))
    !    r = q - d*j
    !    d_tilde = d

    !    call C_m(   A, &
    !                real(d, dp), &
    !                set_target_precision, &
    !                partition_table, &
    !                m_star, &
    !                s, &
    !                mpi_communicator, &
    !                one_norm_series = one_norm_series, &
    !                p_in = p_in)

    !    Z = X(:, 1)

    !    allocate(K(ub - lb + 1, m_star + 1))

    !    K = 0

    !    do i = 1, j + 1

    !        if (i > j) then
    !           d_tilde = r
    !        endif

    !        K(:, 1) = Z
    !        k(:,2:m_star+1) = 0

    !        m_hat = 0

    !        do kay = 1, d_tilde

    !            F = Z

    !            c_1 = Infinity_Norm(Z, MPI_communicator)

    !            do p = 1, m_star

    !                if (p > m_hat) then

    !                    call SpMV_series(   A, &
    !                                        K(:, p), &
    !                                        partition_table, &
    !                                        1, &
    !                                        p, &
    !                                        m_star, &
    !                                        rank, &
    !                                        K(:, p + 1), &
    !                                        MPI_communicator)

    !                    K(:, p + 1) = h*K(:, p+1)/real(p,dp)

    !                endif

    !                F = F + (real(kay,dp)**real(p, dp))*K(:,p+1)

    !                c_2 = (real(kay,qp)**real(p, dp)) &
    !                        *Infinity_Norm(K(:,p + 1), MPI_communicator)

    !                indx = p

    !                if ((c_1 + c_2) <= &
    !                    (tol*Infinity_Norm(F, MPI_communicator))) then

    !                    exit

    !                endif

    !                c_1 = c_2

    !            enddo

    !            m_hat = max(m_hat, indx)

    !            X(:, kay + (i - 1)*d + 1) =  F

    !        enddo

    !        if (i <= j) then
    !            Z = X(:, i*d + 1)
    !        endif

    !    enddo

    !end subroutine Expm_Multiply_Series

end module Expm
