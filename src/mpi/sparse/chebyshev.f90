!   QuOp_MPI - Chebyshev Time Evolution Module
!
!   Copyright (C) 2019-2026 Edric Matwiejew
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

!------------------------------------------------------------------------------
!> @brief Chebyshev polynomial time evolution for sparse Hamiltonians.
!>
!> @details Implements matrix exponential e^{A*t} |B> using Chebyshev polynomial
!> expansion with Bessel function coefficients. This method is particularly
!> efficient for unitary time evolution e^{-i*H*t} where the matrix A = -i*H
!> is anti-Hermitian.
!>
!> The key identity used is:
!>   e^{-i z cos(theta)} = J_0(z) + 2 * sum_{k=1}^{inf} (-i)^k J_k(z) T_k(cos(theta))
!>
!> Where:
!>   - J_k are Bessel functions of the first kind
!>   - T_k are Chebyshev polynomials: T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
!>
!> The matrix A is rescaled by its spectral radius M so eigenvalues lie in
!> the unit circle, then the expansion converges for any t.
!>
!> @note This implementation assumes A = -i*H where H is Hermitian with real
!> eigenvalues. The spectral radius is estimated using the Gershgorin bound
!> on H (not A).
!------------------------------------------------------------------------------

module Chebyshev

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, int64
    use :: Sparse
    use :: MPI

    implicit none

    private

    public :: Chebyshev_Multiply
    public :: Estimate_Spectral_Radius

contains

    !--------------------------------------------------------------------------
    !> @brief Compute (-i)^k as a complex number.
    !>
    !> @param[in] k  The exponent
    !> @return       (-i)^k as complex(dp)
    !--------------------------------------------------------------------------
    pure function minus_i_power(k) result(c)
        integer, intent(in) :: k
        complex(dp) :: c
        integer :: k_mod

        k_mod = mod(k, 4)
        if (k_mod < 0) k_mod = k_mod + 4

        select case (k_mod)
        case (0)
            c = cmplx(1.0_dp, 0.0_dp, dp)     ! (-i)^0 = 1
        case (1)
            c = cmplx(0.0_dp, -1.0_dp, dp)    ! (-i)^1 = -i
        case (2)
            c = cmplx(-1.0_dp, 0.0_dp, dp)    ! (-i)^2 = -1
        case (3)
            c = cmplx(0.0_dp, 1.0_dp, dp)     ! (-i)^3 = i
        end select
    end function minus_i_power

    !--------------------------------------------------------------------------
    !> @brief Estimate spectral radius using Gershgorin Circle Theorem.
    !>
    !> @details For each row i, the Gershgorin disc is centered at |A_ii| with
    !> radius R_i = sum_{j != i} |A_ij|. All eigenvalues lie within the union
    !> of discs. For Hermitian matrices, the spectral radius is bounded by:
    !>   rho(H) <= max_i (|H_ii| + R_i)
    !>
    !> This routine computes the local maximum and reduces across all ranks.
    !>
    !> @param[in]    A                 CSR matrix (assumes A = -i*H)
    !> @param[in]    partition_table   Row distribution across ranks
    !> @param[in]    MPI_communicator  MPI communicator
    !> @param[out]   spectral_radius   Upper bound on spectral radius of H
    !--------------------------------------------------------------------------
    subroutine Estimate_Spectral_Radius(A, partition_table, MPI_communicator, &
                                        spectral_radius)
        type(CSR), intent(in) :: A
        integer, dimension(:), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator
        real(dp), intent(out) :: spectral_radius

        integer :: rank, ierr
        integer :: lb, ub, i
        integer(dp) :: j, start_j, end_j, global_row
        real(dp) :: diag_element, row_sum, local_max, local_bound

        call MPI_Comm_rank(MPI_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1

        local_max = 0.0_dp

        do i = lb, ub
            start_j = A%row_starts(i)
            end_j = A%row_starts(i + 1) - 1
            global_row = i  ! 1-based global row index

            diag_element = 0.0_dp
            row_sum = 0.0_dp

            if (A%has_values) then
                do j = start_j, end_j
                    if (A%col_indexes(j) == global_row) then
                        ! Diagonal element - take magnitude
                        ! For A = -i*H, the diagonal of H is i*A_ii
                        diag_element = abs(A%values(j))
                    else
                        ! Off-diagonal element
                        row_sum = row_sum + abs(A%values(j))
                    end if
                end do
            else
                ! Implicit ones: all nonzeros have value 1
                do j = start_j, end_j
                    if (A%col_indexes(j) == global_row) then
                        diag_element = 1.0_dp
                    else
                        row_sum = row_sum + 1.0_dp
                    end if
                end do
            end if

            local_bound = diag_element + row_sum
            local_max = max(local_max, local_bound)
        end do

        ! Reduce to find global maximum
        call MPI_Allreduce(local_max, spectral_radius, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MPI_communicator, ierr)

    end subroutine Estimate_Spectral_Radius

    !--------------------------------------------------------------------------
    !> @brief Chebyshev time evolution: C = exp(A*t) * B
    !>
    !> @details Uses Chebyshev polynomial expansion to compute the action of
    !> the matrix exponential on a vector. The expansion is:
    !>   exp(A*t) = exp(-i*M*t * (i*A/M))
    !>            = sum_k c_k T_k(i*A/M)
    !>
    !> where A = -i*H (anti-Hermitian), M is the spectral radius of H, and
    !> c_k are Bessel-weighted coefficients.
    !>
    !> The Chebyshev polynomials are computed via the recurrence:
    !>   T_0(X)|psi> = |psi>
    !>   T_1(X)|psi> = X|psi>
    !>   T_{k+1}(X)|psi> = 2*X*T_k(X)|psi> - T_{k-1}(X)|psi>
    !>
    !> @param[inout] A                 CSR matrix (assumes A = -i*H)
    !> @param[in]    B                 Input vector (local portion)
    !> @param[in]    t                 Evolution time
    !> @param[in]    partition_table   Row distribution across ranks
    !> @param[out]   C                 Output vector (local portion)
    !> @param[in]    MPI_communicator  MPI communicator
    !> @param[in]    spectral_radius   Optional: precomputed spectral radius
    !> @param[in]    epsilon           Optional: convergence tolerance (default 1e-14)
    !--------------------------------------------------------------------------
    subroutine Chebyshev_Multiply(A, B, t, partition_table, C, MPI_communicator, &
                                  spectral_radius, epsilon)
        type(CSR), intent(inout) :: A
        complex(dp), dimension(:), intent(in) :: B
        real(dp), intent(in) :: t
        integer, dimension(:), intent(in) :: partition_table
        complex(dp), dimension(:), intent(out) :: C
        integer, intent(in) :: MPI_communicator
        real(dp), intent(in), optional :: spectral_radius
        real(dp), intent(in), optional :: epsilon

        integer :: rank, ierr
        integer :: lb, ub, n_local
        integer :: k, m_order
        real(dp) :: M, eps, z, Jk
        complex(dp) :: ck, scalar_i_inv_M
        complex(dp), allocatable, target :: work1(:), work2(:), work3(:)
        complex(dp), pointer :: w_km1(:), w_k(:), w_kp1(:), w_swap(:)
        complex(dp), allocatable :: coeffs(:)

        call MPI_Comm_rank(MPI_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        ! Get spectral radius
        if (present(spectral_radius)) then
            M = spectral_radius
        else
            call Estimate_Spectral_Radius(A, partition_table, MPI_communicator, M)
        end if

        ! Get convergence tolerance
        if (present(epsilon)) then
            eps = epsilon
        else
            eps = 1.0e-14_dp
        end if

        ! The matrix stores A = -i*H
        ! We want exp(A*t) = exp(-i*H*t)
        ! The Chebyshev expansion for exp(-i*H*t) with H rescaled to [-1,1]:
        !   exp(-i*M*t * (H/M)) = sum_k c_k T_k(H/M)
        ! where z = M*t and c_k = 2*(-i)^k*J_k(z) for k>=1, c_0 = J_0(z)
        !
        ! But we have A = -i*H, so H = i*A
        ! Thus (H/M) = (i/M)*A
        ! So we apply the SpMV with scalar = i/M
        
        z = t * M
        scalar_i_inv_M = cmplx(0.0_dp, 1.0_dp / M, dp)

        ! Determine expansion order based on when Bessel coefficients become negligible
        m_order = 0
        do k = 1, 10000
            Jk = bessel_jn(k, z)
            if (2.0_dp * abs(Jk) < eps) then
                m_order = k - 1
                exit
            end if
        end do

        ! Ensure at least order 1
        if (m_order < 1) m_order = 1

        ! Precompute Bessel coefficients: c_k = 2*(-i)^k*J_k(z) for k>=1, c_0 = J_0(z)
        allocate(coeffs(0:m_order))
        coeffs(0) = cmplx(bessel_jn(0, z), 0.0_dp, dp)
        do k = 1, m_order
            coeffs(k) = 2.0_dp * minus_i_power(k) * bessel_jn(k, z)
        end do

        ! Allocate work arrays
        allocate(work1(n_local), work2(n_local), work3(n_local))
        w_km1 => work1
        w_k   => work2
        w_kp1 => work3

        ! T_0(H/M)|psi> = |psi>
        w_km1 = B

        ! T_1(H/M)|psi> = (H/M)|psi> = (i/M)*A|psi>
        call SpMV_Graph(A, w_km1, partition_table, rank, w_k, scalar_i_inv_M, &
                        MPI_communicator)

        ! Initialize: C = c_0*T_0 + c_1*T_1
        if (m_order >= 1) then
            C = coeffs(0) * w_km1 + coeffs(1) * w_k
        else
            C = coeffs(0) * w_km1
        end if

        ! Chebyshev recurrence: T_{k+1}(X) = 2*X*T_k(X) - T_{k-1}(X)
        ! where X = H/M = (i/M)*A
        do k = 2, m_order
            ! w_kp1 = (i/M)*A*w_k
            call SpMV_Graph(A, w_k, partition_table, rank, w_kp1, scalar_i_inv_M, &
                            MPI_communicator)
            
            ! Apply recurrence and accumulate
            ck = coeffs(k)
            w_kp1 = 2.0_dp * w_kp1 - w_km1
            C = C + ck * w_kp1

            ! Rotate pointers
            w_swap => w_km1
            w_km1  => w_k
            w_k    => w_kp1
            w_kp1  => w_swap
        end do

        nullify(w_km1, w_k, w_kp1, w_swap)
        deallocate(work1, work2, work3, coeffs)

    end subroutine Chebyshev_Multiply

end module Chebyshev
