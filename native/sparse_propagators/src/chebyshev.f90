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

module chebyshev

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64
    use sparse, only: c_null_ptr, c_ptr, csr, spmv_graph
    use :: MPI
#ifdef USE_HIP
    use, intrinsic :: iso_c_binding, only: c_loc, c_size_t, c_int, c_long, &
                                                                              c_double, c_double_complex
    use hipfort
    use hipfort_check
    use hipfort_types, only: dim3
    use hip_sparse_expm_kernels, only: launch_gershgorin_bound_kernel, &
                                       launch_gershgorin_bound_unit_kernel, &
                                       launch_reduce_max_kernel, &
                                       launch_vec_copy_kernel, &
                                       launch_complex_scale_kernel, &
                                       launch_complex_axpy_kernel, &
                                       launch_chebyshev_local_weighted_kernel, &
                                       launch_chebyshev_local_unit_kernel, &
                                       launch_chebyshev_remote_weighted_kernel, &
                                       launch_chebyshev_remote_unit_kernel, &
                                       launch_chebyshev_accumulate_kernel, &
                                       launch_pack_send_buf_kernel, &
                                       launch_spmv_local_weighted_kernel, &
                                       launch_spmv_local_unit_kernel, &
                                       launch_spmv_remote_weighted_kernel, &
                                       launch_spmv_remote_unit_kernel
#endif

    implicit none

    private

    public :: chebyshev_multiply
    public :: estimate_spectral_radius

contains

    !--------------------------------------------------------------------------
    !> @brief Compute (-i)^k as a complex number.
    !>
    !> @param[in] k  The exponent
    !> @return       (-i)^k as complex(real64)
    !--------------------------------------------------------------------------
    pure function minus_i_power(k) result(c)
        integer, intent(in) :: k
        complex(real64) :: c
        integer(int32) :: k_mod

        k_mod = mod(k, 4)
        if (k_mod < 0) k_mod = k_mod + 4

        select case (k_mod)
        case (0)
            c = cmplx(1.0_real64, 0.0_real64, real64) ! (-i)^0 = 1
        case (1)
            c = cmplx(0.0_real64, -1.0_real64, real64) ! (-i)^1 = -i
        case (2)
            c = cmplx(-1.0_real64, 0.0_real64, real64) ! (-i)^2 = -1
        case (3)
            c = cmplx(0.0_real64, 1.0_real64, real64) ! (-i)^3 = i
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
    !> If A%device_ready is true, uses GPU kernels for the computation.
    !>
    !> @param[in]    A                 CSR matrix (assumes A = -i*H)
    !> @param[in]    partition_table   Row distribution across ranks
    !> @param[in]    MPI_communicator  MPI communicator
    !> @param[out]   spectral_radius   Upper bound on spectral radius of H
    !--------------------------------------------------------------------------
    subroutine estimate_spectral_radius(A, partition_table, MPI_communicator, &
                                        spectral_radius)
        type(CSR), intent(in) :: A
        integer(int64), dimension(:), intent(in) :: partition_table
        integer, intent(in) :: MPI_communicator
        real(real64), intent(out) :: spectral_radius

        integer(int32) :: rank, ierr
        integer(int64) :: lb, ub, i
        integer(int64) :: j, start_j, end_j, global_row
        real(real64) :: diag_element, row_sum, local_max, local_bound
        integer(int64) :: local_rows
#ifdef USE_HIP
        type(c_ptr) :: row_bounds_dev, temp_max_dev
        type(dim3) :: grid, block
        integer, parameter :: BLOCKSIZE = 256
        real(real64), target :: local_max_array(1)
        integer(int32) :: num_blocks
#endif

        call MPI_Comm_rank(MPI_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        local_rows = ub - lb + 1

#ifdef USE_HIP
        if (A%device_ready .and. local_rows > 0) then
            ! GPU path: use Gershgorin kernel

            ! Allocate temp arrays on device
            call hipCheck(hipMalloc(row_bounds_dev, int(local_rows * 8, c_size_t)))

            ! Launch Gershgorin kernel
            block = dim3(BLOCKSIZE, 1, 1)
            grid = dim3(int((local_rows + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

            if (A%has_values) then
                call launch_gershgorin_bound_kernel(grid, block, 0, c_null_ptr, &
                                                    A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                    row_bounds_dev, int(local_rows, c_int), int(0, c_int))
            else
                call launch_gershgorin_bound_unit_kernel(grid, block, 0, c_null_ptr, &
                                                         A%row_starts_dev, A%col_indexes_dev, &
                                                         row_bounds_dev, int(local_rows, c_int), int(0, c_int))
            end if

            ! Reduce to find local max
            num_blocks = (local_rows + BLOCKSIZE - 1) / BLOCKSIZE
            call hipCheck(hipMalloc(temp_max_dev, int(num_blocks * 8, c_size_t)))

            call launch_reduce_max_kernel(grid, block, 0, c_null_ptr, &
                                          row_bounds_dev, temp_max_dev, int(local_rows, c_size_t))

            ! If multiple blocks, reduce again
            if (num_blocks > 1) then
                block = dim3(BLOCKSIZE, 1, 1)
                grid = dim3(1, 1, 1)
                call launch_reduce_max_kernel(grid, block, 0, c_null_ptr, &
                                              temp_max_dev, temp_max_dev, int(num_blocks, c_size_t))
            end if

            call hipCheck(hipDeviceSynchronize())

            ! Copy result to host
            call hipCheck(hipMemcpy(c_loc(local_max_array), temp_max_dev, &
                                    int(8, c_size_t), hipMemcpyDeviceToHost))
            local_max = local_max_array(1)

            ! Free temp arrays
            call hipCheck(hipFree(row_bounds_dev))
            call hipCheck(hipFree(temp_max_dev))
        else
#endif
            ! CPU path
            local_max = 0.0_real64

            do i = 1, local_rows
                start_j = A%row_starts(i) + 1 ! Convert 0-based offset to 1-based index
                end_j = A%row_starts(i + 1) ! 0-based offset, use as-is for end
                global_row = lb + i - 2 ! 0-based global row for comparison with col_indexes

                diag_element = 0.0_real64
                row_sum = 0.0_real64

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
                            diag_element = 1.0_real64
                        else
                            row_sum = row_sum + 1.0_real64
                        end if
                    end do
                end if

                local_bound = diag_element + row_sum
                local_max = max(local_max, local_bound)
            end do
#ifdef USE_HIP
        end if
#endif

        ! Reduce to find global maximum
        call MPI_Allreduce(local_max, spectral_radius, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MPI_communicator, ierr)

    end subroutine estimate_spectral_radius

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
    !> GPU acceleration: If A%device_ready is true, B and C are assumed to be
    !> device-allocated arrays. Work arrays are allocated internally on device.
    !>
    !> @param[inout] A                 CSR matrix (assumes A = -i*H)
    !> @param[in]    B                 Input vector (host or device, size n_local)
    !> @param[in]    t                 Evolution time
    !> @param[in]    partition_table   Row distribution across ranks
    !> @param[out]   C                 Output vector (host or device, size n_local)
    !> @param[in]    MPI_communicator  MPI communicator
    !> @param[in]    spectral_radius   Optional: precomputed spectral radius
    !> @param[in]    epsilon           Optional: convergence tolerance (default 1e-14)
    !--------------------------------------------------------------------------
    subroutine chebyshev_multiply(A, B, t, partition_table, C, MPI_communicator, &
                                  spectral_radius, epsilon)
        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in), target :: B
        real(real64), intent(in) :: t
        integer(int64), dimension(:), intent(in) :: partition_table
        complex(real64), dimension(:), intent(out), target :: C
        integer, intent(in) :: MPI_communicator
        real(real64), intent(in), optional :: spectral_radius
        real(real64), intent(in), optional :: epsilon

        integer(int32) :: rank, ierr
        integer(int64) :: lb, ub, n_local
        integer(int32) :: k, m_order
        real(real64) :: M, eps, z, Jk
        real(real64) :: scalar_inv_M
        complex(real64), allocatable, target :: work1(:), work2(:), work3(:)
        complex(real64), pointer :: w_km1(:), w_k(:), w_kp1(:), w_swap(:)
        complex(real64), allocatable :: coeffs(:)

        call MPI_Comm_rank(MPI_communicator, rank, ierr)

        lb = partition_table(rank + 1)
        ub = partition_table(rank + 2) - 1
        n_local = ub - lb + 1

        ! Get spectral radius
        if (present(spectral_radius)) then
            M = spectral_radius
        else
            call estimate_spectral_radius(A, partition_table, MPI_communicator, M)
        end if

        ! Get convergence tolerance
        if (present(epsilon)) then
            eps = epsilon
        else
            eps = 1.0e-14_real64
        end if

#ifdef USE_HIP
        if (A%device_ready) then
            call chebyshev_multiply_gpu_impl(A, B, t, partition_table, C, &
                                             MPI_communicator, M, eps, n_local)
            return
        end if
#endif

        ! CPU path follows
        ! Chebyshev expansion for exp(-i*H*t) where H is Hermitian with
        ! eigenvalues in [-M, M]:
        !
        !   exp(-i*H*t) = exp(-i*(H/M)*M*t)
        !
        ! Let X = H/M (eigenvalues in [-1, 1]) and z = M*t. Then:
        !   exp(-i*X*z) = J_0(z) + 2*sum_{k=1}^inf (-i)^k * J_k(z) * T_k(X)
        !
        ! The Chebyshev polynomials T_k(X) are computed via recurrence:
        !   T_0(X) = I
        !   T_1(X) = X
        !   T_{k+1}(X) = 2*X*T_k(X) - T_{k-1}(X)
        !
        ! Note: The input matrix A is the Hermitian generator (e.g., adjacency
        ! matrix). We scale by 1/M for the Chebyshev recurrence.

        z = t * M
        scalar_inv_M = 1.0_real64 / M

        ! Determine expansion order based on when Bessel coefficients become negligible
        m_order = 0
        do k = 1, 10000
            Jk = bessel_jn(k, z)
            if (2.0_real64 * abs(Jk) < eps) then
                m_order = k - 1
                exit
            end if
        end do

        ! Ensure at least order 1
        if (m_order < 1) m_order = 1

        ! Precompute Bessel coefficients: c_k = 2*(-i)^k*J_k(z) for k>=1, c_0 = J_0(z)
        allocate (coeffs(0:m_order))
        coeffs(0) = cmplx(bessel_jn(0, z), 0.0_real64, real64)
        do k = 1, m_order
            coeffs(k) = 2.0_real64 * minus_i_power(k) * bessel_jn(k, z)
        end do

        ! Allocate work arrays
        allocate (work1(n_local), work2(n_local), work3(n_local))
        w_km1 => work1
        w_k => work2
        w_kp1 => work3

        ! T_0(X)|psi> = |psi>
        w_km1 = B

        ! T_1(X)|psi> = X|psi> = (1/M)*H|psi>
        call spmv_graph(A, w_km1, partition_table, rank, w_k, &
                        cmplx(scalar_inv_M, 0.0_real64, real64), MPI_communicator)

        ! Initialize: C = c_0*T_0 + c_1*T_1
        if (m_order >= 1) then
            C = coeffs(0) * w_km1 + coeffs(1) * w_k
        else
            C = coeffs(0) * w_km1
        end if

        ! Chebyshev recurrence: T_{k+1}(X) = 2*X*T_k(X) - T_{k-1}(X)
        ! where X = H/M
        do k = 2, m_order
            ! w_kp1 = (1/M)*H*w_k = X*w_k
            call spmv_graph(A, w_k, partition_table, rank, w_kp1, &
                            cmplx(scalar_inv_M, 0.0_real64, real64), MPI_communicator)

            ! Apply recurrence: T_{k+1} = 2*X*T_k - T_{k-1}
            w_kp1 = 2.0_real64 * w_kp1 - w_km1

            ! Accumulate contribution
            C = C + coeffs(k) * w_kp1

            ! Rotate pointers
            w_swap => w_km1
            w_km1 => w_k
            w_k => w_kp1
            w_kp1 => w_swap
        end do

        nullify (w_km1, w_k, w_kp1, w_swap)
        deallocate (work1, work2, work3, coeffs)

    end subroutine chebyshev_multiply

#ifdef USE_HIP
    !--------------------------------------------------------------------------
    !> @brief Internal GPU implementation for chebyshev_multiply
    !>
    !> @details GPU-accelerated Chebyshev expansion. Called from chebyshev_multiply
    !> when A%device_ready is true. B and C are device-allocated arrays.
    !> Work arrays are allocated on device internally.
    !--------------------------------------------------------------------------
    subroutine chebyshev_multiply_gpu_impl(A, B, t, partition_table, C, &
                                           MPI_communicator, M, eps, n_local)
        use, intrinsic :: iso_c_binding, only: c_f_pointer
        type(CSR), intent(inout) :: A
        complex(real64), dimension(:), intent(in), target :: B ! Device array
        real(real64), intent(in) :: t
        integer(int64), dimension(:), intent(in) :: partition_table
        complex(real64), dimension(:), intent(out), target :: C ! Device array
        integer, intent(in) :: MPI_communicator
        real(real64), intent(in) :: M ! Spectral radius (precomputed)
        real(real64), intent(in) :: eps ! Tolerance (precomputed)
        integer(int64), intent(in) :: n_local

        integer(int32) :: rank, ierr, request, status(MPI_STATUS_SIZE)
        integer(int64) :: n_local_64
        integer(int32) :: k, m_order
        real(real64) :: z, Jk, inv_M
        complex(real64), allocatable :: coeffs(:)
        type(c_ptr) :: B_dev, C_dev
        type(c_ptr) :: work1_dev, work2_dev
        type(c_ptr) :: w_km1_dev, w_k_dev, w_swap_dev
        complex(real64), pointer :: w_km1_ptr(:), w_k_ptr(:), w_swap_ptr(:)
#ifdef QUOP_GPU_AWARE_MPI
        ! Fortran pointers to device buffers for MPI (c_ptr cannot be passed
        ! directly to MPI -- Fortran would pass the address of the c_ptr variable
        ! on the host, not the device address stored within it).
        complex(real64), dimension(:), pointer :: send_buf_fptr, recv_buf_fptr
#endif

        integer, parameter :: BLOCKSIZE = 256
        type(dim3) :: grid, block

        call MPI_Comm_rank(MPI_communicator, rank, ierr)

        n_local_64 = int(n_local, int64)

        ! lb_graph and ub_graph are 0-based

        ! Get device pointers from the target arrays (already on device)
        B_dev = c_loc(B(1))
        C_dev = c_loc(C(1))

        ! Allocate work arrays on device (2 arrays needed for recurrence)
        call hipCheck(hipMalloc(work1_dev, int(n_local * 16, c_size_t)))
        call hipCheck(hipMalloc(work2_dev, int(n_local * 16, c_size_t)))

        ! Chebyshev expansion parameter
        z = t * M
        inv_M = 1.0_real64 / M

        ! Determine expansion order based on when Bessel coefficients become negligible
        m_order = 0
        do k = 1, 10000
            Jk = bessel_jn(k, z)
            if (2.0_real64 * abs(Jk) < eps) then
                m_order = k - 1
                exit
            end if
        end do
        if (m_order < 1) m_order = 1

        ! Precompute Bessel coefficients
        allocate (coeffs(0:m_order))
        coeffs(0) = cmplx(bessel_jn(0, z), 0.0_real64, real64)
        do k = 1, m_order
            coeffs(k) = 2.0_real64 * minus_i_power(k) * bessel_jn(k, z)
        end do

        ! Set up kernel launch configuration
        block = dim3(BLOCKSIZE, 1, 1)
        grid = dim3(int((n_local + BLOCKSIZE - 1) / BLOCKSIZE), 1, 1)

        ! Initialize pointer aliases - convert c_ptr to Fortran pointers
        w_km1_dev = work1_dev
        w_k_dev = work2_dev
        call c_f_pointer(w_km1_dev, w_km1_ptr, [n_local])
        call c_f_pointer(w_k_dev, w_k_ptr, [n_local])

        !----------------------------------------------------------------------
        ! T_0(X)|psi> = |psi>
        ! work1 = B
        !----------------------------------------------------------------------
        call launch_vec_copy_kernel(grid, block, 0, A%stream, &
                                    B_dev, w_km1_dev, int(n_local_64, c_size_t))

        !----------------------------------------------------------------------
        ! Initialize C = coeff(0) * T_0 = coeff(0) * B
        !----------------------------------------------------------------------
        call launch_vec_copy_kernel(grid, block, 0, A%stream, &
                                    B_dev, C_dev, int(n_local_64, c_size_t))
        call launch_complex_scale_kernel(grid, block, 0, A%stream, &
                                         coeffs(0), C_dev, int(n_local_64, c_size_t))

        !----------------------------------------------------------------------
        ! T_1(X)|psi> = X|psi> = (1/M)*A|psi>
        ! Use spmv_graph for GPU SpMV with scalar = 1/M
        !----------------------------------------------------------------------
        call spmv_graph(A, w_km1_ptr, partition_table, rank, w_k_ptr, &
                        cmplx(inv_M, 0.0_real64, real64), MPI_communicator)

        ! C += coeff(1) * T_1
        if (m_order >= 1) then
            call launch_chebyshev_accumulate_kernel(grid, block, 0, A%stream, &
                                                    coeffs(1), w_k_dev, C_dev, int(n_local_64, c_size_t))
        end if

        !----------------------------------------------------------------------
        ! Main Chebyshev recurrence loop: k = 2 to m_order
        ! T_{k+1} = 2*X*T_k - T_{k-1}
        !----------------------------------------------------------------------
        do k = 2, m_order
            ! Phase 1: Pack send buffer and start MPI
            if (A%total_send > 0) then
                call launch_pack_send_buf_kernel(grid, block, 0, A%stream, &
                                                 w_k_dev, A%send_offsets_dev, A%send_buf_dev, A%total_send)

#ifdef QUOP_GPU_AWARE_MPI
                ! GPU-aware MPI: ensure packed data is visible before MPI reads
                ! send_buf_dev via RDMA.
                call hipCheck(hipDeviceSynchronize())
#else
                ! Non-GPU-aware MPI: stage send buffer through host
                call hipCheck(hipMemcpyAsync(c_loc(A%send_buf(1)), A%send_buf_dev, &
                                             int(A%total_send * 16, c_size_t), hipMemcpyDeviceToHost, A%stream))
                call hipCheck(hipStreamSynchronize(A%stream))
#endif
            end if

            ! Start non-blocking MPI exchange
#ifdef QUOP_GPU_AWARE_MPI
            ! GPU-aware MPI: communicate directly with device buffers.
            ! Convert c_ptr to Fortran pointers so MPI receives the device address,
            ! not the host address of the c_ptr variable.
            call c_f_pointer(A%send_buf_dev, send_buf_fptr, [A%total_send])
            call c_f_pointer(A%recv_buf_dev, recv_buf_fptr, [A%total_recv])
            call MPI_Ineighbor_alltoallv(send_buf_fptr, A%graph_send_counts, A%graph_send_disps, &
                                         MPI_DOUBLE_COMPLEX, &
                                         recv_buf_fptr, A%graph_recv_counts, A%graph_recv_disps, &
                                         MPI_DOUBLE_COMPLEX, &
                                         A%graph_comm, request, ierr)
#else
            ! Non-GPU-aware MPI: use host staging buffers
            call MPI_Ineighbor_alltoallv(A%send_buf, A%graph_send_counts, A%graph_send_disps, &
                                         MPI_DOUBLE_COMPLEX, &
                                         A%recv_buf, A%graph_recv_counts, A%graph_recv_disps, &
                                         MPI_DOUBLE_COMPLEX, &
                                         A%graph_comm, request, ierr)
#endif

            ! Phase 1: Diagonal SpMV (overlaps with MPI)
            ! Computes Aw_k (diagonal-block contribution only)
            if (A%has_values) then
                call launch_chebyshev_local_weighted_kernel(grid, block, 0, A%stream, &
                                                            inv_M, A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                            A%diag_lo_dev, A%diag_hi_dev, &
                                                            w_k_dev, A%Aw_k_dev, n_local_64)
            else
                call launch_chebyshev_local_unit_kernel(grid, block, 0, A%stream, &
                                                        inv_M, A%row_starts_dev, A%col_indexes_dev, &
                                                        A%diag_lo_dev, A%diag_hi_dev, &
                                                        w_k_dev, A%Aw_k_dev, n_local_64)
            end if

            ! Wait for MPI to complete
            call MPI_Wait(request, status, ierr)

            ! H->D transfer of recv buffer (only needed for non-GPU-aware MPI)
            if (A%total_recv > 0) then
#ifdef QUOP_GPU_AWARE_MPI
                ! GPU-aware MPI: ensure RDMA writes are visible before kernels
                ! read recv_buf_dev.
                call hipCheck(hipDeviceSynchronize())
#else
                ! Non-GPU-aware MPI: transfer received data from host to device
                call hipCheck(hipMemcpyAsync(A%recv_buf_dev, c_loc(A%recv_buf(1)), &
                                             int(A%total_recv * 16, c_size_t), hipMemcpyHostToDevice, A%stream))
#endif
            end if

            ! Phase 2: Off-diagonal SpMV + Chebyshev recurrence
            ! w_km1 = 2*inv_M*(Aw_k + A_off * recv_buf) - w_km1  (overwrites T_{k-1} with T_{k+1})
            ! Reads recv_buf via col_halo - n_local. When total_recv == 0 the
            ! off-diagonal segments are empty by construction, so the kernel still
            ! correctly applies the Chebyshev recurrence.
            if (A%has_values) then
                call launch_chebyshev_remote_weighted_kernel(grid, block, 0, A%stream, &
                                                             inv_M, A%row_starts_dev, A%col_indexes_dev, A%values_dev, &
                                                             A%diag_lo_dev, A%diag_hi_dev, A%recv_buf_dev, &
                                                             A%Aw_k_dev, w_km1_dev, w_km1_dev, n_local_64, n_local_64)
            else
                call launch_chebyshev_remote_unit_kernel(grid, block, 0, A%stream, &
                                                         inv_M, A%row_starts_dev, A%col_indexes_dev, &
                                                         A%diag_lo_dev, A%diag_hi_dev, A%recv_buf_dev, &
                                                         A%Aw_k_dev, w_km1_dev, w_km1_dev, n_local_64, n_local_64)
            end if

            ! Accumulate: C += coeff(k) * T_{k+1}
            ! Note: T_{k+1} is now in w_km1_dev (it was overwritten)
            call launch_chebyshev_accumulate_kernel(grid, block, 0, A%stream, &
                                                    coeffs(k), w_km1_dev, C_dev, int(n_local_64, c_size_t))

            ! Swap pointers: w_km1 <-> w_k
            ! After swap: w_km1 holds T_k (old w_k), w_k holds T_{k+1} (was in w_km1)
            w_swap_dev = w_km1_dev
            w_km1_dev = w_k_dev
            w_k_dev = w_swap_dev
            w_swap_ptr => w_km1_ptr
            w_km1_ptr => w_k_ptr
            w_k_ptr => w_swap_ptr
        end do

        ! Synchronize stream before returning
        call hipCheck(hipStreamSynchronize(A%stream))

        ! Free work arrays
        call hipCheck(hipFree(work1_dev))
        call hipCheck(hipFree(work2_dev))
        deallocate (coeffs)

    end subroutine chebyshev_multiply_gpu_impl
#endif

end module chebyshev
