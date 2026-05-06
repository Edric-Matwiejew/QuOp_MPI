!------------------------------------------------------------------------------
!> @brief Fortran bind(c) interfaces for HIP transverse-field kernels.
!>
!> @details Provides Fortran-callable interfaces for the GPU kernels defined
!> in kernels/hip_transverse_field_kernels.cpp:
!>   - launch_tf_local_pair_qubit_kernel    (aligned fast path)
!>   - launch_tf_local_pair_kernel          (segmented boundary)
!>   - launch_tf_local_pair_strided_kernel  (segmented bulk)
!>   - launch_tf_remote_update_kernel
!------------------------------------------------------------------------------
module hip_transverse_field_kernels

    use, intrinsic :: iso_c_binding

    implicit none

    private

    public :: launch_tf_local_pair_qubit_kernel
    public :: launch_tf_local_pair_kernel
    public :: launch_tf_local_pair_strided_kernel
    public :: launch_tf_remote_update_kernel

    interface

        subroutine launch_tf_local_pair_qubit_kernel(psi, n_pairs, q, &
                                                     coeff_diag, coeff_offdiag, stream) &
            bind(c, name='launch_tf_local_pair_qubit_kernel')
            import :: c_ptr, c_int, c_long, c_double_complex
            implicit none
            type(c_ptr), value :: psi
            integer(c_long), value :: n_pairs
            integer(c_int), value :: q
            complex(c_double_complex), value :: coeff_diag
            complex(c_double_complex), value :: coeff_offdiag
            type(c_ptr), value :: stream
        end subroutine launch_tf_local_pair_qubit_kernel

        subroutine launch_tf_local_pair_kernel(psi, lb_global, g0, count, delta, &
                                               coeff_diag, coeff_offdiag, stream) &
            bind(c, name='launch_tf_local_pair_kernel')
            import :: c_ptr, c_long, c_double_complex
            implicit none
            type(c_ptr), value :: psi
            integer(c_long), value :: lb_global
            integer(c_long), value :: g0
            integer(c_long), value :: count
            integer(c_long), value :: delta
            complex(c_double_complex), value :: coeff_diag
            complex(c_double_complex), value :: coeff_offdiag
            type(c_ptr), value :: stream
        end subroutine launch_tf_local_pair_kernel

        subroutine launch_tf_local_pair_strided_kernel(psi, base_local, n_pairs, bit_mask, &
                                                       coeff_diag, coeff_offdiag, stream) &
            bind(c, name='launch_tf_local_pair_strided_kernel')
            import :: c_ptr, c_long, c_double_complex
            implicit none
            type(c_ptr), value :: psi
            integer(c_long), value :: base_local
            integer(c_long), value :: n_pairs
            integer(c_long), value :: bit_mask
            complex(c_double_complex), value :: coeff_diag
            complex(c_double_complex), value :: coeff_offdiag
            type(c_ptr), value :: stream
        end subroutine launch_tf_local_pair_strided_kernel

        subroutine launch_tf_remote_update_kernel(psi, recvbuf, local0, count, &
                                                  coeff_diag, coeff_offdiag, stream) &
            bind(c, name='launch_tf_remote_update_kernel')
            import :: c_ptr, c_long, c_double_complex
            implicit none
            type(c_ptr), value :: psi
            type(c_ptr), value :: recvbuf
            integer(c_long), value :: local0
            integer(c_long), value :: count
            complex(c_double_complex), value :: coeff_diag
            complex(c_double_complex), value :: coeff_offdiag
            type(c_ptr), value :: stream
        end subroutine launch_tf_remote_update_kernel

    end interface

end module hip_transverse_field_kernels
