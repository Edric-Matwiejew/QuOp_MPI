!------------------------------------------------------------------------------
!> @brief Fortran bind(c) interfaces for HIP transverse-field kernels.
!>
!> @details Provides Fortran-callable interfaces for the three GPU kernels
!> defined in kernels/hip_transverse_field_kernels.cpp:
!>   - launch_tf_local_pair_kernel
!>   - launch_tf_remote_update_kernel
!>   - launch_tf_pack_send_kernel
!------------------------------------------------------------------------------
module hip_transverse_field_kernels

    use, intrinsic :: iso_c_binding

    implicit none

    private

    public :: launch_tf_local_pair_kernel
    public :: launch_tf_remote_update_kernel
    public :: launch_tf_pack_send_kernel

    interface

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

        subroutine launch_tf_pack_send_kernel(sendbuf, psi, local0, count, stream) &
            bind(c, name='launch_tf_pack_send_kernel')
            import :: c_ptr, c_long
            implicit none
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: psi
            integer(c_long), value :: local0
            integer(c_long), value :: count
            type(c_ptr), value :: stream
        end subroutine launch_tf_pack_send_kernel

    end interface

end module hip_transverse_field_kernels
