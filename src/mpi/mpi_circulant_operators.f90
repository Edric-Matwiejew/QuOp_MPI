module mpi_circulant_operators

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

    implicit none

    private

    public :: graph_eigenvalues

    real(dp) :: PI = 3.141592653589793_dp; 

    interface graph_eigenvalues
        module procedure graph_eigenvalues_sparse
        module procedure graph_eigenvalues_dense
    end interface graph_eigenvalues

contains

    subroutine graph_eigenvalues_sparse(system_size, local_o, local_o_offset, nnz, indexes, values, eigenvalues)
        ! real eiegenvlaues of circulant graph
        integer(dp), intent(in) :: system_size
        integer(dp), intent(in) :: local_o
        integer(dp), intent(in) :: local_o_offset
        integer(sp), intent(in) :: nnz
        integer(dp), dimension(:), intent(in) :: indexes
        real(dp), dimension(:), intent(in) :: values
        real(dp), dimension(:), intent(inout) :: eigenvalues

        integer(dp) :: i, j

        logical :: unit_complete = .false.
        integer(dp) :: weight_count = 0

        do i = local_o_offset, local_o_offset + local_o - 1
            eigenvalues(i - local_o_offset + 1) = 0.0_dp
            do j = 1, size(indexes)
                eigenvalues(i - local_o_offset + 1) = eigenvalues(i - local_o_offset + 1) &
                + cos(2.0_dp*real(i*(indexes(j) - 1), dp)*PI/real(system_size, dp))*values(j)
            end do
        end do

    end subroutine graph_eigenvalues_sparse


    subroutine graph_eigenvalues_dense(system_size, local_o, local_o_offset, graph_array, eigenvalues)
        ! real eiegenvlaues of circulant graph
        integer(dp), intent(in) :: system_size
        integer(dp), intent(in) :: local_o
        integer(dp), intent(in) :: local_o_offset
        real(dp), dimension(:), intent(in) :: graph_array
        real(dp), dimension(:), intent(inout) :: eigenvalues

        integer(dp) :: i, j

        logical :: unit_complete = .false.
        integer(dp) :: weight_count = 0

        do i = local_o_offset, local_o_offset + local_o - 1
            eigenvalues(i - local_o_offset + 1) = 0.0_dp
            do j = 0, system_size - 1
                eigenvalues(i - local_o_offset + 1) = eigenvalues(i - local_o_offset + 1) &
                                                      + cos(2.0_dp*real(i*j, dp)*PI/real(system_size, dp))*graph_array(j + 1)
            end do
        end do

    end subroutine graph_eigenvalues_dense

end module mpi_circulant_operators
