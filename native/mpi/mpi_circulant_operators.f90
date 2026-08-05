module mpi_circulant_operators

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64

    implicit none

    private

    public :: graph_eigenvalues

    real(real64) :: PI = 3.141592653589793_real64; 
    interface graph_eigenvalues
        module procedure graph_eigenvalues_sparse
        module procedure graph_eigenvalues_dense
    end interface graph_eigenvalues

contains

    subroutine graph_eigenvalues_sparse(system_size, local_o, local_o_offset, nnz, indexes, values, eigenvalues)
        ! Real eigenvalues of a weighted circulant graph.
        integer(int64), intent(in) :: system_size
        integer(int64), intent(in) :: local_o
        integer(int64), intent(in) :: local_o_offset
        integer(int32), intent(in) :: nnz
        integer(int64), dimension(:), intent(in) :: indexes
        real(real64), dimension(:), intent(in) :: values
        real(real64), dimension(:), intent(inout) :: eigenvalues

        integer(int64) :: i, j

        do i = local_o_offset, local_o_offset + local_o - 1
            eigenvalues(i - local_o_offset + 1) = 0.0_real64
            do j = 1, size(indexes)
                eigenvalues(i - local_o_offset + 1) = eigenvalues(i - local_o_offset + 1) &
                                 + cos(2.0_real64 * real(i * (indexes(j) - 1), real64) * PI / real(system_size, real64)) * values(j)
            end do
        end do

    end subroutine graph_eigenvalues_sparse

    subroutine graph_eigenvalues_dense(system_size, local_o, local_o_offset, graph_array, eigenvalues)
        ! Real eigenvalues of a dense circulant graph.
        integer(int64), intent(in) :: system_size
        integer(int64), intent(in) :: local_o
        integer(int64), intent(in) :: local_o_offset
        real(real64), dimension(:), intent(in) :: graph_array
        real(real64), dimension(:), intent(inout) :: eigenvalues

        integer(int64) :: i, j

        do i = local_o_offset, local_o_offset + local_o - 1
            eigenvalues(i - local_o_offset + 1) = 0.0_real64
            do j = 0, system_size - 1
                eigenvalues(i - local_o_offset + 1) = eigenvalues(i - local_o_offset + 1) &
                                       + cos(2.0_real64 * real(i * j, real64) * PI / real(system_size, real64)) * graph_array(j + 1)
            end do
        end do

    end subroutine graph_eigenvalues_dense

end module mpi_circulant_operators
