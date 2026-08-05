add_library(sparse_vector OBJECT sparse_vector.f90)
target_link_libraries(sparse_vector PUBLIC MPI::MPI_Fortran)
