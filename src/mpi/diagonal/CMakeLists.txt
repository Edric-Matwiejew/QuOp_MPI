add_library(mpi_diagonal OBJECT mpi_diagonal.f90)
target_link_libraries(mpi_diagonal PUBLIC MPI::MPI_Fortran)

set(depends mpi_context mpi_diagonal)
set(definitions context=mpi_backend context_type=mpi_context propagator=mpi_diagonal propagator_type=diagonal_propagator propagator_wrapper=diagonal_propagator_wrapper)

add_f2py_library(
	MODULE_NAME mpi_diagonal_propagator 
	DEPENDS "${depends}" 
	DEFINITIONS "${definitions}" 
	SRC ${PROPAGATOR_WRAPPER}
	INSTALL_SUBDIR mpi
	)
