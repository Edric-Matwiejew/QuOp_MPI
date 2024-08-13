add_library(mpi_diagonal OBJECT mpi_diagonal.f90)

set(depends mpi_context mpi_diagonal)
set(definitions context=mpi_backend context_type=mpi_context propagator=mpi_diagonal propagator_type=diagonal_propagator)

add_f2py_library(
	MODULE_NAME mpi_diagonal_propagator 
	DEPENDS "${depends}" 
	DEFINITIONS "${definitions}" 
	SRC ${PROPAGATOR_WRAPPER}
	INSTALL_SUBDIR mpi
	)
