add_library(sparse OBJECT sparse.f90)
add_library(one_norms OBJECT one_norms.f90)
add_dependencies(one_norms sparse)
add_library(expm OBJECT expm.f90)
#add_dependencies(expm one_norms sparse)
add_dependencies(expm one_norms)
add_library(mpi_sparse OBJECT mpi_sparse.f90)
#add_dependencies(mpi_sparse expm one_norms sparse)
add_dependencies(mpi_sparse expm)

set(depends expm one_norms sparse mpi_context mpi_sparse)
set(definitions context=mpi_backend context_type=mpi_context propagator=mpi_sparse propagator_type=sparse_propagator)

add_f2py_library(
	MODULE_NAME mpi_sparse_propagator
	DEPENDS "${depends}"
	DEFINITIONS "${definitions}" 
	SRC ${PROPAGATOR_WRAPPER}
	INSTALL_SUBDIR mpi
	)
