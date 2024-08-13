add_library(mpi_circulant OBJECT mpi_circulant.f90)
 
target_link_libraries(
    mpi_circulant
	"${FFTW_DOUBLE_LIB}"
	"${FFTW_DOUBLE_MPI_LIB}" 
)

target_include_directories(
    mpi_circulant
 	PUBLIC
	${FFTW_INCLUDE_DIRS}
)

set(depends mpi_circulant_operators mpi_context mpi_circulant sparse_vector)
set(definitions context=mpi_backend context_type=mpi_context propagator=mpi_circulant propagator_type=circulant_propagator)

add_f2py_library(
	MODULE_NAME mpi_circulant_propagator 
	DEPENDS "${depends}" 
	DEFINITIONS "${definitions}" 
	SRC ${PROPAGATOR_WRAPPER}
	INSTALL_SUBDIR mpi
	)
