add_library(mpi_composite OBJECT mpi_composite.f90)
 
target_link_libraries(
    mpi_composite
	"${FFTW_DOUBLE_LIB}"
	"${FFTW_DOUBLE_MPI_LIB}" 
)

target_include_directories(
    mpi_composite
 	PUBLIC
	${FFTW_INCLUDE_DIRS}
	${CMAKE_Fortran_MODULE_DIRECTORY}/cartesian
)

add_library(cartesian_obj OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../../cartesian/cartesian.f90)

add_dependencies(mpi_composite cartesian_obj)

set(depends mpi_context mpi_composite mpi_circulant_operators cartesian_obj)
set(definitions context=mpi_backend context_type=mpi_context propagator=mpi_composite propagator_type=composite_propagator propagator_wrapper=composite_propagator_wrapper)

add_f2py_library(
	MODULE_NAME mpi_composite_propagator 
	DEPENDS "${depends}" 
	DEFINITIONS "${definitions}" 
	SRC ${PROPAGATOR_WRAPPER}
	INSTALL_SUBDIR mpi
	)
