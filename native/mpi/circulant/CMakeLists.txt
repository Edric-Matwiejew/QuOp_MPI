add_library(mpi_circulant OBJECT mpi_circulant.f90)
 
target_link_libraries(
    mpi_circulant
  MPI::MPI_Fortran
  "${FFTW_DOUBLE_LIB}"
  "${FFTW_DOUBLE_MPI_LIB}" 
    mpi_circulant_operators
    sparse_vector
    mpi_context
    fftw_mpi_init_guard
)
add_dependencies(mpi_circulant comm_info_mod)

target_include_directories(
    mpi_circulant
    PUBLIC
  ${FFTW_INCLUDE_DIRS}
)

set(_depends mpi_circulant_operators mpi_context mpi_circulant sparse_vector comm_info_mod fftw_mpi_init_guard ${_COMM_INFO_EXTRA})
set(_definitions context=mpi_backend context_type=mpi_context propagator=mpi_circulant propagator_type=circulant_propagator propagator_wrapper=circulant_propagator_wrapper)

add_f2py_library(
  MODULE_NAME mpi_circulant_propagator 
  DEPENDS "${_depends}" 
  DEFINITIONS "${_definitions}" 
  SRC ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR mpi
  )
