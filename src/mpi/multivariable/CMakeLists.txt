add_library(mpi_composite OBJECT mpi_composite.f90)
 
target_link_libraries(
    mpi_composite
  MPI::MPI_Fortran
  "${FFTW_DOUBLE_LIB}"
  "${FFTW_DOUBLE_MPI_LIB}" 
    mpi_circulant_operators
    mpi_context
)
target_include_directories(
    mpi_composite
    PUBLIC
  ${FFTW_INCLUDE_DIRS}
  ${CMAKE_Fortran_MODULE_DIRECTORY}/cartesian
    ${CMAKE_Fortran_MODULE_DIRECTORY}/mpi_circulant_operators
    ${CMAKE_Fortran_MODULE_DIRECTORY}/mpi_context
)

add_library(cartesian_obj OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../../cartesian/cartesian.f90)

add_dependencies(mpi_composite cartesian_obj comm_info_mod)

set(_depends mpi_context mpi_composite mpi_circulant_operators cartesian_obj comm_info_mod ${_COMM_INFO_EXTRA})
set(_definitions context=mpi_backend context_type=mpi_context propagator=mpi_composite propagator_type=composite_propagator propagator_wrapper=composite_propagator_wrapper)

add_f2py_library(
  MODULE_NAME mpi_composite_propagator 
  DEPENDS "${_depends}" 
  DEFINITIONS "${_definitions}" 
  SRC ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR mpi
  )

# MPI Momentum propagator for QOWE algorithm
add_library(mpi_momentum OBJECT mpi_momentum.f90)

target_link_libraries(
    mpi_momentum
    MPI::MPI_Fortran
    "${FFTW_DOUBLE_LIB}"
    "${FFTW_DOUBLE_MPI_LIB}"
    mpi_context
)

target_include_directories(
    mpi_momentum
    PUBLIC
    ${FFTW_INCLUDE_DIRS}
    ${CMAKE_Fortran_MODULE_DIRECTORY}/cartesian
    ${CMAKE_Fortran_MODULE_DIRECTORY}/mpi_context
)

add_dependencies(mpi_momentum cartesian_obj comm_info_mod)

set(_momentum_depends mpi_context mpi_momentum cartesian_obj comm_info_mod ${_COMM_INFO_EXTRA})
set(_momentum_definitions context=mpi_backend context_type=mpi_context propagator=mpi_momentum propagator_type=momentum_propagator propagator_wrapper=momentum_propagator_wrapper)

add_f2py_library(
    MODULE_NAME mpi_momentum_propagator
    DEPENDS "${_momentum_depends}"
    DEFINITIONS "${_momentum_definitions}"
    SRC ${PROPAGATOR_WRAPPER}
    INSTALL_SUBDIR mpi
)
