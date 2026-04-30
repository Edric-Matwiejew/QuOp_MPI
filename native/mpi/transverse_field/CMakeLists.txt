add_library(mpi_transverse_field OBJECT mpi_transverse_field.f90)
target_link_libraries(mpi_transverse_field PUBLIC MPI::MPI_Fortran)
add_dependencies(mpi_transverse_field mpi_context transverse_field_common comm_info_mod)

target_include_directories(mpi_transverse_field PUBLIC
  ${CMAKE_Fortran_MODULE_DIRECTORY}/mpi_context)

set(_depends mpi_context mpi_transverse_field transverse_field_common comm_info_mod ${_COMM_INFO_EXTRA})
set(_definitions context=mpi_backend context_type=mpi_context propagator=mpi_transverse_field propagator_type=transverse_field_propagator propagator_wrapper=transverse_field_propagator_wrapper)

add_f2py_library(
  MODULE_NAME mpi_transverse_field_propagator
  DEPENDS "${_depends}"
  DEFINITIONS "${_definitions}"
  SRC ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR mpi
  )
