add_library(wavefront_diagonal OBJECT wavefront_diagonal.f90)
compile_and_link_hipfort_mpi(wavefront_diagonal)
add_dependencies(wavefront_diagonal wavefront_context wf_gpu_transfer comm_info_mod)

set(_depends
    wavefront_context
    wavefront_diagonal
    hip_kernels
    wavefront_utils
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    comm_info_mod
)
set(_definitions context=wavefront context_type=wavefront_context propagator=wavefront_diagonal
                 propagator_type=diagonal_propagator propagator_wrapper=diagonal_propagator_wrapper
)

set(PROPAGATOR_WRAPPER "${CMAKE_SOURCE_DIR}/src/propagator_wrapper.f90")

get_target_property(_hipfort_inc_dirs ${QUOP_HIPFORT_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
quop_preprocess_fortran_source(
  _wf_diagonal_wrapper_src
  NAME
  "diagonal_propagator_wrapper"
  SOURCE
  "${PROPAGATOR_WRAPPER}"
  DEFINITIONS
  ${_definitions}
  INCLUDE_DIRS
  "${CMAKE_Fortran_MODULE_DIRECTORY}"
  ${_hipfort_inc_dirs}
)

add_library(diagonal_propagator_wrapper_mod OBJECT "${_wf_diagonal_wrapper_src}")
set_target_properties(
  diagonal_propagator_wrapper_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                             POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  diagonal_propagator_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)
add_dependencies(diagonal_propagator_wrapper_mod wavefront_diagonal comm_info_mod)

add_f2py_library(
  MODULE_NAME
  wavefront_diagonal_propagator
  DEPENDS
  "${_depends}"
  DEFINITIONS
  "${_definitions}"
  SRC
  ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR
  wavefront
  LIBRARIES
  ${MPI_Fortran_LIBRARIES}
  ${QUOP_HIPFORT_TARGET}
  ${NUMA_LIBRARY}
)

add_dependencies(wavefront_diagonal_propagatorwrapper diagonal_propagator_wrapper_mod)
add_dependencies(wavefront_diagonal_propagatormodule diagonal_propagator_wrapper_mod)

target_link_libraries(
  wavefront_diagonal_propagator_f2py PRIVATE ${QUOP_HIPFORT_TARGET} ${NUMA_LIBRARY} hip::device
                                             ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_diagonal_propagator_f2py PRIVATE CrayGTL::CrayGTL)
endif()
