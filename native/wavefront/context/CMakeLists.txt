find_library(NUMA_LIBRARY numa REQUIRED)

add_library(wf_gpu_topology OBJECT numa_detect.f90 gpu_topology.f90)
add_library(wf_communicators OBJECT communicators.f90)
add_library(wf_partitions OBJECT partitions.f90)
add_library(wf_transfer OBJECT transfer.f90)
add_dependencies(wf_communicators wf_gpu_topology)
add_library(wavefront_context OBJECT wavefront_context.f90)
add_dependencies(
  wavefront_context
  wf_gpu_topology
  wf_communicators
  wf_partitions
  wf_transfer
  wf_gpu_transfer
  comm_info_mod
)

set(_wf_context_pp_defs "")
if(GPU_AWARE_MPI)
  list(APPEND _wf_context_pp_defs QUOP_GPU_AWARE_MPI)
endif()

quop_preprocess_fortran_source(
  _wf_gpu_transfer_src
  NAME
  "wf_gpu_transfer"
  SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu_transfer.f90"
  DEFINITIONS
  ${_wf_context_pp_defs}
)

add_library(wf_gpu_transfer OBJECT "${_wf_gpu_transfer_src}")

# CMake's Fortran dyndep handling does not always serialize this source-level module edge when wf_gpu_transfer is built
# from a generated preprocessed file.
quop_add_fortran_source_barrier(
  SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/wavefront_context.f90"
  NAME
  "wavefront_context_modules_ready"
  COMMENT
  "Waiting for wavefront context module dependencies"
  DEPENDS
  wf_gpu_transfer
  wf_gpu_topology
  comm_info_mod
)

compile_and_link_hipfort_mpi(wf_gpu_topology)
compile_and_link_hipfort_mpi(wf_communicators)
compile_and_link_hipfort_mpi(wf_partitions)
compile_and_link_hipfort_mpi(wf_transfer)
compile_and_link_hipfort_mpi(wf_gpu_transfer)
compile_and_link_hipfort_mpi(wavefront_context)

target_link_libraries(wf_gpu_topology PUBLIC ${NUMA_LIBRARY})

set(_depends
    wavefront_context
    hip_kernels
    wf_gpu_topology
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_transfer
    comm_info_mod
)
set(_definitions context=wavefront context_type=wavefront_context)

set(CONTEXT_WRAPPER "${CMAKE_SOURCE_DIR}/native/context_wrapper.f90")

quop_preprocess_fortran_source(
  _context_wrapper_src
  NAME
  "context_wrapper_mod"
  SOURCE
  "${CONTEXT_WRAPPER}"
  DEFINITIONS
  ${_definitions}
)

add_library(context_wrapper_mod OBJECT "${_context_wrapper_src}")
target_include_directories(context_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}")
set_target_properties(context_wrapper_mod PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(context_wrapper_mod MPI::MPI_Fortran ${QUOP_HIPFORT_TARGET})
add_dependencies(context_wrapper_mod wavefront_context comm_info_mod)

add_f2py_library(
  MODULE_NAME
  wavefront_context
  DEPENDS
  "${_depends}"
  DEFINITIONS
  "${_definitions}"
  SRC
  ${CONTEXT_WRAPPER}
  INSTALL_SUBDIR
  wavefront
  LIBRARIES
  ${MPI_Fortran_LIBRARIES}
  ${QUOP_HIPFORT_TARGET}
  ${NUMA_LIBRARY}
)

add_dependencies(wavefront_contextwrapper context_wrapper_mod)
add_dependencies(wavefront_contextmodule context_wrapper_mod)

target_link_libraries(
  wavefront_context_f2py PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${QUOP_HIPFORT_TARGET}> ${NUMA_LIBRARY} hip::device ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_context_f2py PRIVATE CrayGTL::CrayGTL)
endif()
