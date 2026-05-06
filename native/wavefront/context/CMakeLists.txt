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

# -----------------------------------------------------------------------------
# Backend-agnostic CPython context-wrapper extension (zero-copy state buffer).
# Mirrors the MPI backend's invocation in native/mpi/CMakeLists.txt.
# -----------------------------------------------------------------------------
add_context_extension(
  NAME           context_wrapper
  INSTALL_SUBDIR wavefront
  DEFINITIONS    ${_definitions}
  DEPENDS        ${_depends}
  LIBRARIES      ${QUOP_HIPFORT_TARGET} ${NUMA_LIBRARY} hip::device
                 ${ROCM_PATH}/lib/libamdhip64.so
)
