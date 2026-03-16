# src/comm_info/CMakeLists.txt Build configuration for comm_info_module (quop_mpi_layout_t + split_info_t) and the
# comm_info_wrapper f2py extension.
#
# MPI backend: uses gpu_topology_stub.f90, no CPP definitions needed. Wavefront backend: uses the real gpu_topology.f90
# and communicators.f90 from src/wavefront/context, defines WAVEFRONT_BACKEND so comm_info_module.f90 enables GPU
# topology detection, communicator creation, and rebuild/shrink cascading.  These wavefront sources are compiled as
# OBJECT libraries here (ci_gpu_topology, ci_communicators) so they are available before the wavefront subdirectory is
# processed.

# ============================================================================
# 1. Fortran OBJECT library for the module (produces comm_info_module.mod)
# ============================================================================
set(_COMM_INFO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/comm_info_module.f90")

if(WAVEFRONT_BACKEND)
  find_library(NUMA_LIBRARY numa REQUIRED)

  # Compile the real gpu_topology and communicators modules as local OBJECT libraries so comm_info_mod can depend on
  # them without circular ordering issues with the wavefront subdirectory.
  #
  # These sources are also compiled in src/wavefront/context/ (wf_gpu_topology etc.).  To avoid duplicate .mod providers
  # - which breaks Ninja's Fortran dyndep module-dependency resolution - the ci_* targets write their .mod files to a
  # private directory instead of the global CMAKE_Fortran_MODULE_DIRECTORY.
  set(_WF_CONTEXT_DIR "${CMAKE_SOURCE_DIR}/src/wavefront/context")
  set(_CI_MOD_DIR "${CMAKE_CURRENT_BINARY_DIR}/ci_modules")

  add_library(ci_gpu_topology OBJECT "${_WF_CONTEXT_DIR}/numa_detect.f90" "${_WF_CONTEXT_DIR}/gpu_topology.f90")
  target_include_directories(ci_gpu_topology PUBLIC "${_CI_MOD_DIR}" "${CMAKE_Fortran_MODULE_DIRECTORY}")
  target_link_libraries(ci_gpu_topology PUBLIC MPI::MPI_Fortran ${QUOP_HIPFORT_TARGET} ${NUMA_LIBRARY})
  if(HIP_PLATFORM STREQUAL "nvidia")
    target_link_libraries(ci_gpu_topology PUBLIC ${CUDA_PATH}/lib64/libcudart.so)
  elseif(HIP_PLATFORM STREQUAL "amd")
    target_link_libraries(ci_gpu_topology PUBLIC ${ROCM_PATH}/lib/libamdhip64.so)
  endif()
  set_target_properties(
    ci_gpu_topology PROPERTIES Fortran_MODULE_DIRECTORY "${_CI_MOD_DIR}" POSITION_INDEPENDENT_CODE ON
  )

  add_library(ci_communicators OBJECT "${_WF_CONTEXT_DIR}/communicators.f90")
  target_include_directories(ci_communicators PUBLIC "${_CI_MOD_DIR}" "${CMAKE_Fortran_MODULE_DIRECTORY}")
  target_link_libraries(ci_communicators PUBLIC MPI::MPI_Fortran ${QUOP_HIPFORT_TARGET})
  if(HIP_PLATFORM STREQUAL "nvidia")
    target_link_libraries(ci_communicators PUBLIC ${CUDA_PATH}/lib64/libcudart.so)
  elseif(HIP_PLATFORM STREQUAL "amd")
    target_link_libraries(ci_communicators PUBLIC ${ROCM_PATH}/lib/libamdhip64.so)
  endif()
  add_dependencies(ci_communicators ci_gpu_topology)
  set_target_properties(
    ci_communicators PROPERTIES Fortran_MODULE_DIRECTORY "${_CI_MOD_DIR}" POSITION_INDEPENDENT_CODE ON
  )

  add_library(ci_partitions OBJECT "${_WF_CONTEXT_DIR}/partitions.f90")
  target_include_directories(ci_partitions PUBLIC "${_CI_MOD_DIR}" "${CMAKE_Fortran_MODULE_DIRECTORY}")
  target_link_libraries(ci_partitions PUBLIC MPI::MPI_Fortran)
  set_target_properties(ci_partitions PROPERTIES Fortran_MODULE_DIRECTORY "${_CI_MOD_DIR}" POSITION_INDEPENDENT_CODE ON)

  quop_preprocess_fortran_source(
    _COMM_INFO_PREPROCESSED
    NAME
    "comm_info_module"
    SOURCE
    "${_COMM_INFO_SRC}"
    DEFINITIONS
    WAVEFRONT_BACKEND
    INCLUDE_DIRS
    "${_CI_MOD_DIR}"
  )

  # Wavefront backend: compile comm_info_module with explicit preprocessing.
  add_library(comm_info_mod OBJECT "${_COMM_INFO_PREPROCESSED}")
  target_include_directories(comm_info_mod PRIVATE "${_CI_MOD_DIR}")
  add_dependencies(comm_info_mod ci_gpu_topology ci_communicators ci_partitions)
  target_link_libraries(comm_info_mod PUBLIC MPI::MPI_Fortran ${QUOP_HIPFORT_TARGET})

else()
  quop_preprocess_fortran_source(_COMM_INFO_PREPROCESSED NAME "comm_info_module" SOURCE "${_COMM_INFO_SRC}")

  # MPI backend: use the stub gpu_topology module (no HIP dependencies).
  add_library(comm_info_mod OBJECT gpu_topology_stub.f90 "${_COMM_INFO_PREPROCESSED}")
  target_link_libraries(comm_info_mod PUBLIC MPI::MPI_Fortran)
endif()

set_target_properties(
  comm_info_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}" POSITION_INDEPENDENT_CODE ON
)
if(WAVEFRONT_BACKEND)
  set_target_properties(
    comm_info_mod PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_Fortran_MODULE_DIRECTORY};${_CI_MOD_DIR}"
  )
else()
  set_target_properties(comm_info_mod PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_Fortran_MODULE_DIRECTORY}")
endif()

# ============================================================================
# 1. f2py wrapper -- produces comm_info_wrapper.cpython-*.so
# ============================================================================
if(WAVEFRONT_BACKEND)
  add_f2py_library(
    MODULE_NAME
    comm_info_wrapper
    DEPENDS
    comm_info_mod
    ci_gpu_topology
    ci_communicators
    ci_partitions
    SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/comm_info_wrapper.f90"
    INSTALL_SUBDIR
    ""
    LIBRARIES
    MPI::MPI_Fortran
    ${QUOP_HIPFORT_TARGET}
    ${NUMA_LIBRARY}
  )
else()
  add_f2py_library(
    MODULE_NAME
    comm_info_wrapper
    DEPENDS
    comm_info_mod
    SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/comm_info_wrapper.f90"
    INSTALL_SUBDIR
    ""
    LIBRARIES
    MPI::MPI_Fortran
  )
endif()
