set(TF_KERNELS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# HIP kernel compilation
if(HIP_PLATFORM STREQUAL "nvidia")
  set_source_files_properties(
    ${TF_KERNELS_SRC}/hip_transverse_field_kernels.cpp PROPERTIES LANGUAGE CUDA
  )
endif()

add_library(tf_hip_kernels OBJECT ${TF_KERNELS_SRC}/hip_transverse_field_kernels.cpp)
target_include_directories(tf_hip_kernels PRIVATE ${TF_KERNELS_SRC} "${ROCM_PATH}/include")
target_link_libraries(tf_hip_kernels hip::device)
set_target_properties(tf_hip_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Fortran HIP kernel bindings
add_library(hip_transverse_field_kernels OBJECT hip_transverse_field_kernels.f90)
compile_and_link_hipfort_mpi(hip_transverse_field_kernels)

# Preprocess wavefront_transverse_field.f90 for GPU-aware MPI
set(_wf_tf_pp_defs "")
if(GPU_AWARE_MPI)
  list(APPEND _wf_tf_pp_defs QUOP_GPU_AWARE_MPI)
endif()

quop_preprocess_fortran_source(
  _wf_tf_src
  NAME "wavefront_transverse_field_source"
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/wavefront_transverse_field.f90"
  DEFINITIONS ${_wf_tf_pp_defs}
)

add_library(wavefront_transverse_field OBJECT "${_wf_tf_src}")
compile_and_link_hipfort_mpi(wavefront_transverse_field)
add_dependencies(
  wavefront_transverse_field
  wavefront_context
  transverse_field_common
  hip_transverse_field_kernels
  comm_info_mod
)

# f2py wrapper
set(_depends
    wavefront_transverse_field
    hip_transverse_field_kernels
    tf_hip_kernels
    hip_kernels
    transverse_field_common
    wavefront_context
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    comm_info_mod
)
set(_definitions
    context=wavefront
    context_type=wavefront_context
    propagator=wavefront_transverse_field
    propagator_type=transverse_field_propagator
    propagator_wrapper=transverse_field_propagator_wrapper
)

set(PROPAGATOR_WRAPPER "${CMAKE_SOURCE_DIR}/native/propagator_wrapper.f90")

get_target_property(_hipfort_inc_dirs ${QUOP_HIPFORT_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
quop_preprocess_fortran_source(
  _wf_tf_wrapper_src
  NAME "wavefront_transverse_field_propagator_wrapper"
  SOURCE "${PROPAGATOR_WRAPPER}"
  DEFINITIONS ${_definitions}
  INCLUDE_DIRS "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)

add_library(wf_tf_propagator_wrapper_mod OBJECT "${_wf_tf_wrapper_src}")
set_target_properties(
  wf_tf_propagator_wrapper_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                          POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  wf_tf_propagator_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)
add_dependencies(wf_tf_propagator_wrapper_mod wavefront_transverse_field comm_info_mod)

add_f2py_library(
  MODULE_NAME wavefront_transverse_field_propagator
  DEPENDS "${_depends}"
  DEFINITIONS "${_definitions}"
  SRC ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR wavefront
  LIBRARIES ${MPI_Fortran_LIBRARIES} ${QUOP_HIPFORT_TARGET} ${NUMA_LIBRARY}
)

add_dependencies(wavefront_transverse_field_propagatorwrapper wf_tf_propagator_wrapper_mod)
add_dependencies(wavefront_transverse_field_propagatormodule wf_tf_propagator_wrapper_mod)

target_link_libraries(
  wavefront_transverse_field_propagator_f2py PRIVATE ${QUOP_HIPFORT_TARGET} ${NUMA_LIBRARY} hip::device
                                                     ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_transverse_field_propagator_f2py PRIVATE CrayGTL::CrayGTL)
endif()
