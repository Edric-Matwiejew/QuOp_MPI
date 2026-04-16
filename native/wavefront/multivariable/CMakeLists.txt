add_library(wavefront_composite OBJECT wavefront_composite.f90)

compile_and_link_hipfort_mpi(wavefront_composite)
add_dependencies(
  wavefront_composite
  wavefront_context
  wavefront_utils
  wf_gpu_topology
  wf_communicators
  wf_partitions
  shafft_comm_dims_utils
  shafft_handler
  shafft_handler_base
  shafft_handler_1d
  shafft_handler_nd
  comm_info_mod
)

add_library(wavefront_momentum OBJECT wavefront_momentum.f90)

compile_and_link_hipfort_mpi(wavefront_momentum)
add_dependencies(
  wavefront_momentum
  wavefront_context
  wavefront_utils
  wf_gpu_topology
  wf_communicators
  wf_partitions
  shafft_comm_dims_utils
  shafft_handler
  shafft_handler_base
  shafft_handler_1d
  shafft_handler_nd
  comm_info_mod
)

get_target_property(_shafft_inc shafft::shafftf03 INTERFACE_INCLUDE_DIRECTORIES)
if(_shafft_inc)
  target_include_directories(wavefront_composite PRIVATE ${_shafft_inc})
  target_include_directories(wavefront_momentum PRIVATE ${_shafft_inc})
endif()

if(HIP_PLATFORM STREQUAL nvidia)
  target_link_libraries(
    wavefront_composite PUBLIC ${CUDA_PATH}/lib64/libcufft.so ${ROCM_PATH}/lib/libhipfft.so shafft::shafftf03
                               shafft::shafftc++
  )
  target_link_libraries(
    wavefront_momentum PUBLIC ${CUDA_PATH}/lib64/libcufft.so ${ROCM_PATH}/lib/libhipfft.so shafft::shafftf03
                              shafft::shafftc++
  )
elseif(HIP_PLATFORM STREQUAL amd)
  target_link_libraries(wavefront_composite PUBLIC ${ROCM_PATH}/lib/libhipfft.so shafft::shafftf03 shafft::shafftc++)
  target_link_libraries(wavefront_momentum PUBLIC ${ROCM_PATH}/lib/libhipfft.so shafft::shafftf03 shafft::shafftc++)
endif()

set(_depends
    wavefront_context
    wavefront_composite
    hip_kernels
    wavefront_utils
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    shafft_comm_dims_utils
    shafft_handler
    shafft_handler_base
    shafft_handler_1d
    shafft_handler_nd
    comm_info_mod
)

set(_definitions context=wavefront context_type=wavefront_context propagator=wavefront_composite
                 propagator_type=composite_propagator propagator_wrapper=composite_propagator_wrapper
)

get_target_property(_hipfort_inc_dirs ${QUOP_HIPFORT_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
quop_preprocess_fortran_source(
  _wf_composite_wrapper_src
  NAME
  "composite_propagator_wrapper"
  SOURCE
  "${PROPAGATOR_WRAPPER}"
  DEFINITIONS
  ${_definitions}
  INCLUDE_DIRS
  "${CMAKE_Fortran_MODULE_DIRECTORY}"
  ${_hipfort_inc_dirs}
)

add_library(composite_propagator_wrapper_mod OBJECT "${_wf_composite_wrapper_src}")
set_target_properties(
  composite_propagator_wrapper_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                              POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  composite_propagator_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)
add_dependencies(composite_propagator_wrapper_mod wavefront_composite comm_info_mod)

add_f2py_library(
  MODULE_NAME
  wavefront_composite_propagator
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
  shafft::shafftf03
  shafft::shafftc++
  ${ROCM_PATH}/lib/libhipfft.so
)

add_dependencies(wavefront_composite_propagatorwrapper composite_propagator_wrapper_mod)
add_dependencies(wavefront_composite_propagatormodule composite_propagator_wrapper_mod)

target_link_libraries(
  wavefront_composite_propagator_f2py PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${QUOP_HIPFORT_TARGET}> ${NUMA_LIBRARY} hip::device
                                              ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_composite_propagator_f2py PRIVATE CrayGTL::CrayGTL)
endif()

# Momentum propagator
set(_momentum_depends
    wavefront_context
    wavefront_momentum
    hip_kernels
    wavefront_utils
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    shafft_comm_dims_utils
    shafft_handler
    shafft_handler_base
    shafft_handler_1d
    shafft_handler_nd
    comm_info_mod
)
set(_momentum_definitions context=wavefront context_type=wavefront_context propagator=wavefront_momentum
                          propagator_type=momentum_propagator propagator_wrapper=momentum_propagator_wrapper
)
quop_preprocess_fortran_source(
  _wf_momentum_wrapper_src
  NAME
  "momentum_propagator_wrapper"
  SOURCE
  "${PROPAGATOR_WRAPPER}"
  DEFINITIONS
  ${_momentum_definitions}
  INCLUDE_DIRS
  "${CMAKE_Fortran_MODULE_DIRECTORY}"
  ${_hipfort_inc_dirs}
)

add_library(momentum_propagator_wrapper_mod OBJECT "${_wf_momentum_wrapper_src}")
set_target_properties(
  momentum_propagator_wrapper_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                             POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  momentum_propagator_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)
add_dependencies(momentum_propagator_wrapper_mod wavefront_momentum comm_info_mod)

add_f2py_library(
  MODULE_NAME
  wavefront_momentum_propagator
  DEPENDS
  "${_momentum_depends}"
  DEFINITIONS
  "${_momentum_definitions}"
  SRC
  ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR
  wavefront
  LIBRARIES
  ${MPI_Fortran_LIBRARIES}
  ${QUOP_HIPFORT_TARGET}
  ${NUMA_LIBRARY}
  shafft::shafftf03
  shafft::shafftc++
  ${ROCM_PATH}/lib/libhipfft.so
)

add_dependencies(wavefront_momentum_propagatorwrapper momentum_propagator_wrapper_mod)
add_dependencies(wavefront_momentum_propagatormodule momentum_propagator_wrapper_mod)

target_link_libraries(
  wavefront_momentum_propagator_f2py PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${QUOP_HIPFORT_TARGET}> ${NUMA_LIBRARY} hip::device
                                             ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_momentum_propagator_f2py PRIVATE CrayGTL::CrayGTL)
endif()
