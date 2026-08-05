add_library(wavefront_circulant OBJECT wavefront_circulant.f90)

compile_and_link_hipfort_mpi(wavefront_circulant)

get_target_property(_shafft_inc shafft::shafftf03 INTERFACE_INCLUDE_DIRECTORIES)
if(_shafft_inc)
  target_include_directories(wavefront_circulant PRIVATE ${_shafft_inc})
endif()

add_dependencies(
  wavefront_circulant
  wavefront_context
  wf_communicators
  wf_gpu_topology
  wf_partitions
  wf_gpu_transfer
  comm_info_mod
)

# Optional: add dependency on shafft_handler for future refactoring. Circulant is built only when WITH_SHAFFT is
# enabled.
if(WITH_SHAFFT AND TARGET shafft_handler)
  add_dependencies(wavefront_circulant shafft_handler shafft_handler_base shafft_handler_1d)
endif()

if(HIP_PLATFORM STREQUAL "nvidia")
  target_link_libraries(
    wavefront_circulant PUBLIC ${CUDA_PATH}/lib64/libcufft.so ${ROCM_PATH}/lib/libhipfft.so shafft::shafftf03
                               shafft::shafftc++
  )
elseif(HIP_PLATFORM STREQUAL "amd")
  target_link_libraries(wavefront_circulant PUBLIC ${ROCM_PATH}/lib/libhipfft.so shafft::shafftf03 shafft::shafftc++)
endif()

set(_depends
    wavefront_context
    wavefront_circulant
    hip_kernels
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    comm_info_mod
)

set(_definitions context=wavefront context_type=wavefront_context propagator=wavefront_circulant
                 propagator_type=circulant_propagator propagator_wrapper=circulant_propagator_wrapper
)

get_target_property(_hipfort_inc_dirs ${QUOP_HIPFORT_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
quop_preprocess_fortran_source(
  _wf_circulant_wrapper_src
  NAME
  "circulant_propagator_wrapper"
  SOURCE
  "${PROPAGATOR_WRAPPER}"
  DEFINITIONS
  ${_definitions}
  INCLUDE_DIRS
  "${CMAKE_Fortran_MODULE_DIRECTORY}"
  ${_hipfort_inc_dirs}
)

add_library(circulant_propagator_wrapper_mod OBJECT "${_wf_circulant_wrapper_src}")
set_target_properties(
  circulant_propagator_wrapper_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                              POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  circulant_propagator_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)
add_dependencies(circulant_propagator_wrapper_mod wavefront_circulant comm_info_mod)

add_f2py_library(
  MODULE_NAME
  wavefront_circulant_propagator
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

add_dependencies(wavefront_circulant_propagatorwrapper circulant_propagator_wrapper_mod)
add_dependencies(wavefront_circulant_propagatormodule circulant_propagator_wrapper_mod)

target_link_libraries(
  wavefront_circulant_propagator_f2py PRIVATE ${QUOP_HIPFORT_TARGET} ${NUMA_LIBRARY} hip::device
                                              ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_circulant_propagator_f2py PRIVATE CrayGTL::CrayGTL)
endif()
