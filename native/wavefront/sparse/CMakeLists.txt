set(SPARSE_PROPAGATORS_SRC ${CMAKE_SOURCE_DIR}/native/sparse_propagators/src)
set(SPARSE_KERNELS_SRC ${SPARSE_PROPAGATORS_SRC}/kernels)

if(HIP_PLATFORM STREQUAL nvidia)
  set_source_files_properties(
    ${SPARSE_KERNELS_SRC}/hip_vector_kernels.cpp ${SPARSE_KERNELS_SRC}/hip_spmv_kernels.cpp
    ${SPARSE_KERNELS_SRC}/hip_chebyshev_kernels.cpp ${SPARSE_KERNELS_SRC}/hip_reduction_kernels.cpp PROPERTIES LANGUAGE
                                                                                                               CUDA
  )
endif()

add_library(
  sparse_expm_kernels OBJECT
  ${SPARSE_KERNELS_SRC}/hip_vector_kernels.cpp ${SPARSE_KERNELS_SRC}/hip_spmv_kernels.cpp
  ${SPARSE_KERNELS_SRC}/hip_chebyshev_kernels.cpp ${SPARSE_KERNELS_SRC}/hip_reduction_kernels.cpp
)
target_include_directories(sparse_expm_kernels PRIVATE ${SPARSE_KERNELS_SRC})

add_library(hip_sparse_expm_kernels OBJECT ${SPARSE_PROPAGATORS_SRC}/hip_sparse_expm_kernels.f90)
compile_and_link_hipfort_mpi(hip_sparse_expm_kernels)

set(_wf_sparse_pp_defs USE_HIP)
if(GPU_AWARE_MPI)
  list(APPEND _wf_sparse_pp_defs QUOP_GPU_AWARE_MPI)
endif()

quop_preprocess_fortran_source(
  _wf_sparse_src
  NAME
  "wavefront_sparse_source"
  SOURCE
  "${SPARSE_PROPAGATORS_SRC}/sparse.f90"
  DEFINITIONS
  ${_wf_sparse_pp_defs}
)

add_library(sparse_hip OBJECT "${_wf_sparse_src}")
compile_and_link_hipfort_mpi(sparse_hip)
add_dependencies(sparse_hip hip_sparse_expm_kernels)

add_library(one_norms_hip OBJECT ${SPARSE_PROPAGATORS_SRC}/one_norms.f90)
target_link_libraries(one_norms_hip PUBLIC MPI::MPI_Fortran)
add_dependencies(one_norms_hip sparse_hip)

add_library(expm_hip OBJECT ${SPARSE_PROPAGATORS_SRC}/expm.f90)
target_compile_definitions(expm_hip PUBLIC USE_HIP)
compile_and_link_hipfort_mpi(expm_hip)
add_dependencies(expm_hip one_norms_hip sparse_hip)

quop_preprocess_fortran_source(
  _wf_chebyshev_src
  NAME
  "wavefront_chebyshev_source"
  SOURCE
  "${SPARSE_PROPAGATORS_SRC}/chebyshev.f90"
  DEFINITIONS
  ${_wf_sparse_pp_defs}
)

add_library(chebyshev_hip OBJECT "${_wf_chebyshev_src}")
compile_and_link_hipfort_mpi(chebyshev_hip)
add_dependencies(chebyshev_hip sparse_hip hip_sparse_expm_kernels)

add_library(wavefront_sparse OBJECT wavefront_sparse.f90)
target_compile_definitions(wavefront_sparse PUBLIC USE_HIP)
compile_and_link_hipfort_mpi(wavefront_sparse)
add_dependencies(
  wavefront_sparse
  wavefront_context
  expm_hip
  chebyshev_hip
  one_norms_hip
  sparse_hip
  hip_sparse_expm_kernels
  wf_transfer
  wf_gpu_topology
  wf_gpu_transfer
  comm_info_mod
)

set(_depends
    wavefront_sparse
    expm_hip
    chebyshev_hip
    one_norms_hip
    sparse_hip
    hip_sparse_expm_kernels
    sparse_expm_kernels
    hip_kernels
    wavefront_context
    wavefront_utils
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    comm_info_mod
)
set(_definitions context=wavefront context_type=wavefront_context propagator=wavefront_sparse
                 propagator_type=sparse_propagator propagator_wrapper=sparse_propagator_wrapper
)

set(PROPAGATOR_WRAPPER "${CMAKE_SOURCE_DIR}/native/propagator_wrapper.f90")

get_target_property(_hipfort_inc_dirs ${QUOP_HIPFORT_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
quop_preprocess_fortran_source(
  _wf_sparse_wrapper_src
  NAME
  "sparse_propagator_wrapper"
  SOURCE
  "${PROPAGATOR_WRAPPER}"
  DEFINITIONS
  ${_definitions}
  INCLUDE_DIRS
  "${CMAKE_Fortran_MODULE_DIRECTORY}"
  ${_hipfort_inc_dirs}
)

add_library(sparse_propagator_wrapper_mod OBJECT "${_wf_sparse_wrapper_src}")
set_target_properties(
  sparse_propagator_wrapper_mod PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                           POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  sparse_propagator_wrapper_mod PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}" ${_hipfort_inc_dirs}
)
add_dependencies(sparse_propagator_wrapper_mod wavefront_sparse comm_info_mod)

add_f2py_library(
  MODULE_NAME
  wavefront_sparse_propagator
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

add_dependencies(wavefront_sparse_propagatorwrapper sparse_propagator_wrapper_mod)
add_dependencies(wavefront_sparse_propagatormodule sparse_propagator_wrapper_mod)

target_link_libraries(
  wavefront_sparse_propagator_f2py PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${QUOP_HIPFORT_TARGET}> ${NUMA_LIBRARY} hip::device
                                           ${ROCM_PATH}/lib/libamdhip64.so
)
if(TARGET CrayGTL::CrayGTL)
  target_link_libraries(wavefront_sparse_propagator_f2py PRIVATE CrayGTL::CrayGTL)
endif()
