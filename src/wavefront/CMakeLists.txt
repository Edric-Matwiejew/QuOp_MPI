find_library(NUMA_LIBRARY numa REQUIRED)

function(append_unique_flags var_name)
  set(_flags "${${var_name}}")
  foreach(_flag ${ARGN})
    string(FIND " ${_flags} " " ${_flag} " _flag_index)
    if(_flag_index EQUAL -1)
      string(APPEND _flags " ${_flag}")
    endif()
  endforeach()
  string(STRIP "${_flags}" _flags)
  set(${var_name}
      "${_flags}"
      PARENT_SCOPE
  )
endfunction()

if(HIP_PLATFORM STREQUAL nvidia)
  enable_language(CUDA)
  set(CMAKE_CUDA_HOST_COMPILER ${ROCM_PATH}/bin/hipcc)
  set_source_files_properties(hip_kernels.cpp PROPERTIES LANGUAGE CUDA)
  append_unique_flags(CMAKE_CUDA_FLAGS "--offload-arch=${OFFLOAD_ARCH}")
  append_unique_flags(CMAKE_CUDA_FLAGS_DEBUG "-g" "-O0")
  append_unique_flags(CMAKE_CUDA_FLAGS_RELEASE "-O3" "-DNDEBUG")
elseif(HIP_PLATFORM STREQUAL amd)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
  append_unique_flags(CMAKE_CXX_FLAGS "--offload-arch=${OFFLOAD_ARCH}")
  append_unique_flags(CMAKE_CXX_FLAGS_DEBUG "-fno-omit-frame-pointer")
  append_unique_flags(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
endif()

function(compile_and_link_hipfort_mpi target)
  set(_extra_includes "")
  set(_extra_links "")

  if(ARGC GREATER 1)
    set(_extra_includes ${ARGV1})
  endif()

  if(ARGC GREATER 2)
    set(_extra_links ${ARGV2})
  endif()

  target_include_directories(${target} PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY} ${_extra_includes})
  target_link_libraries(${target} PUBLIC ${QUOP_HIPFORT_TARGET} MPI::MPI_Fortran ${_extra_links})
  set_target_properties(
    ${target} PROPERTIES POSITION_INDEPENDENT_CODE ON Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )

  if(HIP_PLATFORM STREQUAL nvidia)
    target_link_libraries(${target} PUBLIC ${CUDA_PATH}/lib64/libcudart.so)
  elseif(HIP_PLATFORM STREQUAL amd)
    target_link_libraries(${target} PUBLIC ${ROCM_PATH}/lib/libamdhip64.so)
  endif()
endfunction()

add_library(hip_kernels OBJECT hip_kernels.cpp)
target_include_directories(hip_kernels PRIVATE "${ROCM_PATH}/include")
target_link_libraries(hip_kernels hip::device)
set_target_properties(hip_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(wavefront_utils OBJECT wavefront_utils.f90)
compile_and_link_hipfort_mpi(wavefront_utils)

add_subdirectory(context)

# FFT handler module (requires SHAFFT)
if(WITH_SHAFFT)
  add_subdirectory(shafft_handler)
  add_subdirectory(multivariable)
endif()

# Circulant propagator currently relies on SHAFFT.
if(WITH_SHAFFT)
  add_subdirectory(circulant)
endif()

add_subdirectory(diagonal)
add_subdirectory(sparse)
