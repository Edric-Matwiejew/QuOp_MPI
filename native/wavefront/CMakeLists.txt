find_library(NUMA_LIBRARY numa REQUIRED)

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
  target_link_libraries(${target} PUBLIC ${QUOP_HIPFORT_TARGET} MPI::MPI_Fortran quop_gpu_runtime ${_extra_links})
  set_target_properties(
    ${target} PROPERTIES POSITION_INDEPENDENT_CODE ON Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )
endfunction()

add_library(hip_kernels OBJECT hip_kernels.cpp)
quop_configure_gpu_kernels(hip_kernels)

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
add_subdirectory(transverse_field)
