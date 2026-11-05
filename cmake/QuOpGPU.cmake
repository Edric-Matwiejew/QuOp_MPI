# Shared GPU compiler and runtime configuration for QuOp and its native tests.
include_guard(GLOBAL)

if(NOT HIP_PLATFORM MATCHES "^(amd|nvidia)$")
  message(FATAL_ERROR "HIP_PLATFORM must be amd or nvidia")
endif()
if(NOT OFFLOAD_ARCH)
  message(FATAL_ERROR "Set OFFLOAD_ARCH to the target GPU architecture (gfx90a for MI250X, 90 for GH200)")
endif()

foreach(_prefix IN ITEMS "${HIP_PATH}" "${ROCM_PATH}")
  if(_prefix)
    list(APPEND CMAKE_PREFIX_PATH "${_prefix}")
  endif()
endforeach()

if(HIP_PLATFORM STREQUAL "nvidia")
  # Accept legacy sm_XX spelling, but use CMake's numeric architecture names.
  string(REGEX REPLACE "(^|;)(sm_|compute_)" "\\1" OFFLOAD_ARCH "${OFFLOAD_ARCH}")
  set(OFFLOAD_ARCH "${OFFLOAD_ARCH}" CACHE STRING "Target GPU architectures" FORCE)
  foreach(_arch IN LISTS OFFLOAD_ARCH)
    if(NOT _arch MATCHES "^[0-9]+[a-z]?(-real|-virtual)?$")
      message(FATAL_ERROR "Invalid NVIDIA GPU architecture: ${_arch}")
    endif()
  endforeach()
  if(CUDA_PATH)
    set(CUDAToolkit_ROOT "${CUDA_PATH}")
  endif()
  enable_language(CXX)
  find_package(CUDAToolkit REQUIRED)
  if(DEFINED CMAKE_HIP_PLATFORM AND NOT CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    message(FATAL_ERROR "CMAKE_HIP_PLATFORM must match HIP_PLATFORM=nvidia; use a fresh build directory")
  endif()
  set(CMAKE_HIP_PLATFORM nvidia CACHE STRING "HIP GPU platform")
  if(NOT CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "HIP compiler")
  endif()
  if(NOT CMAKE_HIP_HOST_COMPILER)
    set(CMAKE_HIP_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "HIP host compiler")
  endif()
  # CMake 3.28-3.31 also asks for a HIP root on NVIDIA. Honor an explicit
  # installation without requiring hipconfig to be available on PATH.
  if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
    foreach(_hip_root IN ITEMS "${HIP_PATH}" "$ENV{HIP_PATH}" "${ROCM_PATH}" "$ENV{ROCM_PATH}")
      if(EXISTS "${_hip_root}/include/hip/hip_runtime.h")
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_hip_root}" CACHE PATH "HIP compiler installation root")
        break()
      endif()
    endforeach()
  endif()
  set(CMAKE_HIP_ARCHITECTURES "${OFFLOAD_ARCH}")
  enable_language(HIP)
else()
  # Retain the established hipcc path for AMD builds.
  if(NOT CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "/usr/bin/c++")
    find_program(QUOP_HIPCC hipcc HINTS "${ROCM_PATH}/bin" "${HIP_PATH}/bin" REQUIRED)
    set(CMAKE_CXX_COMPILER "${QUOP_HIPCC}" CACHE FILEPATH "C++ compiler" FORCE)
  endif()
  # hipcc must know the target even during CMake's compiler checks on login nodes.
  foreach(_arch IN LISTS OFFLOAD_ARCH)
    if(NOT " ${CMAKE_CXX_FLAGS} " MATCHES " --offload-arch=${_arch} ")
      string(APPEND CMAKE_CXX_FLAGS " --offload-arch=${_arch}")
    endif()
  endforeach()
  enable_language(CXX)
endif()

find_package(hip REQUIRED CONFIG)

# An imported interface can also be used by try_compile. Do not rely on
# hip::host/device having populated usage requirements on NVIDIA HIP releases.
add_library(quop_gpu_runtime INTERFACE IMPORTED GLOBAL)
if(HIP_PLATFORM STREQUAL "nvidia")
  find_path(QUOP_HIP_INCLUDE_DIR hip/hip_runtime.h
    HINTS ${hip_INCLUDE_DIRS} "${HIP_PATH}" "${ROCM_PATH}" "${hip_DIR}/../../.."
          "${CMAKE_HIP_COMPILER_ROCM_ROOT}"
    ENV HIP_PATH ENV ROCM_PATH
    PATH_SUFFIXES include
    REQUIRED
  )
  target_include_directories(quop_gpu_runtime INTERFACE "${QUOP_HIP_INCLUDE_DIR}")
  target_compile_definitions(quop_gpu_runtime INTERFACE __HIP_PLATFORM_NVIDIA__)
  target_link_libraries(quop_gpu_runtime INTERFACE CUDA::cudart)
  set(_quop_probe_ext hip)
else()
  target_link_libraries(quop_gpu_runtime INTERFACE hip::host)
  set(_quop_probe_ext cpp)
endif()

file(WRITE "${CMAKE_BINARY_DIR}/quop_hip_test.${_quop_probe_ext}"
  "#include <hip/hip_runtime.h>\nint main() { hipDeviceProp_t p; return hipGetDeviceProperties(&p, 0); }\n")
try_compile(HIP_TEST "${CMAKE_BINARY_DIR}/quop-hip-check"
  SOURCES "${CMAKE_BINARY_DIR}/quop_hip_test.${_quop_probe_ext}"
  LINK_LIBRARIES quop_gpu_runtime
  OUTPUT_VARIABLE _quop_hip_output
)
if(NOT HIP_TEST)
  message(FATAL_ERROR "HIP compiler/runtime check failed:\n${_quop_hip_output}")
endif()

function(quop_configure_gpu_kernels target)
  target_link_libraries(${target} PRIVATE quop_gpu_runtime)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(HIP_PLATFORM STREQUAL "nvidia")
    get_target_property(_sources ${target} SOURCES)
    set_source_files_properties(${_sources} PROPERTIES LANGUAGE HIP)
    set_target_properties(${target} PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  else()
    target_link_libraries(${target} PRIVATE hip::device)
  endif()
endfunction()
