cmake_minimum_required(VERSION 3.25...4.0)

project(
  QuOp_MPI
  VERSION 1.6.0
  DESCRIPTION "QuOp_MPI extension modules."
  LANGUAGES C Fortran
)

# Add cmake/ directory to module path for HipfortDependency and other modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Store FetchContent dependencies outside build/ so they persist across clean builds This avoids re-downloading and
# re-building dependencies when doing `rm -rf build`
if(NOT FETCHCONTENT_BASE_DIR)
  set(FETCHCONTENT_BASE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/.deps"
      CACHE PATH "Directory for FetchContent downloads and builds (persists across clean builds)"
  )
endif()

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

# =============================================================================
# User-configurable options (visible in ccmake/cmake-gui)
# =============================================================================
# These are the primary configuration variables. They can be set via: 1. Command-line: cmake -DWAVEFRONT_BACKEND=ON ...
# 2. Build frontend passthrough (e.g., CMAKE_ARGS from setup/install.sh) 3. Interactive tools: ccmake or cmake-gui

# --- Backend Selection ---
# MPI backend modules are always built; wavefront is optional.
option(WAVEFRONT_BACKEND "Build the wavefront (GPU) backend for HIP/ROCm" OFF)
option(GPU_AWARE_MPI "Enable GPU-aware MPI (requires compatible MPI library)" OFF)

# --- Build Configuration ---
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# --- GPU/HIP Configuration (for WAVEFRONT_BACKEND) ---
set(ROCM_PATH
    ""
    CACHE PATH "Path to ROCm installation (e.g., /opt/rocm)"
)
set(HIP_PLATFORM
    "amd"
    CACHE STRING "HIP platform: 'amd' for ROCm, 'nvidia' for CUDA"
)
set_property(CACHE HIP_PLATFORM PROPERTY STRINGS "amd" "nvidia")
set(OFFLOAD_ARCH
    ""
    CACHE STRING "Target GPU architecture (e.g., gfx90a for MI250X, sm_80 for A100)"
)
set(GPU_TARGETS
    ""
    CACHE STRING "GPU architectures for hipfort (defaults to OFFLOAD_ARCH)"
)
set(CUDA_PATH
    ""
    CACHE PATH "Path to CUDA installation (only for HIP_PLATFORM=nvidia)"
)

# --- Optional Components (for WAVEFRONT_BACKEND) ---
option(WITH_SHAFFT "Build with SHAFFT support for multivariable operations" OFF)

# --- External Dependency Paths ---
set(SHAFFT_PATH
    ""
    CACHE PATH "Path to SHAFFT installation"
)
set(HIPFORT_PATH
    ""
    CACHE PATH "Path to hipfort installation (auto-fetched if not found)"
)
set(HIPFORT_COMPILER
    ""
    CACHE STRING "Fortran compiler for hipfort build (e.g., ftn for Cray, mpifort for others)"
)
set(HIPFORT_VERIFY_MOD_COMPAT
    ON
    CACHE BOOL "Verify hipfort .mod compatibility with current compiler"
)

# =============================================================================
# Apply defaults
# =============================================================================

# Set sensible defaults for paths if not specified
if(NOT ROCM_PATH)
  if(EXISTS "/opt/rocm")
    set(ROCM_PATH
        "/opt/rocm"
        CACHE PATH "" FORCE
    )
  endif()
endif()

# Default hipfort hints to the cached FetchContent install so we avoid the system hipfort unless the user overrides.
# These are only hints; the dependency module still auto-fetches if nothing usable is found.
if(NOT hipfort_DIR)
  set(hipfort_DIR
      "${FETCHCONTENT_BASE_DIR}/hipfort-install/lib/cmake/hipfort"
      CACHE PATH "Path to hipfort-config.cmake (defaults to cached .deps install)" FORCE
  )
endif()
if(NOT HIPFORT_PATH)
  set(HIPFORT_PATH
      "${FETCHCONTENT_BASE_DIR}/hipfort-install"
      CACHE PATH "Path to hipfort installation (auto-fetched if not found)" FORCE
  )
endif()
if(NOT HIPFORT_COMPILER)
  set(HIPFORT_COMPILER
      "${CMAKE_Fortran_COMPILER}"
      CACHE STRING "Fortran compiler for hipfort build (e.g., ftn for Cray, mpifort for others)" FORCE
  )
endif()

# WITH_SHAFFT: default to ON when WAVEFRONT_BACKEND is ON
if(WAVEFRONT_BACKEND)
  if(NOT WITH_SHAFFT)
    set(WITH_SHAFFT
        ON
        CACHE BOOL "" FORCE
    )
  endif()
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

include(CMakePrintHelpers)

find_package(MPI REQUIRED COMPONENTS Fortran)
message(STATUS "Found MPI: ${MPI_Fortran_COMPILER}")

if(DEFINED ENV{CRAYPE_VERSION})
  message(STATUS "Cray PE detected")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch -std=gnu")
endif()

if(GPU_AWARE_MPI)
  if(DEFINED ENV{CRAYPE_VERSION})
    find_package(CrayGTL)
    if(CrayGTL_FOUND)
      message(STATUS "Found Cray GTL: ${CrayGTL_LIBRARIES}")
    else()
      message(WARNING "GPU-aware MPI requested but Cray GTL not found")
    endif()
  endif()
  # Intentionally global: every Fortran/C++ translation unit that touches MPI buffers must see this define, so
  # target-scoping is not practical here.
  add_compile_definitions(QUOP_GPU_AWARE_MPI)
  message(STATUS "GPU-aware MPI enabled")
  message(STATUS "NOTE: Set MPICH_GPU_SUPPORT_ENABLED=1 in your runtime environment for Cray MPICH")
else()
  message(STATUS "GPU-aware MPI disabled")
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

if(WAVEFRONT_BACKEND)
  message(STATUS "Configuring wavefront (GPU) backend")

  if(HIP_PLATFORM STREQUAL "nvidia")
    set(HIP_PLATFORM_DEFINE "-D__HIP_PLATFORM_NVIDIA__")
  else()
    set(HIP_PLATFORM_DEFINE "-D__HIP_PLATFORM_AMD__")
  endif()

  if(NOT CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "/usr/bin/c++")
    if(EXISTS "${ROCM_PATH}/bin/hipcc")
      set(CMAKE_CXX_COMPILER
          "${ROCM_PATH}/bin/hipcc"
          CACHE FILEPATH "C++ compiler" FORCE
      )
    else()
      message(FATAL_ERROR "HIP compiler not found at ${ROCM_PATH}/bin/hipcc")
    endif()
  endif()

  set(CMAKE_CXX_COMPILER_WORKS 1)
  enable_language(CXX)

  message(CHECK_START "Checking HIP compiler")
  file(
    WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/test.cpp
    "#include <hip/hip_runtime.h>
    int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    return 0;
  }"
  )
  try_compile(
    HIP_TEST ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/test.cpp
    CMAKE_FLAGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    COMPILE_DEFINITIONS "-I${ROCM_PATH}/include" "${HIP_PLATFORM_DEFINE}"
    LINK_LIBRARIES "-L${ROCM_PATH}/lib" "-lamdhip64"
    OUTPUT_VARIABLE OUTPUT
  )
  if(NOT HIP_TEST)
    cmake_print_variables(OUTPUT)
    cmake_print_variables(CXX)
    cmake_print_variables(ROCM_PATH)
    cmake_print_variables(HIP_PLATFORM)
    if(HIP_PLATFORM STREQUAL "nvidia")
      cmake_print_variables(CUDA_PATH)
    endif()
    message(FATAL_ERROR "HIP compiler ${CMAKE_CXX_COMPILER} is not configured correctly")
  endif()
  set(WAVEFRONT_CONFIGURED TRUE)

  # Find HIP runtime - provides hip::device target for linking
  find_package(hip REQUIRED CONFIG PATHS "${ROCM_PATH}")
  if(NOT TARGET hip::device)
    message(FATAL_ERROR "hip::device target not found - check your ROCm/HIP installation")
  endif()
  message(STATUS "Found HIP: ${hip_DIR}")

  # Use HipfortDependency.cmake to find or fetch hipfort automatically This will create the hipfort::hipfort target and
  # set HIPFORT_PROVIDER
  include(HipfortDependency)

  # Check that hipfort was configured successfully
  if(NOT TARGET hipfort::hipfort)
    message(WARNING "hipfort::hipfort target not available - hipfort configuration failed")
    set(WAVEFRONT_CONFIGURED FALSE)
  endif()
  if(WITH_SHAFFT)
    # Use SHAFFTDependency.cmake to find or fetch SHAFFT automatically. This will create shafft::shafftc++,
    # shafft::shafftc, shafft::shafftf03 targets and set SHAFFT_PROVIDER, SHAFFT_VERSION, SHAFFT_INCLUDE_DIR.
    include(SHAFFTDependency)

    if(NOT TARGET shafft::shafftf03 OR NOT TARGET shafft::shafftc++)
      message(WARNING "SHAFFT targets not available - SHAFFT configuration failed")
      set(WAVEFRONT_CONFIGURED FALSE)
    endif()
  endif()

  if(NOT OFFLOAD_ARCH)
    message(WARNING "OFFLOAD_ARCH not set (use -DOFFLOAD_ARCH=..., e.g.: gfx90a)")
    set(WAVEFRONT_CONFIGURED FALSE)
  endif()
  if(NOT WAVEFRONT_CONFIGURED)
    message(FATAL_ERROR "Wavefront backend configuration failed")
  endif()

  # Platform-specific hipfort target - set once, used by all subdirectories.
  if(HIP_PLATFORM STREQUAL "nvidia")
    set(QUOP_HIPFORT_TARGET hipfort::hipfort-nvptx)
  else()
    set(QUOP_HIPFORT_TARGET hipfort::hipfort-amdgcn)
  endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development NumPy)

# F2PY headers
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -Wall")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ggdb -O0 -Wall -fbounds-check")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -march=native -fopenmp")
endif()

add_subdirectory(src)

option(BUILD_TESTING "Build native tests" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(src_tests)
endif()

message("-------------------------------------------------")
message("Build Configuration")
message("")
cmake_print_variables(CMAKE_BUILD_TYPE)
cmake_print_variables(CMAKE_INSTALL_PREFIX)
cmake_print_variables(CMAKE_Fortran_COMPILER)
cmake_print_variables(Python3_EXECUTABLE)
cmake_print_variables(Python3_INCLUDE_DIRS)
cmake_print_variables(Python3_NumPy_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(SKBUILD)
cmake_print_variables(HDF5_Fortran_INCLUDE_DIRS)
cmake_print_variables(HDF5_Fortran_LIBRARIES)
cmake_print_variables(GPU_AWARE_MPI)
if(GPU_AWARE_MPI)
  cmake_print_variables(MPI_GTL_LIB_DIR)
endif()
cmake_print_variables(FFTW_INCLUDE_DIRS)
cmake_print_variables(FFTW_DOUBLE_LIB)
cmake_print_variables(FFTW_DOUBLE_MPI_LIB)
cmake_print_variables(WAVEFRONT_BACKEND)
if(WAVEFRONT_BACKEND)
  cmake_print_variables(HIP_PLATFORM)
  cmake_print_variables(CMAKE_CXX_COMPILER)
  cmake_print_variables(ROCM_PATH)
  cmake_print_variables(HIPFORT_PROVIDER)
  cmake_print_variables(HIPFORT_VERSION)
  cmake_print_variables(HIPFORT_COMPILER)
  cmake_print_variables(OFFLOAD_ARCH)
  cmake_print_variables(WITH_SHAFFT)
  if(WITH_SHAFFT)
    cmake_print_variables(SHAFFT_PROVIDER)
    cmake_print_variables(SHAFFT_VERSION)
    cmake_print_variables(SHAFFT_PATH)
  endif()
  if(HIP_PLATFORM STREQUAL nvidia)
    cmake_print_variables(CUDA_PATH)
  endif()
endif()
cmake_print_variables(BUILD_TESTING)
message("")
message("-------------------------------------------------")

# Build LD_LIBRARY_PATH with only the relevant paths
# Note: Linux wheel builds repair SHAFFT into the final artifact via auditwheel,
# so it does not need to appear here at runtime.
set(_ld_paths "")
if(ROCM_PATH)
  list(APPEND _ld_paths "${ROCM_PATH}/lib")
endif()
if(HIP_PLATFORM STREQUAL nvidia AND CUDA_PATH)
  list(APPEND _ld_paths "${CUDA_PATH}/lib64")
endif()
string(REPLACE ";" ":" LD_LIBRARY_PATH "${_ld_paths}")
message("\nSet LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:\$LD_LIBRARY_PATH\n")
