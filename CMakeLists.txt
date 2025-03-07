cmake_minimum_required(VERSION 3.18...3.25)

project(QuOp_MPI,
  VERSION 2.0.0
  DESCRIPTION "QuOp_MPI extension modules."
  LANGUAGES C Fortran
)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

set(MPI_BACKEND $ENV{MPI_BACKEND} CACHE STRING "Build mpi backend.")
set(GPU_AWARE_MPI $ENV{GPU_AWARE_MPI} CACHE STRING "Link against GPU-aware MPI.")
set(MPI_GTL_LIB_DIR $ENV{MPI_GTL_LIB_DIR} CACHE STRING "Link path for GPU-aware MPI.")
set(WAVEFRONT_BACKEND $ENV{WAVEFRONT_BACKEND} CACHE STRING "Build wavefront backend.")
set(CXX $ENV{CXX} CACHE STRING "For the wavefront backend CXX must point to the HIP compiler.")
set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "Path to ROCM installation.")
set(HIPFORT_PATH $ENV{HIPFORT_PATH} CACHE STRING "Path to hipfort installation.")    
set(HIPFORT_COMPILER $ENV{HIPFORT_COMPILER} CACHE STRING "Fortran MPI compiler-wrapper for hipfort compiler-wrapper, e.g.: ftn (Cray), mpifort (other).")
set(HIP_PLATFORM $ENV{HIP_PLATFORM} CACHE STRING "HIP platform, nvidia or amd.")
set(CUDA_PATH $ENV{CUDA_PATH} CACHE STRING "Path to CUDA installation.")
set(OFFLOAD_ARCH $ENV{OFFLOAD_ARCH} CACHE STRING "Target GPU architecture, e.g.: sm_86.")
set(SHAFFT_PATH $ENV{SHAFFT_PATH} CACHE STRING "Path to HIFFTND installation.")
set(BUILD $ENV{BUILD} CACHE STRING "Build type, RELEASE or DEBUG.")

set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT DEFINED ENV{MPI_BACKEND})
  set(MPI_BACKEND ON)
endif()

if(NOT DEFINED ENV{WAVEFRONT_BACKEND})
  set(WAVEFRONT_BACKEND OFF)
endif()

if(NOT DEFINED ENV{GPU_AWARE_MPI})
  set(GPU_AWARE_MPI OFF)
endif()

if(NOT DEFINED ENV{BUILD})
  set(BUILD RELEASE)
endif()

set(CMAKE_BUILD_TYPE ${BUILD})

include(CMakePrintHelpers)

find_package(MPI REQUIRED Fortran)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MPI_Fortran_COMPILE_FLAGS}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_Fortran_LINK_FLAGS}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MPI_Fortran_LINK_FLAGS}")

if(DEFINED ENV{CRAYPE_VERSION})
  message("Cray PE detected")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch -std=gnu")
endif()

if(${GPU_AWARE_MPI})
  link_directories(${MPI_GTL_LIB_DIR})
  link_libraries(-lmpi_gtl_hsa) 
  set(ENV{MPICH_GPU_SUPPORT_ENABLED} 1)
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

if(${WAVEFRONT_BACKEND})
  message("-- Wavefront backend selected")
  set(CMAKE_CXX_COMPILER_WORKS 1)
  enable_language(CXX)
  message(CHECK_START "Checking whether CXX compiler ${CMAKE_CXX_COMPILER} is configured for HIP.")
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/test.cpp
    "#include <hip/hip_runtime.h>
    int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    return 0;
  }")
  try_compile(
    HIP_TEST 
    ${CMAKE_BINARY_DIR} 
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/test.cpp 
    COMPILE_DEFINITIONS "-I${ROCM_PATH}/include"
    OUTPUT_VARIABLE OUTPUT 
    ) 
  if(NOT HIP_TEST)
    cmake_print_variables(OUTPUT)
    cmake_print_variables(CXX)
    cmake_print_variables(ROCM_PATH)
    cmake_print_variables(HIP_PLATFORM)
    if(HIP_PLATFORM STREQUAL nvidia)
      cmake_print_variables(CUDA_PATH)
    endif()
    message(FATAL_ERROR "HIP compiler ${CMAKE_CXX_COMPILER} is not configured correctly")
  endif()
  set(WAVEFRONT_CONFIGURED TRUE)
  if(NOT DEFINED ENV{HIPFORT_PATH})
    message(SEND_ERROR "HIPFORT_PATH environment variable not set")
    set(WAVEFRONT_CONFIGURED FALSE)
  endif()
  if(NOT DEFINED ENV{SHAFFT_PATH})
    message(SEND_ERROR "SHAFFT_PATH environment variable not set")
    set(WAVEFRONT_CONFIGURED FALSE)
  endif()
  if(NOT DEFINED ENV{OFFLOAD_ARCH})
      message(SEND_ERROR "TARGET_ARCH environment variable not set")
    set(WAVEFRONT_CONFIGURED FALSE)
  endif()
  if(NOT ${WAVEFRONT_CONFIGURED})
    message(FATAL_ERROR "\nWavefront backend not configured correctly")
  endif()
  set(HIPFORT_COMPILER ${CMAKE_Fortran_COMPILER})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FindPython3)
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter Development NumPy)

# Ensure scikit-build modules
if (NOT DEFINED SKBUILD)
  # If skbuild is not the driver; include its utilities in CMAKE_MODULE_PATH
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
    -c "import os, skbuild; print(os.path.dirname(skbuild.__file__))"
    OUTPUT_VARIABLE SKBLD_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  list(APPEND CMAKE_MODULE_PATH "${SKBLD_DIR}/resources/cmake")
  message(STATUS "Looking in ${SKBLD_DIR}/resources/cmake for CMake modules")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
endif()

# F2PY headers
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (CMAKE_BUILD_TYPE STREQUAL DEBUG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -Wall")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ggdb -O0 -Wall -fbounds-check")
elseif(CMAKE_BUILD_TYPE STREQUAL RELEASE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -march=native")
endif()

add_subdirectory(src)

message("-------------------------------------------------")
message("Build Configuration")
message("")
cmake_print_variables(BUILD)
cmake_print_variables(CMAKE_INSTALL_PREFIX)
cmake_print_variables(CMAKE_Fortran_COMPILER)
cmake_print_variables(Python3_EXECUTABLE)
cmake_print_variables(Python3_INCLUDE_DIRS)
cmake_print_variables(Python3_NumPy_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(SKBUILD)
cmake_print_variables(SKBLD_DIR)
cmake_print_variables(HDF5_Fortran_INCLUDE_DIRS)
cmake_print_variables(HDF5_Fortran_LIBRARIES)
cmake_print_variables(GPU_AWARE_MPI)
if(${GPU_AWARE_MPI})
  cmake_print_variables(MPI_GTL_LIB_DIR)
endif()
cmake_print_variables(MPI_BACKEND)
if(${MPI_BACKEND})
  cmake_print_variables(FFTW_INCLUDE_DIRS)
  cmake_print_variables(FFTW_DOUBLE_LIB)
  cmake_print_variables(FFTW_DOUBLE_MPI_LIB)
endif()
cmake_print_variables(WAVEFRONT_BACKEND)
if(${WAVEFRONT_BACKEND})
  cmake_print_variables(ROCM_PATH)
  cmake_print_variables(HIPFORT_PATH)
  cmake_print_variables(HIPFORT_COMPILER)
  cmake_print_variables(OFFLOAD_ARCH)
  cmake_print_variables(SHAFFT_PATH)
  if(HIP_PLATFORM STREQUAL nvidia)
    cmake_print_variables(CUDA_PATH)
  endif()
endif()
message("")
message("-------------------------------------------------")

set(LD_LIBRARY_PATH ${ROCM_PATH}/lib:${CUDA_PATH}/lib64:${SHAFFT_PATH}/lib:$LD_LIBRARY_PATH)
message("\nQuOp_MPI requires that the following environment variable be set:\n")
message("LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\n\n")
