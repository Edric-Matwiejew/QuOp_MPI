# CMakeLists.txt for SparseExpm kernel unit tests
#
# Build instructions: cmake -S . -B build -DHIP_PLATFORM=amd -DOFFLOAD_ARCH=gfx1100
# NVIDIA: cmake -S . -B build -DHIP_PLATFORM=nvidia -DOFFLOAD_ARCH=90 -DHIP_PATH=/path/to/hip
# Then: cmake --build build
#
# To enable GPU-aware MPI (e.g., on Cray systems with GPU-aware transport): cmake .. -DCMAKE_CXX_COMPILER=hipcc
# -DGPU_AWARE_MPI=ON
#
# Run tests: ./test_vector_kernels ./test_reduction_kernels ./test_spmv_kernels ./test_chebyshev_kernels mpirun -np 1
# ./test_sparse_gpu  # Integration tests

cmake_minimum_required(VERSION 3.28)

project(SparseExpm_Tests LANGUAGES C Fortran)

# Add cmake module path from parent project
get_filename_component(QUOP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${QUOP_ROOT}/cmake")

set(HIP_PLATFORM "amd" CACHE STRING "GPU platform: amd or nvidia")
set(ROCM_PATH "" CACHE PATH "ROCm installation")
set(HIP_PATH "" CACHE PATH "HIP installation, including HIP-on-NVIDIA headers")
set(CUDA_PATH "" CACHE PATH "CUDA toolkit installation")
set(OFFLOAD_ARCH "" CACHE STRING "GPU architecture (gfx1100 for AMD, 90 for GH200)")
if(NOT OFFLOAD_ARCH)
  if(GPU_TARGETS)
    set(OFFLOAD_ARCH "${GPU_TARGETS}" CACHE STRING "GPU architecture" FORCE)
  elseif(HIP_PLATFORM STREQUAL "amd")
    set(OFFLOAD_ARCH gfx1100 CACHE STRING "GPU architecture" FORCE)
  endif()
endif()
set(GPU_TARGETS "${OFFLOAD_ARCH}" CACHE STRING "GPU architectures for hipfort")
set(HIPFORT_COMPILER "${CMAKE_Fortran_COMPILER}" CACHE FILEPATH "Fortran compiler for hipfort")

# Use parent project's FetchContent base dir to reuse hipfort build
set(FETCHCONTENT_BASE_DIR
    "${QUOP_ROOT}/.deps"
    CACHE PATH ""
)

# Configure the same GPU language/runtime as the main project.
include(QuOpGPU)

# Find MPI
find_package(MPI REQUIRED COMPONENTS Fortran)

# GPU-aware MPI option
option(GPU_AWARE_MPI "Enable GPU-aware MPI (direct device buffer communication)" OFF)

# Use the HipfortDependency module from parent project
include(HipfortDependency)
include(QuOpFortranPreprocess)

# Include the parent src directory for the kernel module
set(SPARSE_EXPM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Compile the HIP kernels into a library
add_library(
  sparse_expm_kernels SHARED
  ${SPARSE_EXPM_SRC_DIR}/kernels/hip_vector_kernels.cpp ${SPARSE_EXPM_SRC_DIR}/kernels/hip_spmv_kernels.cpp
  ${SPARSE_EXPM_SRC_DIR}/kernels/hip_chebyshev_kernels.cpp ${SPARSE_EXPM_SRC_DIR}/kernels/hip_reduction_kernels.cpp
)

target_include_directories(sparse_expm_kernels PRIVATE ${SPARSE_EXPM_SRC_DIR}/kernels)

quop_configure_gpu_kernels(sparse_expm_kernels)

# Compile the Fortran interface module
add_library(sparse_expm_fortran_interface ${SPARSE_EXPM_SRC_DIR}/hip_sparse_expm_kernels.f90)

target_link_libraries(sparse_expm_fortran_interface PUBLIC hipfort::hipfort)

# Compile the full Sparse module with HIP support This requires preprocessing with USE_HIP defined
set(_test_sparse_defs USE_HIP)
if(GPU_AWARE_MPI)
  list(APPEND _test_sparse_defs QUOP_GPU_AWARE_MPI)
endif()

quop_preprocess_fortran_source(
  _test_sparse_src
  NAME
  "test_sparse_module"
  SOURCE
  "${SPARSE_EXPM_SRC_DIR}/sparse.f90"
  DEFINITIONS
  ${_test_sparse_defs}
)

add_library(sparse_module "${_test_sparse_src}")

# Note: QUOP_GPU_AWARE_MPI is passed explicitly during preprocessing when enabled.

target_link_libraries(
  sparse_module PUBLIC sparse_expm_fortran_interface sparse_expm_kernels hipfort::hipfort quop_gpu_runtime MPI::MPI_Fortran
)

# Make sure the module files are found
target_include_directories(sparse_module PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Compile the Chebyshev module with HIP support
quop_preprocess_fortran_source(
  _test_chebyshev_src
  NAME
  "test_chebyshev_module"
  SOURCE
  "${SPARSE_EXPM_SRC_DIR}/chebyshev.f90"
  DEFINITIONS
  ${_test_sparse_defs}
)

add_library(chebyshev_module "${_test_chebyshev_src}")

target_link_libraries(
  chebyshev_module PUBLIC sparse_module sparse_expm_fortran_interface sparse_expm_kernels hipfort::hipfort quop_gpu_runtime
                          MPI::MPI_Fortran
)

target_include_directories(chebyshev_module PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(chebyshev_module sparse_module)

# Add dependency to ensure interface module is built first
add_dependencies(sparse_module sparse_expm_fortran_interface)

# Compile the test utilities module
add_library(test_utils src/test_hash_utils.f90)

# Function to create a kernel unit test executable
function(add_kernel_test TEST_NAME)
  add_executable(${TEST_NAME} src/${TEST_NAME}.f90)

  target_link_libraries(
    ${TEST_NAME} PRIVATE sparse_expm_fortran_interface sparse_expm_kernels test_utils hipfort::hipfort quop_gpu_runtime
  )

  # Make sure the module files are found
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  # Add dependency to ensure module is built first
  add_dependencies(${TEST_NAME} sparse_expm_fortran_interface)

  # Register as a CTest
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Function to create an MPI integration test executable
function(add_mpi_test TEST_NAME)
  add_executable(${TEST_NAME} src/${TEST_NAME}.f90)

  target_link_libraries(
    ${TEST_NAME} PRIVATE sparse_module sparse_expm_fortran_interface sparse_expm_kernels hipfort::hipfort quop_gpu_runtime
                         MPI::MPI_Fortran
  )

  # Make sure the module files are found
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  # Add dependency to ensure modules are built first
  add_dependencies(${TEST_NAME} sparse_module sparse_expm_fortran_interface)

  # Register as a CTest with MPI launcher
  add_test(NAME ${TEST_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:${TEST_NAME}>)

  # Also add multi-rank test if applicable
  add_test(NAME ${TEST_NAME}_mpi2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:${TEST_NAME}>)
endfunction()

# Function to create an MPI integration test that also uses the Chebyshev module
function(add_chebyshev_test TEST_NAME)
  add_executable(${TEST_NAME} src/${TEST_NAME}.f90)

  target_link_libraries(
    ${TEST_NAME}
    PRIVATE chebyshev_module
            sparse_module
            sparse_expm_fortran_interface
            sparse_expm_kernels
            hipfort::hipfort
            quop_gpu_runtime
            MPI::MPI_Fortran
  )

  # Make sure the module files are found
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  # Add dependency to ensure modules are built first
  add_dependencies(${TEST_NAME} chebyshev_module sparse_module sparse_expm_fortran_interface)

  # Register as a CTest with MPI launcher for 1, 2, and 4 ranks
  add_test(NAME ${TEST_NAME}_mpi1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:${TEST_NAME}>)
  add_test(NAME ${TEST_NAME}_mpi2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:${TEST_NAME}>)
  add_test(NAME ${TEST_NAME}_mpi4 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:${TEST_NAME}>)
endfunction()

# Enable testing
enable_testing()

# Add kernel unit test executables
add_kernel_test(test_vector_kernels)
add_kernel_test(test_reduction_kernels)
add_kernel_test(test_spmv_kernels)
add_kernel_test(test_chebyshev_kernels)

# Add integration tests
add_mpi_test(test_sparse_gpu)
add_chebyshev_test(test_chebyshev_gpu)

# Custom target to run all tests
add_custom_target(
  run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS test_vector_kernels test_reduction_kernels test_spmv_kernels test_chebyshev_kernels test_sparse_gpu
          test_chebyshev_gpu
  COMMENT "Running all kernel unit tests and integration tests"
)
