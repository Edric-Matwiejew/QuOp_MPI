# src_tests/wavefront/CMakeLists.txt
# CTest configuration for wavefront backend tests
#
# These tests validate the wavefront backend components:
# - Communicator hierarchy (NODECOMM, DEVCOMM, DEVCOMM_NODE)
# - Partition calculations for data distribution
# - Array transfer patterns (Alltoallv)

find_library(NUMA_LIBRARY numa REQUIRED)

# ============================================================================
# Platform-specific configuration
# ============================================================================
set(ROCM_INCLUDE_DIR "${ROCM_PATH}/include")
set(ROCM_LIB_DIR "${ROCM_PATH}/lib")

# Wavefront module directory (where .mod files are generated)
set(WAVEFRONT_MODULE_DIR "${CMAKE_BINARY_DIR}/modules")

# ============================================================================
# Create a static library from the wavefront OBJECT libraries for testing
# This bundles all wavefront objects into a single linkable library
# ============================================================================
add_library(wavefront_test_lib STATIC
    $<TARGET_OBJECTS:wf_gpu_topology>
    $<TARGET_OBJECTS:wf_gpu_transfer>
    $<TARGET_OBJECTS:wf_communicators>
    $<TARGET_OBJECTS:wf_partitions>
    $<TARGET_OBJECTS:wf_transfer>
)

target_include_directories(wavefront_test_lib PUBLIC
    ${ROCM_INCLUDE_DIR}
    ${WAVEFRONT_MODULE_DIR}
)

target_link_directories(wavefront_test_lib PUBLIC
    ${ROCM_LIB_DIR}
)

# Link using modern CMake target from HipfortDependency.cmake
if(HIP_PLATFORM STREQUAL nvidia)
    target_link_libraries(wavefront_test_lib PUBLIC
        ${CUDA_PATH}/lib64/libcudart.so
        hipfort::hipfort
        MPI::MPI_Fortran
        ${NUMA_LIBRARY}
    )
else()
    target_link_libraries(wavefront_test_lib PUBLIC
        ${ROCM_PATH}/lib/libamdhip64.so
        hipfort::hipfort
        MPI::MPI_Fortran
        ${NUMA_LIBRARY}
    )
endif()

if(TARGET CrayGTL::CrayGTL)
    target_link_libraries(wavefront_test_lib PUBLIC CrayGTL::CrayGTL)
endif()

add_dependencies(wavefront_test_lib 
    wf_gpu_topology
    wf_gpu_transfer
    wf_communicators 
    wf_partitions 
    wf_transfer
)

# ============================================================================
# Test 1: Communicator Creation (Unit Test)
# 
# Tests the fundamental communicator hierarchy:
# - create_NODECOMM: splits COMM_WORLD by shared memory (node)
# - get_GPUs_per_node: detects available GPUs
# - create_devcomm: creates GPU-assigned process communicator
# - create_DEVCOMM_NODE: node-local GPU communicator
#
# This is a UNIT TEST - tests individual module functions in isolation
# ============================================================================
add_executable(test_wavefront_comms
    test_wavefront_comms.f90
)

# Use Fortran linker (gfortran) instead of hipcc to avoid link order issues
set_target_properties(test_wavefront_comms PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(test_wavefront_comms PRIVATE
    ${WAVEFRONT_MODULE_DIR}
)

target_link_libraries(test_wavefront_comms PRIVATE
    wavefront_test_lib
)

# MPI test configurations - different process counts test different code paths
add_gpu_test(wavefront_comms_1proc test_wavefront_comms 1)
add_gpu_test(wavefront_comms_2proc test_wavefront_comms 2)
add_gpu_test(wavefront_comms_4proc test_wavefront_comms 4)
add_gpu_test(wavefront_comms_8proc test_wavefront_comms 8)

# ============================================================================
# Test 1b: GPU Topology Detection (Unit Test)
#
# Tests the gpu_topology module for detecting and configuring GPU assignment:
# - init_gpu_topology - topology detection via PCI bus IDs
# - Environment variable handling (QUOP_RANKS_PER_GPU, QUOP_GPU_BINDING_MODE)
# - DEVCOMM membership determination
# - hipSetDevice with assigned_device_id
#
# This is a UNIT TEST - tests the topology module in isolation
# Run with different environment variables to test configurations:
#   QUOP_RANKS_PER_GPU=2 - test multi-rank-per-GPU
#   QUOP_GPU_BINDING_MODE=sequential - force sequential assignment
#   QUOP_DEBUG_COMMUNICATORS=1 - enable debug output
# ============================================================================
add_executable(test_gpu_topology
    test_gpu_topology.f90
)

# Use Fortran linker (gfortran) instead of hipcc to avoid link order issues
set_target_properties(test_gpu_topology PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(test_gpu_topology PRIVATE
    ${WAVEFRONT_MODULE_DIR}
)

target_link_libraries(test_gpu_topology PRIVATE
    wavefront_test_lib
)

# ============================================================================
# Demo: NUMA-aware GPU binding fallback
#
# Standalone demo that shows how a rank can combine its CPU affinity mask with
# GPU PCI/NUMA information from sysfs to choose a better local GPU than the
# current sequential fallback when no external binding is present.
# ============================================================================
add_executable(numa_demo
    numa_demo.f90
)

set_target_properties(numa_demo PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(numa_demo PRIVATE
    ${WAVEFRONT_MODULE_DIR}
)

target_link_libraries(numa_demo PRIVATE
    wavefront_test_lib
)

# Standard topology tests with different process counts
add_gpu_test(gpu_topology_1proc test_gpu_topology 1)
add_gpu_test(gpu_topology_2proc test_gpu_topology 2)
add_gpu_test(gpu_topology_4proc test_gpu_topology 4)
add_gpu_test(gpu_topology_8proc test_gpu_topology 8)

# ============================================================================
# Test 1c: GPU Transfer Operations (Unit Test)
#
# Tests the gpu_transfer module for abstracted GPU memory transfers:
# - gpu_allgatherv_dtoh: Device to Host gather operation
# - gpu_allscatterv_htod: Host to Device scatter operation
# - Host-staged fallback path (uses regular MPI)
# - GPU-aware MPI path (when compiled with QUOP_GPU_AWARE_MPI)
# - Balanced and unbalanced vector partitionings
#
# This is a UNIT TEST - tests gpu_transfer functions in isolation
# Run with different QUOP_RANKS_PER_GPU to test different configurations:
#   Default:     QUOP_RANKS_PER_GPU=1 mpirun -n 4 ./test_gpu_transfer
#   Multi-rank:  QUOP_RANKS_PER_GPU=2 mpirun -n 4 ./test_gpu_transfer
# ============================================================================
add_executable(test_gpu_transfer
    test_gpu_transfer.f90
)

set_target_properties(test_gpu_transfer PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(test_gpu_transfer PRIVATE
    ${WAVEFRONT_MODULE_DIR}
)

target_link_libraries(test_gpu_transfer PRIVATE
    wavefront_test_lib
    comm_info_mod
    MPI::MPI_CXX
)

# Standard tests with default settings
add_gpu_test(gpu_transfer_2proc test_gpu_transfer 2)
add_gpu_test(gpu_transfer_4proc test_gpu_transfer 4)
add_gpu_test(gpu_transfer_8proc test_gpu_transfer 8)

# Tests with multi-rank-per-GPU (QUOP_RANKS_PER_GPU=2)
# These test the host-staged fallback with shared GPUs
add_test(
    NAME gpu_transfer_staged_2proc
    COMMAND ${CMAKE_COMMAND} -E env QUOP_RANKS_PER_GPU=2
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gpu_transfer> ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(gpu_transfer_staged_2proc PROPERTIES
    PROCESSORS 2
    TIMEOUT 60
    LABELS "GPU"
)

add_test(
    NAME gpu_transfer_staged_4proc
    COMMAND ${CMAKE_COMMAND} -E env QUOP_RANKS_PER_GPU=2
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gpu_transfer> ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(gpu_transfer_staged_4proc PROPERTIES
    PROCESSORS 4
    TIMEOUT 60
    LABELS "GPU"
)

add_test(
    NAME gpu_transfer_staged_8proc
    COMMAND ${CMAKE_COMMAND} -E env QUOP_RANKS_PER_GPU=4
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gpu_transfer> ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(gpu_transfer_staged_8proc PROPERTIES
    PROCESSORS 8
    TIMEOUT 60
    LABELS "GPU"
)

# ============================================================================
# Test 3: Array Transfer Patterns (Integration Test)
#
# Tests the complete data distribution workflow:
# - Partition calculation via quop_mpi_layout_t
# - Distribution arrays via MPI_Allgather
# - Overlap-based Alltoallv scatter/gather operations
#
# This is an INTEGRATION TEST - tests multiple modules working together
# to perform the complete host<->device data distribution pattern
# ============================================================================
add_executable(test_array_transfer
    test_array_transfer.f90
)

# Use Fortran linker (gfortran) instead of hipcc to avoid link order issues
set_target_properties(test_array_transfer PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(test_array_transfer PRIVATE
    ${WAVEFRONT_MODULE_DIR}
)

target_link_libraries(test_array_transfer PRIVATE
    wavefront_test_lib
    comm_info_mod
)

# Requires multiple processes to test the distribution pattern
add_gpu_test(array_transfer_2proc test_array_transfer 2)
add_gpu_test(array_transfer_4proc test_array_transfer 4)
add_gpu_test(array_transfer_8proc test_array_transfer 8)

# ============================================================================
# Test 4: Wavefront Context Integration Tests
#
# Tests the complete wavefront_context functionality via context_wrapper:
# - Context setup/destroy lifecycle
# - State vector round-trip (set_state -> get_state)
# - Observables round-trip (set_observables -> get_observables)
# - State norm calculation
# - Expectation value calculation
#
# This is an INTEGRATION TEST - tests the full context through the wrapper API
# Tests run with 1, 2, and 4 processes to validate single/multi-GPU paths
# ============================================================================

# Preprocess the context wrapper with wavefront definitions
set(WAVEFRONT_CONTEXT_WRAPPER_SRC "${CMAKE_SOURCE_DIR}/src/context_wrapper.f90")
set(WAVEFRONT_CONTEXT_WRAPPER_PREPROCESSED "${CMAKE_CURRENT_BINARY_DIR}/preprocessed_wavefront_context_wrapper.F90")

add_custom_command(
    OUTPUT "${WAVEFRONT_CONTEXT_WRAPPER_PREPROCESSED}"
    COMMAND ${CMAKE_Fortran_COMPILER}
        -cpp -E
        -Dcontext=wavefront
        -Dcontext_type=wavefront_context
        "${WAVEFRONT_CONTEXT_WRAPPER_SRC}"
        -o "${WAVEFRONT_CONTEXT_WRAPPER_PREPROCESSED}"
    DEPENDS "${WAVEFRONT_CONTEXT_WRAPPER_SRC}"
    COMMENT "Preprocessing context_wrapper.f90 for wavefront tests"
)

add_custom_target(generate_test_context_wrapper
    DEPENDS "${WAVEFRONT_CONTEXT_WRAPPER_PREPROCESSED}"
)

# Create a test library that includes the wavefront context and wrapper
add_library(wavefront_context_test_lib STATIC
    $<TARGET_OBJECTS:wavefront_context>
    $<TARGET_OBJECTS:wf_communicators>
    $<TARGET_OBJECTS:wf_partitions>
    $<TARGET_OBJECTS:wf_transfer>
    $<TARGET_OBJECTS:wf_gpu_topology>
    $<TARGET_OBJECTS:wf_gpu_transfer>
    $<TARGET_OBJECTS:hip_kernels>
    "${WAVEFRONT_CONTEXT_WRAPPER_PREPROCESSED}"
)

add_dependencies(wavefront_context_test_lib
    generate_test_context_wrapper
    wavefront_context
    wf_communicators
    wf_partitions
    wf_transfer
    wf_gpu_topology
    wf_gpu_transfer
    hip_kernels
)

target_include_directories(wavefront_context_test_lib PUBLIC
    ${ROCM_INCLUDE_DIR}
    ${WAVEFRONT_MODULE_DIR}
)

target_link_directories(wavefront_context_test_lib PUBLIC
    ${ROCM_LIB_DIR}
)

if(HIP_PLATFORM STREQUAL nvidia)
    target_link_libraries(wavefront_context_test_lib PUBLIC
        ${CUDA_PATH}/lib64/libcudart.so
        hipfort::hipfort
        MPI::MPI_Fortran
        MPI::MPI_CXX
        ${NUMA_LIBRARY}
    )
else()
    target_link_libraries(wavefront_context_test_lib PUBLIC
        ${ROCM_PATH}/lib/libamdhip64.so
        hipfort::hipfort
        MPI::MPI_Fortran
        MPI::MPI_CXX
        ${NUMA_LIBRARY}
    )
endif()

if(TARGET CrayGTL::CrayGTL)
    target_link_libraries(wavefront_context_test_lib PUBLIC CrayGTL::CrayGTL)
endif()

# Build the test executable
add_executable(test_wavefront_context
    test_wavefront_context.f90
)

set_target_properties(test_wavefront_context PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(test_wavefront_context PRIVATE
    ${WAVEFRONT_MODULE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(test_wavefront_context PRIVATE
    wavefront_context_test_lib
    comm_info_mod
    ci_gpu_topology
    ci_communicators
)

# Run tests with 1, 2, and 4 processes
# Set QUOP_RANKS_PER_GPU to match NUM_PROCS so all ranks become GPU ranks
# on single-GPU systems.
add_gpu_test(wavefront_context_1proc test_wavefront_context 1)
add_gpu_test(wavefront_context_2proc test_wavefront_context 2)
set_tests_properties(wavefront_context_2proc PROPERTIES
    ENVIRONMENT "QUOP_RANKS_PER_GPU=2"
)
add_gpu_test(wavefront_context_4proc test_wavefront_context 4)
set_tests_properties(wavefront_context_4proc PROPERTIES
    ENVIRONMENT "QUOP_RANKS_PER_GPU=4"
)
