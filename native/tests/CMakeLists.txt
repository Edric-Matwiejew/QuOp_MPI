# native/tests/CMakeLists.txt CTest configuration for QuOp_MPI native (Fortran/C++) tests

# Find MPI (should already be found by parent, but ensure it's available)
find_package(MPI REQUIRED COMPONENTS Fortran)

# Helper function for MPI tests
function(add_mpi_test TEST_NAME EXECUTABLE NUM_PROCS)
  add_test(NAME ${TEST_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS} ${MPIEXEC_PREFLAGS}
                                     $<TARGET_FILE:${EXECUTABLE}> ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(${TEST_NAME} PROPERTIES PROCESSORS ${NUM_PROCS} TIMEOUT 60)
endfunction()

# Helper function for GPU tests (adds GPU label for filtering)
function(add_gpu_test TEST_NAME EXECUTABLE NUM_PROCS)
  add_mpi_test(${TEST_NAME} ${EXECUTABLE} ${NUM_PROCS})
  set_tests_properties(${TEST_NAME} PROPERTIES LABELS "GPU")
endfunction()

# Include backend-specific tests based on build configuration
message(STATUS "Building MPI backend tests")
add_subdirectory(mpi)

if(${WAVEFRONT_BACKEND})
  message(STATUS "Building wavefront backend tests")
  add_subdirectory(wavefront)
endif()
