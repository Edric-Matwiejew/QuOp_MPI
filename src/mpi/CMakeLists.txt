configure_file(${CMAKE_SOURCE_DIR}/cmake/downloadFindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
endif()

set(_findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${_findFFTW_DIR}")

find_package(FFTW REQUIRED COMPONENTS DOUBLE_MPI_LIB)

set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIRS} CACHE STRING "")
set(FFTW_DOUBLE_LIB ${FFTW_DOUBLE_LIB} CACHE STRING "")
set(FFTW_DOUBLE_MPI_LIB ${FFTW_DOUBLE_MPI_LIB} CACHE STRING "")

add_library(mpi_context OBJECT mpi_context.f90)
target_link_libraries(mpi_context PUBLIC MPI::MPI_Fortran)
# mpi_context now uses comm_info_module (quop_mpi_layout_t)
add_dependencies(mpi_context comm_info_mod)

# When WAVEFRONT_BACKEND is ON, comm_info_mod references symbols from
# ci_gpu_topology, ci_communicators, and ci_partitions. All f2py targets
# that include comm_info_mod must also link these object libraries.
if(WAVEFRONT_BACKEND)
    set(_COMM_INFO_EXTRA ci_gpu_topology ci_communicators ci_partitions)
else()
    set(_COMM_INFO_EXTRA "")
endif()

set(_depends mpi_context comm_info_mod ${_COMM_INFO_EXTRA})
set(_definitions context=mpi_backend context_type=mpi_context)

add_f2py_library(
  MODULE_NAME mpi_context
  DEPENDS "${_depends}" 
  DEFINITIONS ${_definitions}
  SRC ${CONTEXT_WRAPPER}
  INSTALL_SUBDIR mpi
  )

add_library(mpi_circulant_operators OBJECT mpi_circulant_operators.f90)
target_link_libraries(mpi_circulant_operators PUBLIC MPI::MPI_Fortran)

# Export the common module directory so consumers can find mpi_backend.mod, etc.
set_target_properties(mpi_context PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_Fortran_MODULE_DIRECTORY}")

set_target_properties(mpi_circulant_operators PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_Fortran_MODULE_DIRECTORY}")

add_subdirectory(sparse)
add_subdirectory(diagonal)
add_subdirectory(circulant)
add_subdirectory(multivariable)
