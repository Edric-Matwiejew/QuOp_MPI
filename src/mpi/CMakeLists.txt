configure_file(downloadFindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
else()
    message("CMake step for findFFTW completed (${result}).")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
endif()

set(findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${findFFTW_DIR}")

find_package(FFTW REQUIRED COMPONENTS DOUBLE_MPI_LIB)

set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIRS} CACHE STRING "")
set(FFTW_DOUBLE_LIB ${FFTW_DOUBLE_LIB} CACHE STRING "")
set(FFTW_DOUBLE_MPI_LIB ${FFTW_FOUND} CACHE STRING "")

add_library(mpi_context OBJECT mpi_context.f90)


set(depends mpi_context)
set(definitions context=mpi_backend context_type=mpi_context)

add_f2py_library(
	MODULE_NAME mpi_context
	DEPENDS "${depends}" 
	DEFINITIONS "${definitions}" 
	SRC ${CONTEXT_WRAPPER}
	INSTALL_SUBDIR mpi
	)

add_library(mpi_circulant_operators OBJECT mpi_circulant_operators.f90)

add_subdirectory(sparse)
add_subdirectory(diagonal)
add_subdirectory(circulant)
add_subdirectory(multivariable)

