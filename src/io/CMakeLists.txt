include(FindHDF5)
set(HDF5_PREFER_PARALLEL TRUE)
# Check if building on Fedora.
if(DEFINED ENV{MPI_INCLUDE})
	set(HDF5_ROOT "$ENV{MPI_BIN};$ENV{MPI_INCLUDE};$ENV{MPI_LIB};$ENV{MPI_FORTRAN_MOD_DIR}")
endif()

find_package(HDF5 COMPONENTS C Fortran)

set(HDF5_ERROR_INSTRUCTIONS "\
\nIf parallel HDF5 is installed on your system, pass its library and include directories to CMake. 
For example, if building the QuOp_MPI extension modules via pip:

$:python3 -m pip install . -DHDF5_Fortran_LIBRARIES=<path to lib dir>\\ -DHDF5_Fortran_INCLUDE_DIR=<path to include dir>

Or, if calling CMake directly:

$:cmake -B build -S . -DHDF5_Fortran_LIBRARIES=<path to lib dir>\\ -DHDF5_Fortran_INCLUDE_DIR=<path to include dir>
\
")

if ( NOT HDF5_FOUND )
	message(FATAL_ERROR "HDF5 not found on system.${HDF5_ERROR_INSTRUCTIONS}")	
endif()
if ( NOT HDF5_IS_PARALLEL)
	message(FATAL_ERROR "Found HDF5 installation is not parallel.${HDF5_ERROR_INSTRUCTIONS}")
endif()

set(HDF5_Fortran_INCLUDE_DIRS ${HDF5_Fortran_INCLUDE_DIRS} CACHE STRING "")
set(HDF5_Fortran_LIBRARIES ${HDF5_Fortran_LIBRARIES} CACHE STRING "")


add_f2py_library(
	MODULE_NAME parallel_io 
	SRC ${CMAKE_CURRENT_SOURCE_DIR}/io.f90 
	INCLUDE_DIRS "${HDF5_Fortran_INCLUDE_DIRS}"
	LIBRARIES "${HDF5_Fortran_LIBRARIES}"
	)
