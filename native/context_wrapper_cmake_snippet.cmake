# context_wrapper_cmake_snippet.cmake
#
# Drop-in CMake replacement for the add_f2py_library call that currently
# builds the mpi_context f2py module from context_wrapper.f90.
#
# Place this content inside native/mpi/CMakeLists.txt in place of (or
# alongside) the existing add_f2py_library(MODULE_NAME mpi_context ...) block.
#
# Prerequisites already expected to be set by the parent CMakeLists:
#   mpi_context         OBJECT library (mpi_context.f90)
#   comm_info_mod       OBJECT library
#   ${_COMM_INFO_EXTRA} optional wavefront comm_info objects
#   MPI::MPI_Fortran    imported target
#   Python3             found with Development.Module and NumPy components

# ----------------------------------------------------------------------------
# 1. bind(C) Fortran shim — compiled with the same preprocessor definitions
#    as context_wrapper.f90 so that 'context'/'context_type' resolve to the
#    MPI backend types.
# ----------------------------------------------------------------------------
add_library(mpi_context_wrapper_c OBJECT
    ${CMAKE_SOURCE_DIR}/native/context_wrapper_c.f90
)

target_compile_definitions(mpi_context_wrapper_c PRIVATE
    context=mpi_backend
    context_type=mpi_context
)

target_link_libraries(mpi_context_wrapper_c PUBLIC
    mpi_context
    comm_info_mod
    ${_COMM_INFO_EXTRA}
    MPI::MPI_Fortran
)

set_target_properties(mpi_context_wrapper_c PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

# ----------------------------------------------------------------------------
# 2. CPython extension — named 'context_wrapper' so that
#    backend.context.context_wrapper resolves correctly via the MPI __init__.py
#    module alias ("context" -> "mpi_context").
# ----------------------------------------------------------------------------
Python3_add_library(context_wrapper MODULE
    ${CMAKE_SOURCE_DIR}/native/context_wrapper_ext.c
)

target_include_directories(context_wrapper PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(context_wrapper PRIVATE
    mpi_context_wrapper_c
    mpi_context
    comm_info_mod
    ${_COMM_INFO_EXTRA}
    MPI::MPI_Fortran
)

install(
    TARGETS context_wrapper
    LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/quop_mpi/_lib/mpi"
)
