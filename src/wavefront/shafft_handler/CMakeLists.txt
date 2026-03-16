# CMakeLists.txt for wavefront/shafft_handler module
#
# Unified FFT handler module using Fortran 2003 OOP features

# Base module (abstract type and interfaces)
add_library(shafft_handler_base OBJECT shafft_handler_base.f90)
compile_and_link_hipfort_mpi(shafft_handler_base)
target_include_directories(shafft_handler_base PRIVATE ${SHAFFT_INCLUDE_DIR})
target_link_libraries(shafft_handler_base PUBLIC  shafft::shafftf03 ${SHAFFT_LIBRARIES})

# 1D handler implementation
add_library(shafft_handler_1d OBJECT shafft_handler_1d.f90)
compile_and_link_hipfort_mpi(shafft_handler_1d)
target_include_directories(shafft_handler_1d PRIVATE ${SHAFFT_INCLUDE_DIR})
target_link_libraries(shafft_handler_1d PUBLIC shafft_handler_base shafft::shafftf03)
add_dependencies(shafft_handler_1d shafft_handler_base)

# ND handler implementation
add_library(shafft_comm_dims_utils OBJECT shafft_comm_dims_utils.f90)
compile_and_link_hipfort_mpi(shafft_comm_dims_utils)

add_library(shafft_handler_nd OBJECT shafft_handler_nd.f90)
compile_and_link_hipfort_mpi(shafft_handler_nd)
target_include_directories(shafft_handler_nd PRIVATE ${SHAFFT_INCLUDE_DIR})
target_link_libraries(shafft_handler_nd PUBLIC shafft_handler_base shafft_comm_dims_utils shafft::shafftf03)
add_dependencies(shafft_handler_nd shafft_handler_base shafft_comm_dims_utils)

# Public module (factory function)
add_library(shafft_handler OBJECT shafft_handler.f90)
compile_and_link_hipfort_mpi(shafft_handler)
target_include_directories(shafft_handler PRIVATE ${SHAFFT_INCLUDE_DIR})
target_link_libraries(shafft_handler PUBLIC shafft_handler_base shafft_handler_1d shafft_handler_nd shafft::shafftf03)
add_dependencies(shafft_handler shafft_handler_base shafft_handler_1d shafft_comm_dims_utils shafft_handler_nd)
