include(add_f2py_library.cmake)

set(PROPAGATOR_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/propagator_wrapper.f90)
set(CONTEXT_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/context_wrapper.f90)

add_subdirectory(io)
add_subdirectory(cartesian)
add_subdirectory(csr_generators)
add_subdirectory(sparse_vector)

if(${MPI_BACKEND})
    add_subdirectory(mpi)
endif()
