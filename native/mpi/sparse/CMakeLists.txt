# Source files in sparse_propagators directory
set(SPARSE_PROPAGATORS_SRC ${CMAKE_SOURCE_DIR}/native/sparse_propagators/src)
set(_sparse_pp_defs "")
if(GPU_AWARE_MPI)
  list(APPEND _sparse_pp_defs QUOP_GPU_AWARE_MPI)
endif()

quop_preprocess_fortran_source(
  _mpi_sparse_src
  NAME
  "mpi_sparse_source"
  SOURCE
  "${SPARSE_PROPAGATORS_SRC}/sparse.f90"
  DEFINITIONS
  ${_sparse_pp_defs}
)

add_library(sparse OBJECT "${_mpi_sparse_src}")
target_link_libraries(sparse PUBLIC MPI::MPI_Fortran)

add_library(one_norms OBJECT ${SPARSE_PROPAGATORS_SRC}/one_norms.f90)
target_link_libraries(one_norms PUBLIC MPI::MPI_Fortran)
add_dependencies(one_norms sparse)

add_library(expm OBJECT ${SPARSE_PROPAGATORS_SRC}/expm.f90)
target_link_libraries(expm PUBLIC MPI::MPI_Fortran)
add_dependencies(expm one_norms sparse)

quop_preprocess_fortran_source(
  _mpi_chebyshev_src
  NAME
  "mpi_chebyshev_source"
  SOURCE
  "${SPARSE_PROPAGATORS_SRC}/chebyshev.f90"
  DEFINITIONS
  ${_sparse_pp_defs}
)

add_library(chebyshev OBJECT "${_mpi_chebyshev_src}")
target_link_libraries(chebyshev PUBLIC MPI::MPI_Fortran)
add_dependencies(chebyshev sparse)
target_include_directories(chebyshev PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY}/Sparse)

add_library(mpi_sparse OBJECT mpi_sparse.f90)
target_link_libraries(mpi_sparse PUBLIC MPI::MPI_Fortran)
add_dependencies(mpi_sparse expm chebyshev one_norms sparse comm_info_mod)
target_include_directories(
  mpi_sparse PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY}/Sparse ${CMAKE_Fortran_MODULE_DIRECTORY}/One_Norms
                    ${CMAKE_Fortran_MODULE_DIRECTORY}/Chebyshev
)

set(_depends
    mpi_sparse
    expm
    chebyshev
    one_norms
    sparse
    mpi_context
    comm_info_mod
    ${_COMM_INFO_EXTRA}
)
set(_definitions context=mpi_backend context_type=mpi_context propagator=mpi_sparse propagator_type=sparse_propagator
                 propagator_wrapper=sparse_propagator_wrapper
)

add_f2py_library(
  MODULE_NAME
  mpi_sparse_propagator
  DEPENDS
  "${_depends}"
  DEFINITIONS
  "${_definitions}"
  SRC
  ${PROPAGATOR_WRAPPER}
  INSTALL_SUBDIR
  mpi
)
