include(QuOpFortranPreprocess)
include(QuOpF2pyLibrary)

# =============================================================================
# Fortran module dependencies
# =============================================================================
# We now rely on explicit target dependencies instead of serializing all Fortran compilation. Targets that produce
# modules (.mod) must be depended on by any target that USEs those modules. This allows safe parallel builds with both
# Ninja and Make.

set(PROPAGATOR_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/propagator_wrapper.f90)
set(CONTEXT_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/context_wrapper.f90)

add_subdirectory(comm_info)
add_subdirectory(io)
add_subdirectory(cartesian)
add_subdirectory(csr_generators)
add_subdirectory(sparse_vector)
add_subdirectory(transverse_field)
add_subdirectory(mpi)

if(${WAVEFRONT_BACKEND})
  add_subdirectory(wavefront)
else()
  # scikit-build-core can be asked to build an explicit target list from pyproject.toml. Provide no-op placeholders for
  # optional wavefront targets so MPI-only builds can reuse the same build.targets list.
  add_custom_target(wavefront_circulant_propagator_f2py)
  add_custom_target(wavefront_context_f2py)
  add_custom_target(wavefront_diagonal_propagator_f2py)
  add_custom_target(wavefront_composite_propagator_f2py)
  add_custom_target(wavefront_momentum_propagator_f2py)
  add_custom_target(wavefront_sparse_propagator_f2py)
  add_custom_target(wavefront_transverse_field_propagator_f2py)
endif()
