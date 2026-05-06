#[=======================================================================[.rst:
QuOpContextExtension
--------------------

Build helper for the backend-agnostic CPython context-wrapper extension.

Provides
^^^^^^^^
- ``add_context_extension(...)``

Overview
^^^^^^^^

The native context wrapper consists of two backend-agnostic source files:

- ``native/context_wrapper_c.f90`` -- a ``bind(C)`` Fortran shim that
  routes through the per-backend ``context_type`` (resolved at compile
  time via ``-Dcontext=<backend>`` and ``-Dcontext_type=<type>``).
- ``native/context_wrapper_ext.c`` -- a CPython extension that owns the
  Python-side state ndarray and calls into the shim.

Each backend invokes ``add_context_extension`` from its own
``CMakeLists.txt``, supplying its OBJECT-library dependencies and the
preprocessor definitions that select the backend's ``context_type``.
The resulting Python extension is installed at
``quop_mpi/_lib/<INSTALL_SUBDIR>/<NAME>.<soabi>.so``.

Mirrors the design of :command:`add_f2py_library` for the f2py wrappers.

Arguments
^^^^^^^^^

``NAME``
    The Python module name (e.g. ``context_wrapper``).  Becomes the
    target name and the installed ``.so`` basename.

``INSTALL_SUBDIR``
    Backend subdirectory under ``quop_mpi/_lib/`` to install into
    (e.g. ``mpi`` or ``wavefront``).

``DEFINITIONS``
    Preprocessor definitions for the Fortran shim, typically
    ``context=<backend_module> context_type=<derived_type>``.

``DEPENDS``
    OBJECT-library dependencies (e.g. the backend ``*_context``
    OBJECT lib and ``comm_info_mod``).  Used both for ordering
    (``add_dependencies``) and linking.

``LIBRARIES``
    Additional link libraries (e.g. ``${QUOP_HIPFORT_TARGET}``).

#]=======================================================================]

include_guard(GLOBAL)

function(add_context_extension)
  set(options "")
  set(oneValueArgs NAME INSTALL_SUBDIR)
  set(multiValueArgs DEFINITIONS DEPENDS LIBRARIES)
  cmake_parse_arguments(CTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT CTX_NAME)
    message(FATAL_ERROR "add_context_extension: NAME is required")
  endif()
  if(NOT CTX_INSTALL_SUBDIR)
    message(FATAL_ERROR "add_context_extension: INSTALL_SUBDIR is required")
  endif()

  set(_target       "${CTX_INSTALL_SUBDIR}_${CTX_NAME}")
  set(_shim_target  "${_target}_shim")
  set(_shim_src     "${CMAKE_SOURCE_DIR}/native/context_wrapper_c.f90")
  set(_ext_src      "${CMAKE_SOURCE_DIR}/native/context_wrapper_ext.c")

  # ----- Per-backend Fortran shim (bind(C) entry points) -----
  add_library(${_shim_target} OBJECT ${_shim_src})
  target_compile_definitions(${_shim_target} PRIVATE ${CTX_DEFINITIONS})
  set_target_properties(${_shim_target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    Fortran_MODULE_DIRECTORY  "${CMAKE_BINARY_DIR}/modules/${_shim_target}"
  )
  target_include_directories(${_shim_target} PRIVATE
    "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )
  target_link_libraries(${_shim_target} PRIVATE MPI::MPI_Fortran)
  if(CTX_DEPENDS)
    add_dependencies(${_shim_target} ${CTX_DEPENDS})
  endif()

  # ----- Python extension module -----
  # The CMake target is namespaced by INSTALL_SUBDIR so that multiple
  # backends can install a module of the same Python-facing NAME (e.g.
  # both `mpi` and `wavefront` produce a module called `context_wrapper`)
  # without colliding in CMake's target registry.  OUTPUT_NAME restores
  # the unprefixed library basename used by the Python loader.
  Python3_add_library(${_target} MODULE WITH_SOABI ${_ext_src})
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME "${CTX_NAME}")
  target_include_directories(${_target} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
  )
  target_link_libraries(${_target} PRIVATE
    ${_shim_target}
    ${CTX_DEPENDS}
    ${CTX_LIBRARIES}
    MPI::MPI_Fortran
  )
  set_target_properties(${_target} PROPERTIES
    LINKER_LANGUAGE Fortran
  )
  install(TARGETS ${_target}
    LIBRARY DESTINATION "quop_mpi/_lib/${CTX_INSTALL_SUBDIR}"
  )
endfunction()
