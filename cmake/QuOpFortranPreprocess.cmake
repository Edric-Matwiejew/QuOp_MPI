#[=======================================================================[.rst:
QuOpFortranPreprocess
---------------------

Shared helpers for preprocessing Fortran sources and adding explicit source
barriers for generated module dependencies.

Provides
^^^^^^^^
- ``quop_preprocess_fortran_source(out_var ...)``
- ``quop_add_fortran_source_barrier(...)``

#]=======================================================================]

include_guard(GLOBAL)

function(quop_preprocess_fortran_source out_var)
  set(options "")
  set(oneValueArgs NAME SOURCE)
  set(multiValueArgs DEFINITIONS INCLUDE_DIRS DEPENDS)
  cmake_parse_arguments(QUOP_PP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT QUOP_PP_SOURCE)
    message(FATAL_ERROR "quop_preprocess_fortran_source requires SOURCE")
  endif()

  if(QUOP_PP_NAME)
    set(_name "${QUOP_PP_NAME}")
  else()
    get_filename_component(_name "${QUOP_PP_SOURCE}" NAME_WE)
  endif()

  set(_pp_definitions "")
  foreach(_def ${QUOP_PP_DEFINITIONS})
    list(APPEND _pp_definitions "-D${_def}")
  endforeach()

  set(_pp_includes "")
  foreach(_inc ${QUOP_PP_INCLUDE_DIRS})
    list(APPEND _pp_includes "-I${_inc}")
  endforeach()

  set(_output "${CMAKE_CURRENT_BINARY_DIR}/preprocessed_${_name}.f90")

  add_custom_command(
    OUTPUT "${_output}"
    COMMAND ${CMAKE_Fortran_COMPILER} -cpp -E ${_pp_definitions} ${_pp_includes} "${QUOP_PP_SOURCE}" -o "${_output}"
    DEPENDS "${QUOP_PP_SOURCE}" ${QUOP_PP_DEPENDS}
    COMMENT "Preprocessing ${QUOP_PP_SOURCE}"
    VERBATIM
  )

  set(${out_var}
      "${_output}"
      PARENT_SCOPE
  )
endfunction()

function(quop_add_fortran_source_barrier)
  set(options "")
  set(oneValueArgs SOURCE NAME COMMENT)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(QUOP_BARRIER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT QUOP_BARRIER_SOURCE)
    message(FATAL_ERROR "quop_add_fortran_source_barrier requires SOURCE")
  endif()
  if(NOT QUOP_BARRIER_NAME)
    message(FATAL_ERROR "quop_add_fortran_source_barrier requires NAME")
  endif()

  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/${QUOP_BARRIER_NAME}.stamp")
  if(QUOP_BARRIER_COMMENT)
    set(_comment "${QUOP_BARRIER_COMMENT}")
  else()
    set(_comment "Waiting for prerequisites of ${QUOP_BARRIER_SOURCE}")
  endif()

  add_custom_command(
    OUTPUT "${_stamp}"
    COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
    DEPENDS ${QUOP_BARRIER_DEPENDS}
    COMMENT "${_comment}"
    VERBATIM
  )

  set_source_files_properties("${QUOP_BARRIER_SOURCE}" PROPERTIES OBJECT_DEPENDS "${_stamp}")
endfunction()
