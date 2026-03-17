#[=======================================================================[.rst:
QuOpF2pyLibrary
---------------

Shared helper for building and installing f2py-backed extension modules.

Provides
^^^^^^^^
- ``add_f2py_library(...)``

#]=======================================================================]

include_guard(GLOBAL)

function(add_f2py_library)
  set(options "")
  set(oneValueArgs MODULE_NAME SRC INSTALL_SUBDIR)
  set(multiValueArgs DEPENDS DEFINITIONS INCLUDE_DIRS LIBRARIES)
  cmake_parse_arguments(F2PY_LIBRARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(local_mod_dir "${CMAKE_BINARY_DIR}/modules/${F2PY_LIBRARY_MODULE_NAME}")

  if(NOT DEFINED CMAKE_Fortran_MODULE_DIRECTORY)
    set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules")
  endif()

  set(f2py_cmap "${CMAKE_SOURCE_DIR}/src/.f2py_f2cmap")
  set(module_pyf "${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}.pyf")
  set(module_f2py_wrapper "${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}-f2pywrappers2.f90")
  set(module_f2py_c "${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}module.c")

  quop_preprocess_fortran_source(
    PREPROCESSED_SRC
    NAME
    "${F2PY_LIBRARY_MODULE_NAME}"
    SOURCE
    "${F2PY_LIBRARY_SRC}"
    DEFINITIONS
    ${F2PY_LIBRARY_DEFINITIONS}
    INCLUDE_DIRS
    ${F2PY_LIBRARY_INCLUDE_DIRS}
  )

  add_custom_command(
    OUTPUT "${module_pyf}"
    COMMAND "${Python3_EXECUTABLE}" -m numpy.f2py -h "${module_pyf}" -m "${F2PY_LIBRARY_MODULE_NAME}"
            "${PREPROCESSED_SRC}" --overwrite-signature
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${PREPROCESSED_SRC}"
    COMMENT "Generating .pyf file using numpy.f2py with preprocessed source"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${module_f2py_wrapper}" "${module_f2py_c}"
    COMMAND "${Python3_EXECUTABLE}" -m numpy.f2py --f2cmap "${f2py_cmap}" "${module_pyf}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${module_pyf}"
    COMMENT "Generating Fortran f2py wrapper and C module"
    VERBATIM
  )

  set(generate_f2py_target_name "generate_f2py_files_${F2PY_LIBRARY_MODULE_NAME}")
  add_custom_target(
    ${generate_f2py_target_name}
    DEPENDS "${module_pyf}" "${module_f2py_wrapper}" "${module_f2py_c}"
    COMMENT "Generating all f2py intermediary files for ${F2PY_LIBRARY_MODULE_NAME}"
  )

  add_library("${F2PY_LIBRARY_MODULE_NAME}module" OBJECT "${PREPROCESSED_SRC}")
  add_dependencies("${F2PY_LIBRARY_MODULE_NAME}module" ${generate_f2py_target_name})

  target_include_directories(
    "${F2PY_LIBRARY_MODULE_NAME}module" PRIVATE ${F2PY_LIBRARY_INCLUDE_DIRS} "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )
  target_compile_definitions("${F2PY_LIBRARY_MODULE_NAME}module" PRIVATE ${F2PY_LIBRARY_DEFINITIONS})
  set_target_properties(
    "${F2PY_LIBRARY_MODULE_NAME}module" PROPERTIES POSITION_INDEPENDENT_CODE ON Fortran_MODULE_DIRECTORY
                                                                                "${local_mod_dir}"
  )

  # Link libraries to the module target so that imported targets like MPI::MPI_Fortran propagate their include
  # directories (for mpi.mod) during compilation. This is required because OBJECT libraries need the include paths at
  # compile time.
  if(DEFINED F2PY_LIBRARY_LIBRARIES)
    target_link_libraries("${F2PY_LIBRARY_MODULE_NAME}module" PRIVATE ${F2PY_LIBRARY_LIBRARIES})
  endif()

  if(DEFINED F2PY_LIBRARY_DEPENDS)
    add_dependencies("${F2PY_LIBRARY_MODULE_NAME}module" ${F2PY_LIBRARY_DEPENDS})
  endif()

  # CMake/Ninja tracks Fortran .mod files through dyndep, not as normal declared outputs. Force the wrapper source to
  # wait for the module target that provides the underlying Fortran module interface.
  quop_add_fortran_source_barrier(
    SOURCE
    "${module_f2py_wrapper}"
    NAME
    "${F2PY_LIBRARY_MODULE_NAME}_module_ready"
    COMMENT
    "Waiting for ${F2PY_LIBRARY_MODULE_NAME}module before compiling wrapper"
    DEPENDS
    "${F2PY_LIBRARY_MODULE_NAME}module"
  )

  add_library("${F2PY_LIBRARY_MODULE_NAME}wrapper" OBJECT "${module_f2py_wrapper}")
  set_target_properties(
    "${F2PY_LIBRARY_MODULE_NAME}wrapper" PROPERTIES Fortran_MODULE_DIRECTORY "${local_mod_dir}"
                                                    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(
    "${F2PY_LIBRARY_MODULE_NAME}wrapper" PRIVATE ${F2PY_LIBRARY_INCLUDE_DIRS} "${CMAKE_Fortran_MODULE_DIRECTORY}"
                                                 "${local_mod_dir}"
  )

  # Link libraries to the wrapper target so that imported targets like hipfort propagate their include directories
  # during compilation (for .mod files).
  if(DEFINED F2PY_LIBRARY_LIBRARIES)
    target_link_libraries("${F2PY_LIBRARY_MODULE_NAME}wrapper" PRIVATE ${F2PY_LIBRARY_LIBRARIES})
  endif()

  add_dependencies("${F2PY_LIBRARY_MODULE_NAME}wrapper" ${generate_f2py_target_name})

  # The f2py wrapper source uses the underlying Fortran module from SRC, so it needs the corresponding .mod file before
  # compilation.
  add_dependencies("${F2PY_LIBRARY_MODULE_NAME}wrapper" "${F2PY_LIBRARY_MODULE_NAME}module")

  set(f2py_target_name "${F2PY_LIBRARY_MODULE_NAME}_f2py")

  add_library("${f2py_target_name}" SHARED "${module_f2py_c}" "${F2PY_INCLUDE_DIR}/fortranobject.c")

  set_target_properties(
    "${f2py_target_name}"
    PROPERTIES PREFIX "" # remove "lib" prefix for Python extension
               OUTPUT_NAME "${F2PY_LIBRARY_MODULE_NAME}.${Python3_SOABI}"
               SUFFIX ".so"
               LINKER_LANGUAGE Fortran
               # scikit-build-core installs directly from the build tree, so keep the
               # extension output in the current binary dir for every configuration.
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(
      "${f2py_target_name}" PROPERTIES "LIBRARY_OUTPUT_DIRECTORY_${_cfg}" "${CMAKE_CURRENT_BINARY_DIR}"
                                       "RUNTIME_OUTPUT_DIRECTORY_${_cfg}" "${CMAKE_CURRENT_BINARY_DIR}"
    )
  endforeach()

  target_include_directories(
    "${f2py_target_name}" PUBLIC ${Python3_INCLUDE_DIRS} "${F2PY_INCLUDE_DIR}" ${Python3_NumPy_INCLUDE_DIRS}
                                 ${F2PY_LIBRARY_INCLUDE_DIRS} "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )

  target_link_libraries(
    "${f2py_target_name}" PRIVATE "${F2PY_LIBRARY_MODULE_NAME}module" "${F2PY_LIBRARY_MODULE_NAME}wrapper"
                                  ${F2PY_LIBRARY_DEPENDS} ${F2PY_LIBRARY_LIBRARIES}
  )
  add_dependencies(
    "${f2py_target_name}" ${generate_f2py_target_name} "${F2PY_LIBRARY_MODULE_NAME}module"
    "${F2PY_LIBRARY_MODULE_NAME}wrapper" ${F2PY_LIBRARY_DEPENDS}
  )

  if(APPLE)
    set_target_properties("${f2py_target_name}" PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
  else()
    set_target_properties("${f2py_target_name}" PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")
  endif()

  install(TARGETS "${f2py_target_name}" DESTINATION "quop_mpi/_lib/${F2PY_LIBRARY_INSTALL_SUBDIR}")
endfunction()
