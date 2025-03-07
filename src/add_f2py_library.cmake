include(CMakeParseArguments REQUIRED)

function(add_f2py_library)
  set(options "")
  set(oneValueArgs MODULE_NAME SRC INSTALL_SUBDIR)
  set(multiValueArgs DEPENDS DEFINITIONS INCLUDE_DIRS LIBRARIES)
  cmake_parse_arguments(F2PY_LIBRARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(local_mod_dir "${CMAKE_BINARY_DIR}/modules/${F2PY_LIBRARY_MODULE_NAME}")

  if(NOT DEFINED CMAKE_Fortran_MODULE_DIRECTORY)
    set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules")
  endif()

  set(f2py_cmap           "${CMAKE_SOURCE_DIR}/src/.f2py_f2cmap")
  set(module_pyf          "${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}.pyf")
  set(module_f2py_wrapper "${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}-f2pywrappers2.f90")
  set(module_f2py_c       "${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}module.c")

  # Removed forced rebuild stamp and target

  set(F2PY_PP_DEFINITIONS "")
  foreach(def ${F2PY_LIBRARY_DEFINITIONS})
    list(APPEND F2PY_PP_DEFINITIONS "-D${def}")
  endforeach()

  set(F2PY_PP_INCLUDES "")
  foreach(inc ${F2PY_LIBRARY_INCLUDE_DIRS})
    list(APPEND F2PY_PP_INCLUDES "-I${inc}")
  endforeach()

  set(PREPROCESSED_SRC "${CMAKE_CURRENT_BINARY_DIR}/preprocessed_${F2PY_LIBRARY_MODULE_NAME}.F90")

  add_custom_command(
    OUTPUT "${PREPROCESSED_SRC}"
    COMMAND ${CMAKE_Fortran_COMPILER}
        -cpp -E
        ${F2PY_PP_DEFINITIONS}
        ${F2PY_PP_INCLUDES}
        "${F2PY_LIBRARY_SRC}"
        -o "${PREPROCESSED_SRC}"
    DEPENDS "${F2PY_LIBRARY_SRC}"
    COMMENT "Preprocessing ${F2PY_LIBRARY_SRC} with Fortran compiler and definitions"
  )

  add_custom_command(
    OUTPUT  "${module_pyf}"
    COMMAND "${Python3_EXECUTABLE}" -m numpy.f2py
            -h "${module_pyf}"
            "${PREPROCESSED_SRC}"
            --overwrite-signature
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${PREPROCESSED_SRC}"
    COMMENT "Generating .pyf file using numpy.f2py with preprocessed source"
    VERBATIM
  )

  add_custom_command(
    OUTPUT  "${module_f2py_wrapper}" "${module_f2py_c}"
    COMMAND "${Python3_EXECUTABLE}" -m numpy.f2py
            --f2cmap "${f2py_cmap}"
            -m "${F2PY_LIBRARY_MODULE_NAME}"
            "${module_pyf}"
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

  target_include_directories("${F2PY_LIBRARY_MODULE_NAME}module"
    PRIVATE
      ${F2PY_LIBRARY_INCLUDE_DIRS}
      "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )
  target_compile_definitions("${F2PY_LIBRARY_MODULE_NAME}module"
    PRIVATE
      ${F2PY_LIBRARY_DEFINITIONS}
  )
  target_compile_options("${F2PY_LIBRARY_MODULE_NAME}module"
    PRIVATE -cpp
  )
  set_target_properties("${F2PY_LIBRARY_MODULE_NAME}module"
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      Fortran_MODULE_DIRECTORY "${local_mod_dir}"
  )

  if(DEFINED F2PY_LIBRARY_DEPENDS)
    add_dependencies("${F2PY_LIBRARY_MODULE_NAME}module"
      ${F2PY_LIBRARY_DEPENDS})
  endif()

  add_library("${F2PY_LIBRARY_MODULE_NAME}wrapper" OBJECT "${module_f2py_wrapper}")
  set_target_properties("${F2PY_LIBRARY_MODULE_NAME}wrapper"
    PROPERTIES
      Fortran_MODULE_DIRECTORY "${local_mod_dir}"
  )
  target_include_directories("${F2PY_LIBRARY_MODULE_NAME}wrapper"
    PRIVATE
      ${F2PY_LIBRARY_INCLUDE_DIRS}
      "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )

  add_dependencies("${F2PY_LIBRARY_MODULE_NAME}module"  ${generate_f2py_target_name})
  add_dependencies("${F2PY_LIBRARY_MODULE_NAME}wrapper" ${generate_f2py_target_name})

  set(f2py_target_name "${F2PY_LIBRARY_MODULE_NAME}_f2py")

  add_library("${f2py_target_name}" SHARED
    "${module_f2py_c}"
    "${F2PY_INCLUDE_DIR}/fortranobject.c"
  )

  set_target_properties("${f2py_target_name}"
    PROPERTIES
      PREFIX ""  # remove "lib" prefix for Python extension
      OUTPUT_NAME "${F2PY_LIBRARY_MODULE_NAME}.${Python3_SOABI}"
      SUFFIX ".so"
      LINKER_LANGUAGE Fortran
  )

  target_include_directories("${f2py_target_name}"
    PUBLIC
      ${Python3_INCLUDE_DIRS}
      "${F2PY_INCLUDE_DIR}"
      ${Python3_NumPy_INCLUDE_DIRS}
      ${F2PY_LIBRARY_INCLUDE_DIRS}
      "${CMAKE_Fortran_MODULE_DIRECTORY}"
  )

  target_link_libraries("${f2py_target_name}"
    "${F2PY_LIBRARY_MODULE_NAME}module"
    "${F2PY_LIBRARY_MODULE_NAME}wrapper"
    ${F2PY_LIBRARY_DEPENDS}
    ${F2PY_LIBRARY_LIBRARIES}
  )

  if(APPLE)
    set_target_properties("${f2py_target_name}"
      PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
  else()
    set_target_properties("${f2py_target_name}"
      PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")
  endif()

  install(
    TARGETS "${f2py_target_name}"
    DESTINATION "quop_mpi/__lib/${F2PY_LIBRARY_INSTALL_SUBDIR}"
  )

  if(NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" CACHE STRING "Path to the Python 3 executable")
  endif()
  if(NOT DEFINED Python3_NumPy_INCLUDE_DIRS)
    set(Python3_NumPy_INCLUDE_DIRS "${Python3_NumPy_INCLUDE_DIRS}" CACHE STRING "Path to Python3 NumPy include directories")
  endif()
  if(NOT DEFINED F2PY_INCLUDE_DIR)
    set(F2PY_INCLUDE_DIR "${F2PY_INCLUDE_DIR}" CACHE STRING "Path to F2PY include directory")
  endif()

endfunction()

