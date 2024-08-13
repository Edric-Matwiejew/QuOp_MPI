include(CMakeParseArguments REQUIRED)

function(add_f2py_library)
set(OPTIONS)
set(oneValueArgs MODULE_NAME SRC INSTALL_SUBDIR)
set(multiValueArgs DEPENDS DEFINITIONS INCLUDE_DIRS LIBRARIES)
cmake_parse_arguments(F2PY_LIBRARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

set(f2py_cmap ${CMAKE_SOURCE_DIR}/src/.f2py_f2cmap)
set(module_pyf ${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}.pyf)
set(module_f2py_wrapper ${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}-f2pywrappers2.f90)
set(module_f2py_c ${CMAKE_CURRENT_BINARY_DIR}/${F2PY_LIBRARY_MODULE_NAME}module.c)

add_custom_command(
	OUTPUT ${module_pyf} 
	PRE_BUILD
  	COMMAND ${Python3_EXECUTABLE} -m numpy.f2py
  	-h ${module_pyf}
  	${F2PY_LIBRARY_SRC}	
  	--overwrite-signature > /dev/null
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  	DEPENDS ${F2PY_LIBRARY_SRC}
)

add_custom_command( 
  OUTPUT ${module_f2py_wrapper} ${module_f2py_c}
  PRE_BUILD
  COMMAND ${Python3_EXECUTABLE} -m numpy.f2py
  --f2cmap ${f2py_cmap}
  -m ${F2PY_LIBRARY_MODULE_NAME}
  ${module_pyf} > /dev/null
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${module_pyf}
)

add_library(${F2PY_LIBRARY_MODULE_NAME}module OBJECT ${F2PY_LIBRARY_SRC})
target_include_directories(${F2PY_LIBRARY_MODULE_NAME}module PRIVATE ${F2PY_LIBRARY_INCLUDE_DIRS})

target_compile_definitions(${F2PY_LIBRARY_MODULE_NAME}module PRIVATE ${F2PY_LIBRARY_DEFINITIONS})
target_compile_options(${F2PY_LIBRARY_MODULE_NAME}module PRIVATE -cpp)

if(DEFINED F2PY_LIBRARY_DEPENDS)
add_dependencies(${F2PY_LIBRARY_MODULE_NAME}module ${F2PY_LIBRARY_DEPENDS})
endif()

add_library(${F2PY_LIBRARY_MODULE_NAME}wrapper OBJECT ${module_f2py_wrapper})
add_dependencies(${F2PY_LIBRARY_MODULE_NAME}wrapper ${F2PY_LIBRARY_MODULE_NAME}module)

set(f2py_module ${F2PY_LIBRARY_MODULE_NAME}.${Python3_SOABI}.so)

add_library(
	${f2py_module}
	SHARED
 	${module_f2py_c}
	${F2PY_INCLUDE_DIR}/fortranobject.c
)

set_target_properties(
 	${f2py_module}
 	PROPERTIES
 	SUFFIX ""
 	PREFIX ""	
)
 
target_include_directories(
 	${f2py_module}
 	PUBLIC
 	${Python3_INCLUDE_DIRS}
 	${F2PY_INCLUDE_DIR}
 	${Python3_NumPy_INCLUDE_DIRS}
	${F2PY_LIBRARY_INCLUDE_DIRS}
)

target_link_libraries(
 	${f2py_module}
	${F2PY_LIBRARY_MODULE_NAME}module
	${F2PY_LIBRARY_MODULE_NAME}wrapper
  	${F2PY_LIBRARY_DEPENDS}
	${F2PY_LIBRARY_LIBRARIES}
) 

set_target_properties(
	${f2py_module}
	PROPERTIES
	LINKER_LANGUAGE Fortran
)

# Linker fixes
if (UNIX)
  if (APPLE)
    set_target_properties(${f2py_module} PROPERTIES
	    LINK_FLAGS  '-Wl,-dylib,-undefined,dynamic_lookup,-fPIC,-shared')
  else()
    set_target_properties(${f2py_module} PROPERTIES
	    LINK_FLAGS  '-Wl,--allow-shlib-undefined,-fPIC,-shared')
  endif()
endif()

string(TOLOWER ${F2PY_LIBRARY_MODULE_NAME} F2PY_LIBRARY_MODULE_NAME_lowercase)
#install(TARGETS ${f2py_module} DESTINATION ./${F2PY_LIBRARY_INSTALL_SUBDIR})
install(TARGETS ${f2py_module} DESTINATION quop_mpi/__lib/${F2PY_LIBRARY_INSTALL_SUBDIR})
#add_custom_command(
#	TARGET ${f2py_module}
# 	COMMENT "Test module import."
#	POST_BUILD	
#	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#	COMMAND ${Python3_EXECUTABLE} -c \'import ${F2PY_LIBRARY_MODULE_NAME_lowercase}\;print\(${F2PY_LIBRARY_MODULE_NAME_lowercase}.__doc__\)\'
#)

set(Python3_EXECUTABLE ${Python3_EXECUTABLE} CACHE STRING "")
set(Python3_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS} CACHE STRING "")
set(F2PY_INCLUDE_DIR ${F2PY_INCLUDE_DIR} CACHE STRING "")

endfunction()

