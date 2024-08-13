add_library(cartesian_obj OBJECT cartesian.f90)
add_f2py_library(MODULE_NAME cartesian SRC ${CMAKE_CURRENT_SOURCE_DIR}/cartesian.f90)


