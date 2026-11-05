#[=======================================================================[.rst:
SHAFFTDependency
----------------

Finds SHAFFT via its CMake config, or optionally fetches and builds it
using ExternalProject.

SHAFFT is built to match the current ``GPU_AWARE_MPI`` setting.  If a
cached build exists but was configured with different source or toolchain
inputs, configuration fails with a clear message instructing the user to
use a new install prefix or rerun the installer with ``--veryclean``.

Requirements
^^^^^^^^^^^^
- CMake 3.28 or later
- ROCm/HIP installation (for the hipFFT backend)
- MPI
- Fortran language enabled (SHAFFT Fortran interface is built)

Provides
^^^^^^^^
- ``shafft::shafftc++``, ``shafft::shafftc``, ``shafft::shafftf03`` targets
- ``shafft::shafft_options`` interface target
- ``SHAFFT_PROVIDER`` cache variable: ``"system"``, ``"cached"``, or ``"fetched"``
- ``SHAFFT_VERSION`` variable
- ``SHAFFT_INCLUDE_DIR`` variable (for legacy ``target_include_directories`` usage)

Usage
^^^^^
.. code-block:: cmake

   include(SHAFFTDependency)
   target_link_libraries(myapp PRIVATE shafft::shafftf03 shafft::shafftc++)

#]=======================================================================]

include_guard(GLOBAL)

cmake_minimum_required(VERSION 3.28...4.0)

# =============================================================================
# Configuration options
# =============================================================================

option(SHAFFT_AUTO_FETCH "Fetch and build SHAFFT if find_package fails" ON)

set(SHAFFT_GIT_TAG
    "main"
    CACHE STRING "Git tag/branch for SHAFFT"
)

set(SHAFFT_GIT_REPO
    "https://github.com/Edric-Matwiejew/SHAFFT.git"
    CACHE STRING "Git repository URL for SHAFFT"
)

set(SHAFFT_SOURCE_URL
    ""
    CACHE STRING "Source URL or local path; overrides git if set"
)

set(SHAFFT_SOURCE_URL_HASH
    ""
    CACHE STRING "URL hash (e.g. SHA256=...)"
)

mark_as_advanced(SHAFFT_AUTO_FETCH SHAFFT_GIT_TAG SHAFFT_GIT_REPO SHAFFT_SOURCE_URL SHAFFT_SOURCE_URL_HASH)

# =============================================================================
# Helper functions
# =============================================================================

# Populate SHAFFT_INCLUDE_DIR from the found/created targets so that legacy target_include_directories() calls in
# sub-projects still work.
function(_shafft_set_include_dir)
  foreach(_tgt shafft::shafftf03 shafft::shafftc++)
    if(TARGET ${_tgt})
      get_target_property(_inc ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
      if(_inc)
        set(SHAFFT_INCLUDE_DIR
            "${_inc}"
            CACHE PATH "SHAFFT include directories" FORCE
        )
        return()
      endif()
    endif()
  endforeach()
endfunction()

set(_shafft_cache_inputs
    SHAFFT_GIT_TAG SHAFFT_GIT_REPO SHAFFT_SOURCE_URL SHAFFT_SOURCE_URL_HASH
    OFFLOAD_ARCH CMAKE_BUILD_TYPE CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_Fortran_COMPILER
    HIP_PLATFORM CMAKE_HIP_PLATFORM CMAKE_HIP_COMPILER CMAKE_HIP_HOST_COMPILER CMAKE_HIP_ARCHITECTURES
    CMAKE_HIP_COMPILER_ROCM_ROOT
    ROCM_PATH CUDA_PATH CUDAToolkit_ROOT CMAKE_PREFIX_PATH GPU_AWARE_MPI
    MPI_C_COMPILER MPI_CXX_COMPILER MPI_Fortran_COMPILER
)

function(_shafft_write_cache_stamp stamp_file)
  file(WRITE "${stamp_file}" "set(_QUOP_SHAFFT_STAMP_VERSION 2)\n")
  foreach(_input IN LISTS _shafft_cache_inputs)
    file(APPEND "${stamp_file}" "set(_QUOP_${_input} [==[${${_input}}]==])\n")
  endforeach()
endfunction()

function(_shafft_check_cache_stamp stamp_file result_var mismatch_var)
  set(_mismatches "")

  if(NOT EXISTS "${stamp_file}")
    set(${result_var}
        FALSE
        PARENT_SCOPE
    )
    set(${mismatch_var}
        "missing cache stamp"
        PARENT_SCOPE
    )
    return()
  endif()

  include("${stamp_file}")

  if(NOT _QUOP_SHAFFT_STAMP_VERSION EQUAL 2)
    list(APPEND _mismatches "cache stamp format")
  endif()
  foreach(_input IN LISTS _shafft_cache_inputs)
    if(NOT "${_QUOP_${_input}}" STREQUAL "${${_input}}")
      list(APPEND _mismatches "${_input}")
    endif()
  endforeach()

  if(_mismatches)
    list(JOIN _mismatches ", " _joined_mismatches)
    set(${result_var}
        FALSE
        PARENT_SCOPE
    )
    set(${mismatch_var}
        "${_joined_mismatches}"
        PARENT_SCOPE
    )
    return()
  endif()

  set(${result_var}
      TRUE
      PARENT_SCOPE
  )
  set(${mismatch_var}
      ""
      PARENT_SCOPE
  )
endfunction()

# Imported packages may set HIP_PLATFORM themselves. Keep QuOp's selected platform,
# and reject mismatches before any of those imported targets are used.
set(_shafft_expected_platform "${HIP_PLATFORM}")
macro(_shafft_validate_package)
  set(HIP_PLATFORM "${_shafft_expected_platform}")
  if(NOT SHAFFT_BACKEND_HIPFFT)
    message(FATAL_ERROR "SHAFFTDependency: SHAFFT must be built with the hipFFT GPU backend")
  endif()
  if(NOT "${SHAFFT_GPU_PLATFORM}" STREQUAL "${_shafft_expected_platform}")
    message(FATAL_ERROR "SHAFFTDependency: SHAFFT GPU platform '${SHAFFT_GPU_PLATFORM}' does not match "
                        "QuOp HIP_PLATFORM=${_shafft_expected_platform}. Rebuild SHAFFT for the selected platform.")
  endif()
  if(NOT SHAFFT_BUILD_FORTRAN OR NOT TARGET shafft::shafftf03 OR NOT TARGET shafft::shafftc++)
    message(FATAL_ERROR "SHAFFTDependency: SHAFFT must provide its Fortran and C++ interfaces")
  endif()
  if(NOT DEFINED SHAFFT_GPU_AWARE_MPI
     OR (GPU_AWARE_MPI AND NOT SHAFFT_GPU_AWARE_MPI)
     OR (NOT GPU_AWARE_MPI AND SHAFFT_GPU_AWARE_MPI))
    message(FATAL_ERROR "SHAFFTDependency: SHAFFT_GPU_AWARE_MPI must match QuOp GPU_AWARE_MPI=${GPU_AWARE_MPI}")
  endif()
endmacro()

# =============================================================================
# Ensure we have a valid base directory for dependencies
# =============================================================================

if(NOT FETCHCONTENT_BASE_DIR)
  set(FETCHCONTENT_BASE_DIR
      "${PROJECT_SOURCE_DIR}/.deps"
      CACHE PATH "Directory for FetchContent downloads and builds" FORCE
  )
endif()

set(_shafft_install_dir "${FETCHCONTENT_BASE_DIR}/shafft-install")
set(_shafft_stamp_file "${_shafft_install_dir}/quop_shafft_stamp.cmake")
set(_shafft_skip_system FALSE)

# =============================================================================
# Check for previously-built SHAFFT in .deps (persists across clean builds)
# =============================================================================

# Locate the cached cmake config (could be in lib/ or lib64/)
set(_shafft_cached_config "")
foreach(_libdir lib lib64)
  set(_candidate "${_shafft_install_dir}/${_libdir}/cmake/shafft/shafftConfig.cmake")
  if(EXISTS "${_candidate}")
    set(_shafft_cached_config "${_candidate}")
    set(_shafft_cached_cmake_dir "${_shafft_install_dir}/${_libdir}/cmake/shafft")
    set(_shafft_cached_lib_dir "${_shafft_install_dir}/${_libdir}")
    break()
  endif()
endforeach()

set(_shafft_use_cached TRUE)
if(SHAFFT_PATH AND NOT SHAFFT_PATH STREQUAL _shafft_install_dir)
  set(_shafft_use_cached FALSE)
  if(shafft_DIR MATCHES "^${_shafft_install_dir}/")
    unset(shafft_DIR CACHE)
    unset(shafft_DIR)
  endif()
elseif(shafft_DIR AND NOT shafft_DIR MATCHES "^${_shafft_install_dir}/")
  set(_shafft_use_cached FALSE)
endif()

if(_shafft_cached_config AND _shafft_use_cached)
  message(STATUS "Found cached SHAFFT in ${_shafft_install_dir}")

  _shafft_check_cache_stamp("${_shafft_stamp_file}" _shafft_cache_matches _shafft_cache_mismatches)
  if(NOT _shafft_cache_matches)
    message(
      FATAL_ERROR
        "SHAFFTDependency: Cached SHAFFT inputs do not match the current configuration.\n"
        "Use a new install prefix or rerun the installer with --veryclean.\n"
        "Changed: ${_shafft_cache_mismatches}"
    )
  endif()

  find_package(shafft CONFIG QUIET PATHS "${_shafft_cached_cmake_dir}" NO_DEFAULT_PATH)
  set(HIP_PLATFORM "${_shafft_expected_platform}")

  if(shafft_FOUND)
    _shafft_validate_package()
    set(_shafft_skip_system TRUE)
    message(STATUS "Using cached SHAFFT ${shafft_VERSION}")
    set(SHAFFT_PROVIDER
        "cached"
        CACHE INTERNAL ""
    )
    set(SHAFFT_VERSION
        "${shafft_VERSION}"
        CACHE INTERNAL ""
    )
    set(SHAFFT_PATH
        "${_shafft_install_dir}"
        CACHE PATH "Path to SHAFFT installation" FORCE
    )
    _shafft_set_include_dir()
    return()
  endif()
endif()

# =============================================================================
# Try system SHAFFT (only if no cached build exists)
# =============================================================================

if(NOT _shafft_skip_system)
  # Check user-specified path first
  if(SHAFFT_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${SHAFFT_PATH}")
  elseif(DEFINED ENV{SHAFFT_PATH})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{SHAFFT_PATH}")
  endif()

  find_package(shafft CONFIG QUIET)
  set(HIP_PLATFORM "${_shafft_expected_platform}")

  if(shafft_FOUND)
    _shafft_validate_package()
    message(STATUS "Found system SHAFFT ${shafft_VERSION} at ${shafft_DIR}")
    set(SHAFFT_PROVIDER
        "system"
        CACHE INTERNAL ""
    )
    set(SHAFFT_VERSION
        "${shafft_VERSION}"
        CACHE INTERNAL ""
    )
    _shafft_set_include_dir()
    return()
  endif()
endif()

# =============================================================================
# Fetch and build SHAFFT
# =============================================================================

if(NOT SHAFFT_AUTO_FETCH)
  message(
    FATAL_ERROR
      "SHAFFTDependency: SHAFFT not found and SHAFFT_AUTO_FETCH is OFF.\n" "Options:\n"
      "  - Set SHAFFT_PATH to an existing installation\n" "  - Add SHAFFT location to CMAKE_PREFIX_PATH\n"
      "  - Set SHAFFT_AUTO_FETCH=ON to download and build automatically"
  )
endif()

message(STATUS "Fetching SHAFFT ${SHAFFT_GIT_TAG}...")

# ---- Source specification ----

if(SHAFFT_SOURCE_URL)
  if(NOT SHAFFT_SOURCE_URL MATCHES "^[a-zA-Z][a-zA-Z0-9+.-]*://")
    cmake_path(ABSOLUTE_PATH SHAFFT_SOURCE_URL OUTPUT_VARIABLE _source_url)
    if(NOT EXISTS "${_source_url}")
      message(FATAL_ERROR "SHAFFTDependency: Source path not found: ${_source_url}")
    endif()
  else()
    set(_source_url "${SHAFFT_SOURCE_URL}")
  endif()

  set(_fc_args URL "${_source_url}")
  if(SHAFFT_SOURCE_URL_HASH)
    list(APPEND _fc_args URL_HASH "${SHAFFT_SOURCE_URL_HASH}")
  endif()
else()
  set(_fc_args
      GIT_REPOSITORY
      "${SHAFFT_GIT_REPO}"
      GIT_TAG
      "${SHAFFT_GIT_TAG}"
      GIT_SHALLOW
      TRUE
      GIT_PROGRESS
      TRUE
  )
endif()

include(ExternalProject)

# ---- Build cmake arguments ----

set(_shafft_cmake_args
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=lib
    -DSHAFFT_ENABLE_HIPFFT=ON
    -DSHAFFT_BUILD_FORTRAN=ON
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DBUILD_TESTING=OFF
    -DSHAFFT_BUILD_EXAMPLES=OFF
)

# GPU-aware MPI - must match the parent project
if(GPU_AWARE_MPI)
  list(APPEND _shafft_cmake_args -DSHAFFT_GPU_AWARE_MPI=ON)
else()
  list(APPEND _shafft_cmake_args -DSHAFFT_GPU_AWARE_MPI=OFF)
endif()

# ExternalProject's list separator keeps multi-prefix installs and architecture
# lists as single -D arguments instead of splitting them into stray arguments.
set(_shafft_prefixes ${CMAKE_PREFIX_PATH})
if(ROCM_PATH)
  list(APPEND _shafft_prefixes "${ROCM_PATH}")
endif()
list(REMOVE_DUPLICATES _shafft_prefixes)
string(REPLACE ";" "|" _shafft_prefixes_arg "${_shafft_prefixes}")
list(APPEND _shafft_cmake_args "-DCMAKE_PREFIX_PATH=${_shafft_prefixes_arg}")
string(REPLACE ";" "|" _shafft_architectures_arg "${OFFLOAD_ARCH}")
list(APPEND _shafft_cmake_args "-DSHAFFT_GPU_ARCHITECTURES=${_shafft_architectures_arg}")
list(APPEND _shafft_cmake_args "-DCMAKE_HIP_PLATFORM=${HIP_PLATFORM}")
foreach(_input CMAKE_HIP_COMPILER CMAKE_HIP_HOST_COMPILER CMAKE_HIP_ARCHITECTURES CMAKE_HIP_COMPILER_ROCM_ROOT
               CUDA_PATH CUDAToolkit_ROOT hip_DIR hipfft_DIR
               MPI_C_COMPILER MPI_CXX_COMPILER MPI_Fortran_COMPILER)
  if(DEFINED ${_input} AND NOT "${${_input}}" STREQUAL "")
    string(REPLACE ";" "|" _value "${${_input}}")
    list(APPEND _shafft_cmake_args "-D${_input}=${_value}")
  endif()
endforeach()

# Fortran compiler - should match the parent project to ensure .mod compatibility
if(CMAKE_Fortran_COMPILER)
  list(APPEND _shafft_cmake_args "-DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}")
endif()

# CXX compiler - SHAFFT uses enable_language(HIP) but also compiles plain C++ sources (e.g. gpuTT) with the CXX
# compiler. Forward it so that the SHAFFT sub-build uses the same toolchain as the parent project.
if(CMAKE_CXX_COMPILER)
  list(APPEND _shafft_cmake_args "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()

# C compiler
if(CMAKE_C_COMPILER)
  list(APPEND _shafft_cmake_args "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
endif()

# AR/RANLIB - for consistent archive creation
if(CMAKE_Fortran_COMPILER_AR)
  list(APPEND _shafft_cmake_args "-DCMAKE_AR=${CMAKE_Fortran_COMPILER_AR}")
endif()
if(CMAKE_Fortran_COMPILER_RANLIB)
  list(APPEND _shafft_cmake_args "-DCMAKE_RANLIB=${CMAKE_Fortran_COMPILER_RANLIB}")
endif()

# Cray GTL - forward the setting so SHAFFT can find GPU-transport libs
if(GPU_AWARE_MPI AND DEFINED ENV{CRAYPE_VERSION})
  list(APPEND _shafft_cmake_args -DSHAFFT_REQUIRE_GTL=OFF)
endif()

# ---- ExternalProject_Add ----

externalproject_add(
  shafft_external
  ${_fc_args}
  PREFIX "${FETCHCONTENT_BASE_DIR}/shafft-ep"
  INSTALL_DIR "${_shafft_install_dir}"
  LIST_SEPARATOR "|"
  CMAKE_ARGS ${_shafft_cmake_args}
  BUILD_BYPRODUCTS "${_shafft_install_dir}/lib/libshafftc++.so" "${_shafft_install_dir}/lib/libshafftc.so"
                   "${_shafft_install_dir}/lib/libshafftf03.so"
)

set(_shafft_generated_stamp "${FETCHCONTENT_BASE_DIR}/shafft-cache-stamp.cmake")
_shafft_write_cache_stamp("${_shafft_generated_stamp}")

ExternalProject_Add_Step(
  shafft_external
  write_cache_stamp
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_shafft_generated_stamp}" "${_shafft_stamp_file}"
  DEPENDEES install
)

# ---- Create imported targets ----

# Pre-create include directories so CMake does not complain about non-existent INTERFACE_INCLUDE_DIRECTORIES during
# configure.
file(MAKE_DIRECTORY "${_shafft_install_dir}/include")
file(MAKE_DIRECTORY "${_shafft_install_dir}/include/shafft")

set(_shafft_inc_dirs "${_shafft_install_dir}/include" "${_shafft_install_dir}/include/shafft")

# shafft::shafftc++ - main C++ library
add_library(shafft::shafftc++ SHARED IMPORTED GLOBAL)
add_dependencies(shafft::shafftc++ shafft_external)
set_target_properties(
  shafft::shafftc++ PROPERTIES IMPORTED_LOCATION "${_shafft_install_dir}/lib/libshafftc++.so"
                               INTERFACE_INCLUDE_DIRECTORIES "${_shafft_inc_dirs}"
)

# shafft::shafftc - C interface library
add_library(shafft::shafftc SHARED IMPORTED GLOBAL)
add_dependencies(shafft::shafftc shafft_external)
set_target_properties(
  shafft::shafftc PROPERTIES IMPORTED_LOCATION "${_shafft_install_dir}/lib/libshafftc.so" INTERFACE_INCLUDE_DIRECTORIES
                                                                                          "${_shafft_inc_dirs}"
)

# shafft::shafftf03 - Fortran 2003 interface library
add_library(shafft::shafftf03 SHARED IMPORTED GLOBAL)
add_dependencies(shafft::shafftf03 shafft_external)
set_target_properties(
  shafft::shafftf03 PROPERTIES IMPORTED_LOCATION "${_shafft_install_dir}/lib/libshafftf03.so"
                               INTERFACE_INCLUDE_DIRECTORIES "${_shafft_inc_dirs}"
)

# shafft::shafft_options - interface target carrying include dirs / defs
if(NOT TARGET shafft::shafft_options)
  add_library(shafft::shafft_options INTERFACE IMPORTED GLOBAL)
  add_dependencies(shafft::shafft_options shafft_external)
  set_target_properties(shafft::shafft_options PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_shafft_inc_dirs}")
endif()

# ---- Set output variables ----

set(SHAFFT_PROVIDER
    "fetched"
    CACHE INTERNAL ""
)
set(SHAFFT_VERSION
    "${SHAFFT_GIT_TAG}"
    CACHE INTERNAL ""
)
set(SHAFFT_PATH
    "${_shafft_install_dir}"
    CACHE PATH "Path to SHAFFT installation" FORCE
)
set(SHAFFT_INCLUDE_DIR
    "${_shafft_inc_dirs}"
    CACHE PATH "SHAFFT include directories" FORCE
)

message(STATUS "SHAFFT ${SHAFFT_GIT_TAG} will be installed to ${_shafft_install_dir}")
