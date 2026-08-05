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
- CMake 3.25 or later
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

cmake_minimum_required(VERSION 3.25...4.0)

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

function(_shafft_write_cache_stamp stamp_file)
  file(WRITE "${stamp_file}" "")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_GIT_TAG [==[${SHAFFT_GIT_TAG}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_GIT_REPO [==[${SHAFFT_GIT_REPO}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_SOURCE_URL [==[${SHAFFT_SOURCE_URL}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_SOURCE_URL_HASH [==[${SHAFFT_SOURCE_URL_HASH}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_OFFLOAD_ARCH [==[${OFFLOAD_ARCH}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_BUILD_TYPE [==[${CMAKE_BUILD_TYPE}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_C_COMPILER [==[${CMAKE_C_COMPILER}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_CXX_COMPILER [==[${CMAKE_CXX_COMPILER}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_Fortran_COMPILER [==[${CMAKE_Fortran_COMPILER}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_ROCM_PATH [==[${ROCM_PATH}]==])\n")
  file(APPEND "${stamp_file}" "set(_QUOP_SHAFFT_GPU_AWARE_MPI [==[${GPU_AWARE_MPI}]==])\n")
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

  if(NOT "${_QUOP_SHAFFT_GIT_TAG}" STREQUAL "${SHAFFT_GIT_TAG}")
    list(APPEND _mismatches "SHAFFT_GIT_TAG")
  endif()
  if(NOT "${_QUOP_SHAFFT_GIT_REPO}" STREQUAL "${SHAFFT_GIT_REPO}")
    list(APPEND _mismatches "SHAFFT_GIT_REPO")
  endif()
  if(NOT "${_QUOP_SHAFFT_SOURCE_URL}" STREQUAL "${SHAFFT_SOURCE_URL}")
    list(APPEND _mismatches "SHAFFT_SOURCE_URL")
  endif()
  if(NOT "${_QUOP_SHAFFT_SOURCE_URL_HASH}" STREQUAL "${SHAFFT_SOURCE_URL_HASH}")
    list(APPEND _mismatches "SHAFFT_SOURCE_URL_HASH")
  endif()
  if(NOT "${_QUOP_SHAFFT_OFFLOAD_ARCH}" STREQUAL "${OFFLOAD_ARCH}")
    list(APPEND _mismatches "OFFLOAD_ARCH")
  endif()
  if(NOT "${_QUOP_SHAFFT_BUILD_TYPE}" STREQUAL "${CMAKE_BUILD_TYPE}")
    list(APPEND _mismatches "CMAKE_BUILD_TYPE")
  endif()
  if(NOT "${_QUOP_SHAFFT_C_COMPILER}" STREQUAL "${CMAKE_C_COMPILER}")
    list(APPEND _mismatches "CMAKE_C_COMPILER")
  endif()
  if(NOT "${_QUOP_SHAFFT_CXX_COMPILER}" STREQUAL "${CMAKE_CXX_COMPILER}")
    list(APPEND _mismatches "CMAKE_CXX_COMPILER")
  endif()
  if(NOT "${_QUOP_SHAFFT_Fortran_COMPILER}" STREQUAL "${CMAKE_Fortran_COMPILER}")
    list(APPEND _mismatches "CMAKE_Fortran_COMPILER")
  endif()
  if(NOT "${_QUOP_SHAFFT_ROCM_PATH}" STREQUAL "${ROCM_PATH}")
    list(APPEND _mismatches "ROCM_PATH")
  endif()
  if(NOT "${_QUOP_SHAFFT_GPU_AWARE_MPI}" STREQUAL "${GPU_AWARE_MPI}")
    list(APPEND _mismatches "GPU_AWARE_MPI")
  endif()

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

if(_shafft_cached_config)
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

  if(shafft_FOUND)
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

  if(shafft_FOUND)
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

# GPU architectures
if(OFFLOAD_ARCH)
  list(APPEND _shafft_cmake_args "-DSHAFFT_GPU_ARCHITECTURES=${OFFLOAD_ARCH}")
endif()

# ROCm path - needed for find_package(hip), find_package(hipfft), etc.
if(ROCM_PATH)
  list(APPEND _shafft_cmake_args "-DCMAKE_PREFIX_PATH=${ROCM_PATH}")
endif()

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
