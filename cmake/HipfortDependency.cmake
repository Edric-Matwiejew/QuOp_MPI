#[=======================================================================[.rst:
HipfortDependency
-----------------

Finds hipfort via its CMake config, or optionally fetches and builds it
using the current Fortran toolchain.

Requirements
^^^^^^^^^^^^
- CMake 3.28 or later
- Fortran language enabled in the parent project
- HIP installation (ROCm on AMD, CUDA-backed HIP on NVIDIA)

Provides
^^^^^^^^
- ``hipfort::hipfort`` target (alias to ``hipfort::hip`` if needed)
- ``HIPFORT_PROVIDER`` cache variable: ``"system"`` or ``"fetched"``
- ``HIPFORT_VERSION`` variable (if detected)

Usage
^^^^^
.. code-block:: cmake

   include(HipfortDependency)
   target_link_libraries(myapp PRIVATE hipfort::hipfort)

#]=======================================================================]

include_guard(GLOBAL)

cmake_minimum_required(VERSION 3.28...4.0)

# =============================================================================
# Configuration options
# =============================================================================

option(HIPFORT_AUTO_FETCH "Fetch and build if find_package fails" ON)

option(HIPFORT_VERIFY_MOD_COMPAT "Verify .mod compatibility with Fortran compiler" ON)

if(HIP_PLATFORM STREQUAL "nvidia")
  set(_hipfort_require_rocm_default OFF)
else()
  set(_hipfort_require_rocm_default ON)
endif()
option(HIPFORT_REQUIRE_ROCM "Require ROCm installation for AMD builds" ${_hipfort_require_rocm_default})

set(HIPFORT_GIT_TAG
    ""
    CACHE STRING "Git tag/branch; auto-detected from ROCm if empty"
)

set(HIPFORT_GIT_REPO
    "https://github.com/ROCm/hipfort.git"
    CACHE STRING "Git repository URL"
)

set(HIPFORT_SOURCE_URL
    ""
    CACHE STRING "Source URL or path; overrides git"
)

set(HIPFORT_SOURCE_URL_HASH
    ""
    CACHE STRING "URL hash (e.g. SHA256=...)"
)

mark_as_advanced(
  HIPFORT_AUTO_FETCH
  HIPFORT_VERIFY_MOD_COMPAT
  HIPFORT_REQUIRE_ROCM
  HIPFORT_GIT_TAG
  HIPFORT_GIT_REPO
  HIPFORT_SOURCE_URL
  HIPFORT_SOURCE_URL_HASH
)

# =============================================================================
# Helper function and macro
# =============================================================================

# Check .mod file compatibility with current Fortran compiler
function(_hipfort_check_mod_compatibility target result_var)
  get_target_property(_inc_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)

  message(VERBOSE "HipfortDependency: checking mod compatibility target=${target}; inc_dirs=${_inc_dirs}")

  set(_mod_flags "")
  foreach(_dir IN LISTS _inc_dirs)
    if(_dir
       AND NOT _dir MATCHES "^\\$<"
       AND IS_DIRECTORY "${_dir}"
    )
      string(APPEND _mod_flags " -I${_dir}")
    endif()
  endforeach()
  message(VERBOSE "HipfortDependency: _mod_flags='${_mod_flags}'")

  include(CheckFortranSourceCompiles)
  set(_save_req_includes ${CMAKE_REQUIRED_INCLUDES})
  set(CMAKE_REQUIRED_INCLUDES ${_inc_dirs})
  # CMake's check result is cached by name, so never reuse another platform's
  # successful module check after a toolchain or package change.
  unset(_compat_ok CACHE)
  # Use SRC_EXT F90 to force free-form Fortran (default .F is fixed-form)
  check_fortran_source_compiles("program p; use hipfort; use hipfort_check; end program" _compat_ok SRC_EXT F90)
  set(_compat_log "CMAKE_REQUIRED_INCLUDES='${CMAKE_REQUIRED_INCLUDES}' _mod_flags='${_mod_flags}'")
  set(CMAKE_REQUIRED_INCLUDES ${_save_req_includes})

  set(${result_var}
      ${_compat_ok}
      PARENT_SCOPE
  )
  set(${result_var}_LOG
      ${_compat_log}
      PARENT_SCOPE
  )
endfunction()

# ROCm 6.x installs the CUDA archive and modules, but its CMake package exports
# only amdgcn targets. Import that documented layout without loading AMD runtime
# dependencies from the package's optional "hip" component.
function(_hipfort_import_nvptx prefix result_var)
  find_library(_nvptx_library NAMES hipfort-nvptx PATHS "${prefix}"
               PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH NO_CACHE)
  find_path(_nvptx_modules hipfort.mod PATHS "${prefix}"
            PATH_SUFFIXES include/hipfort/nvptx NO_DEFAULT_PATH NO_CACHE)
  if(_nvptx_library AND _nvptx_modules)
    if(NOT TARGET hipfort::hipfort-nvptx)
      add_library(hipfort::hipfort-nvptx STATIC IMPORTED GLOBAL)
    endif()
    set_target_properties(hipfort::hipfort-nvptx PROPERTIES
                          IMPORTED_LOCATION "${_nvptx_library}"
                          INTERFACE_INCLUDE_DIRECTORIES "${_nvptx_modules}")
    if(TARGET quop_gpu_runtime)
      set_property(TARGET hipfort::hipfort-nvptx PROPERTY INTERFACE_LINK_LIBRARIES quop_gpu_runtime)
    endif()
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Ensure hipfort::hipfort alias exists, linking to the appropriate platform target
macro(_hipfort_ensure_alias)
  if(NOT TARGET hipfort::hipfort)
    # Determine platform
    if(HIP_PLATFORM STREQUAL "nvidia")
      set(_alias_platform "nvptx")
    else()
      set(_alias_platform "amdgcn")
    endif()

    # Prefer platform-specific target, fall back to hipfort::hip
    if(TARGET hipfort::hipfort-${_alias_platform})
      add_library(hipfort::hipfort INTERFACE IMPORTED)
      set_property(
        TARGET hipfort::hipfort
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES hipfort::hipfort-${_alias_platform}
      )
    elseif(TARGET hipfort::hip)
      add_library(hipfort::hipfort INTERFACE IMPORTED)
      set_property(
        TARGET hipfort::hipfort
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES hipfort::hip
      )
    endif()
  endif()
endmacro()

# =============================================================================
# Validate prerequisites
# =============================================================================

if(NOT CMAKE_Fortran_COMPILER)
  message(FATAL_ERROR "HipfortDependency: Fortran compiler not found. "
                      "Call project(... LANGUAGES Fortran) before including this module."
  )
endif()

if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
  enable_language(C)
endif()

if(NOT HIP_PLATFORM AND DEFINED ENV{HIP_PLATFORM})
  set(HIP_PLATFORM "$ENV{HIP_PLATFORM}")
endif()
if(HIP_PLATFORM STREQUAL "nvidia")
  set(_hipfort_platform "nvptx")
elseif(HIP_PLATFORM STREQUAL "amd")
  set(_hipfort_platform "amdgcn")
else()
  message(FATAL_ERROR "HipfortDependency: set HIP_PLATFORM to amd or nvidia")
endif()
set(_hipfort_platform_target hipfort::hipfort-${_hipfort_platform})

# =============================================================================
# ROCm detection
# =============================================================================

set(HIPFORT_ROCM_ROOT
    ""
    CACHE PATH "ROCm installation root; auto-detected if empty"
)
mark_as_advanced(HIPFORT_ROCM_ROOT)

if(NOT HIPFORT_ROCM_ROOT)
  block(PROPAGATE HIPFORT_ROCM_ROOT)
  if(DEFINED ROCM_PATH AND IS_DIRECTORY "${ROCM_PATH}")
    set(HIPFORT_ROCM_ROOT
        "${ROCM_PATH}"
        CACHE PATH "" FORCE
    )
  elseif(NOT HIPFORT_ROCM_ROOT)
    foreach(_path IN LISTS CMAKE_PREFIX_PATH)
      if(_path MATCHES "[Rr][Oo][Cc][Mm]" AND IS_DIRECTORY "${_path}")
        set(HIPFORT_ROCM_ROOT
            "${_path}"
            CACHE PATH "" FORCE
        )
        break()
      endif()
    endforeach()
  endif()
  if(NOT HIPFORT_ROCM_ROOT AND IS_DIRECTORY "/opt/rocm")
    set(HIPFORT_ROCM_ROOT
        "/opt/rocm"
        CACHE PATH "" FORCE
    )
  endif()
  endblock()
endif()

if(HIP_PLATFORM STREQUAL "amd" AND HIPFORT_REQUIRE_ROCM AND NOT HIPFORT_ROCM_ROOT)
  message(FATAL_ERROR "HipfortDependency: ROCm not found. Options:\n" "  - Set ROCM_PATH to your ROCm installation\n"
                      "  - Add ROCm to CMAKE_PREFIX_PATH\n" "  - Set HIPFORT_REQUIRE_ROCM=OFF if ROCm is not needed"
  )
endif()

if(HIPFORT_ROCM_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${HIPFORT_ROCM_ROOT}" "${HIPFORT_ROCM_ROOT}/lib/cmake/hipfort")
  message(VERBOSE "HipfortDependency: ROCm root = ${HIPFORT_ROCM_ROOT}")
endif()

# =============================================================================
# ROCm version detection (for auto-tagging)
# =============================================================================

# Only set default if not already defined (preserves user-supplied or previously detected value)
if(NOT DEFINED HIPFORT_ROCM_VERSION)
  set(HIPFORT_ROCM_VERSION
      ""
      CACHE STRING "Detected ROCm version"
  )
endif()
mark_as_advanced(HIPFORT_ROCM_VERSION)

if(HIPFORT_ROCM_ROOT AND NOT HIPFORT_ROCM_VERSION)
  set(_version_files "${HIPFORT_ROCM_ROOT}/.info/version" "${HIPFORT_ROCM_ROOT}/lib/cmake/hip/hip-config-version.cmake")
  foreach(_vfile IN LISTS _version_files)
    if(EXISTS "${_vfile}")
      file(READ "${_vfile}" _ver_content)
      if(_ver_content MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(HIPFORT_ROCM_VERSION
            "${CMAKE_MATCH_1}"
            CACHE STRING "" FORCE
        )
        break()
      endif()
    endif()
  endforeach()

  if(HIPFORT_ROCM_VERSION)
    message(VERBOSE "HipfortDependency: ROCm version = ${HIPFORT_ROCM_VERSION}")
  endif()
endif()

if(NOT HIPFORT_GIT_TAG)
  if(HIP_PLATFORM STREQUAL "nvidia")
    # Keep CUDA builds on the supported flat nvptx archive/module layout;
    # the version of a separate HIP header installation need not match it.
    set(HIPFORT_GIT_TAG "rocm-6.2.0" CACHE STRING "Git tag/branch for hipfort" FORCE)
  elseif(HIPFORT_ROCM_VERSION)
    set(HIPFORT_GIT_TAG
        "rocm-${HIPFORT_ROCM_VERSION}"
        CACHE STRING "Git tag/branch; auto-detected from ROCm if empty" FORCE
    )
  else()
    set(HIPFORT_GIT_TAG
        "rocm-6.2.0"
        CACHE STRING "Git tag/branch; auto-detected from ROCm if empty" FORCE
    )
    if(HIPFORT_REQUIRE_ROCM)
      message(WARNING "HipfortDependency: Could not detect ROCm version, using ${HIPFORT_GIT_TAG}")
    endif()
  endif()
endif()

# =============================================================================
# Ensure we have a valid base directory for dependencies
# =============================================================================

if(NOT FETCHCONTENT_BASE_DIR)
  set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/.deps")
  message(VERBOSE "HipfortDependency: FETCHCONTENT_BASE_DIR not set, using ${FETCHCONTENT_BASE_DIR}")
endif()

set(_hipfort_install_dir "${FETCHCONTENT_BASE_DIR}/hipfort-install")
set(_hipfort_cached_config "${_hipfort_install_dir}/lib/cmake/hipfort/hipfort-config.cmake")
set(_hipfort_stamp_file "${_hipfort_install_dir}/quop_hipfort_stamp.cmake")
set(_hipfort_cache_inputs
    HIPFORT_GIT_TAG HIPFORT_GIT_REPO HIPFORT_SOURCE_URL HIPFORT_SOURCE_URL_HASH
    HIPFORT_COMPILER CMAKE_Fortran_COMPILER CMAKE_Fortran_COMPILER_VERSION CMAKE_Fortran_FLAGS
    HIP_PLATFORM HIPFORT_ROCM_ROOT CUDA_PATH CUDAToolkit_ROOT CMAKE_PREFIX_PATH
    OFFLOAD_ARCH GPU_TARGETS CMAKE_BUILD_TYPE
)

function(_hipfort_write_cache_stamp stamp_file)
  file(WRITE "${stamp_file}" "set(_QUOP_HIPFORT_STAMP_VERSION 1)\n")
  foreach(_input IN LISTS _hipfort_cache_inputs)
    file(APPEND "${stamp_file}" "set(_QUOP_${_input} [==[${${_input}}]==])\n")
  endforeach()
endfunction()

function(_hipfort_validate_cache_stamp)
  if(NOT EXISTS "${_hipfort_stamp_file}")
    message(FATAL_ERROR "HipfortDependency: Cached hipfort has no toolchain stamp. "
                        "Use a new install prefix or rerun the installer with --veryclean.")
  endif()
  include("${_hipfort_stamp_file}")
  set(_changed "")
  if(NOT _QUOP_HIPFORT_STAMP_VERSION EQUAL 1)
    list(APPEND _changed "cache stamp format")
  endif()
  foreach(_input IN LISTS _hipfort_cache_inputs)
    if(NOT "${_QUOP_${_input}}" STREQUAL "${${_input}}")
      list(APPEND _changed "${_input}")
    endif()
  endforeach()
  if(_changed)
    message(FATAL_ERROR "HipfortDependency: Cached hipfort inputs changed: ${_changed}. "
                        "Use a new install prefix or rerun the installer with --veryclean.")
  endif()
endfunction()

set(_hipfort_skip_system FALSE)
if(CMAKE_MESSAGE_LOG_LEVEL STREQUAL "VERBOSE")
  message(VERBOSE "HipfortDependency: expecting cached config at ${_hipfort_cached_config}")
endif()

# =============================================================================
# Check for previously-built hipfort in .deps (persists across clean builds) This is checked FIRST - if a cached build
# exists, we use it or rebuild it
# =============================================================================

set(_hipfort_use_cached TRUE)
if(HIPFORT_PATH AND NOT HIPFORT_PATH STREQUAL _hipfort_install_dir)
  set(_hipfort_use_cached FALSE)
  if(hipfort_DIR MATCHES "^${_hipfort_install_dir}/")
    unset(hipfort_DIR CACHE)
    unset(hipfort_DIR)
  endif()
elseif(hipfort_DIR AND NOT hipfort_DIR MATCHES "^${_hipfort_install_dir}/")
  set(_hipfort_use_cached FALSE)
endif()

if((EXISTS "${_hipfort_cached_config}"
    OR (HIP_PLATFORM STREQUAL "nvidia" AND EXISTS "${_hipfort_install_dir}/lib/libhipfort-nvptx.a"))
   AND _hipfort_use_cached)
  message(STATUS "Found cached hipfort in ${_hipfort_install_dir}")
  _hipfort_validate_cache_stamp()

  set(hipfort_DIR "${_hipfort_install_dir}/lib/cmake/hipfort")
  if(HIP_PLATFORM STREQUAL "nvidia")
    _hipfort_import_nvptx("${_hipfort_install_dir}" hipfort_FOUND)
    if(NOT hipfort_FOUND)
      find_package(hipfort CONFIG QUIET PATHS "${_hipfort_install_dir}" NO_DEFAULT_PATH)
    endif()
  else()
    find_package(hipfort CONFIG QUIET COMPONENTS hip
                 PATHS "${_hipfort_install_dir}" "${hipfort_DIR}" NO_DEFAULT_PATH)
  endif()

  if(hipfort_FOUND)
    set(_hipfort_skip_system TRUE)
    set(_use_cached TRUE)

    if(NOT TARGET ${_hipfort_platform_target})
      message(FATAL_ERROR "HipfortDependency: Cached hipfort does not provide ${_hipfort_platform_target}. "
                          "Use a new install prefix or rerun the installer with --veryclean.")
    endif()

    if(HIPFORT_VERIFY_MOD_COMPAT)
      _hipfort_check_mod_compatibility(${_hipfort_platform_target} _cached_mod_compat_ok)

      if(NOT _cached_mod_compat_ok)
        message(
          FATAL_ERROR
            "HipfortDependency: Cached hipfort .mod files are incompatible with "
            "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}.\n"
            "Either delete ${_hipfort_install_dir} (and ${FETCHCONTENT_BASE_DIR}/hipfort-ep) to force a rebuild, "
            "or configure with -DHIPFORT_VERIFY_MOD_COMPAT=OFF to bypass this check.\n"
            "check output:\n${_cached_mod_compat_ok_LOG}"
        )
      endif()
    endif()

    if(_use_cached)
      message(STATUS "Using cached hipfort")
      _hipfort_ensure_alias()
      set(HIPFORT_PROVIDER
          "cached"
          CACHE INTERNAL ""
      )
      set(HIPFORT_VERSION
          "${hipfort_VERSION}"
          CACHE INTERNAL ""
      )
      return()
    endif()
  endif()
endif()

# =============================================================================
# Try system hipfort (only if no cached build exists) We skip this if a cached build was found (even if incompatible) to
# avoid creating conflicting targets - we'll rebuild instead
# =============================================================================

if(NOT _hipfort_skip_system)
  # Check user-specified path first (backward compatibility with HIPFORT_PATH)
  if(HIPFORT_PATH)
    set(_hf_path "${HIPFORT_PATH}")
  elseif(DEFINED ENV{HIPFORT_PATH})
    set(_hf_path "$ENV{HIPFORT_PATH}")
  endif()
  if(HIP_PLATFORM STREQUAL "nvidia")
    set(_hipfort_prefixes "${_hf_path}")
    if(hipfort_DIR)
      get_filename_component(_hipfort_config_prefix "${hipfort_DIR}/../../.." ABSOLUTE)
      list(APPEND _hipfort_prefixes "${_hipfort_config_prefix}")
    endif()
    list(APPEND _hipfort_prefixes ${CMAKE_PREFIX_PATH} "${HIPFORT_ROCM_ROOT}" /usr/local /usr)
    foreach(_prefix IN LISTS _hipfort_prefixes)
      if(_prefix)
        _hipfort_import_nvptx("${_prefix}" hipfort_FOUND)
        if(hipfort_FOUND)
          break()
        endif()
      endif()
    endforeach()
    if(NOT hipfort_FOUND)
      # Request no ROCm components: legacy packages otherwise load AMD runtime
      # libraries before we can inspect their platform-specific targets.
      find_package(hipfort CONFIG QUIET PATHS ${_hipfort_prefixes})
    endif()
  elseif(_hf_path)
    find_package(
      hipfort CONFIG QUIET COMPONENTS hip
      PATHS "${_hf_path}" "${_hf_path}/lib/cmake" "${_hf_path}/lib/cmake/hipfort"
      NO_DEFAULT_PATH
    )
  endif()

  # Request 'hip' component - required for hipfort::hip target
  if(NOT hipfort_FOUND AND HIP_PLATFORM STREQUAL "amd")
    find_package(hipfort CONFIG QUIET COMPONENTS hip)
  endif()

  if(hipfort_FOUND)
    message(VERBOSE "HipfortDependency: Found system hipfort at ${hipfort_DIR}")

    # Fortran modules are platform-specific even when a package exports both.
    if(TARGET ${_hipfort_platform_target})
      set(_hipfort_target ${_hipfort_platform_target})
    else()
      message(STATUS "HipfortDependency: Package does not provide ${_hipfort_platform_target}; will fetch instead")
      set(hipfort_FOUND FALSE)
    endif()
  endif()

  if(hipfort_FOUND AND HIPFORT_VERIFY_MOD_COMPAT)
    _hipfort_check_mod_compatibility(${_hipfort_target} _mod_compat_ok)

    if(_mod_compat_ok)
      message(VERBOSE "HipfortDependency: Module compatibility check passed")
    else()
      message(STATUS "HipfortDependency: System hipfort .mod files incompatible with "
                     "${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION} "
                     "(likely built with different compiler); will fetch and build instead"
      )
      message(VERBOSE "Module compatibility check output:\n${_mod_compat_ok_LOG}")
      set(hipfort_FOUND FALSE)
    endif()
  endif()

  if(hipfort_FOUND)
    _hipfort_ensure_alias()

    set(HIPFORT_PROVIDER
        "system"
        CACHE INTERNAL ""
    )
    set(HIPFORT_VERSION
        "${hipfort_VERSION}"
        CACHE INTERNAL ""
    )

    message(STATUS "Found hipfort ${hipfort_VERSION} (system)")
    return()
  endif()
endif()

# =============================================================================
# Fetch and build hipfort
# =============================================================================

if(NOT HIPFORT_AUTO_FETCH)
  message(
    FATAL_ERROR
      "HipfortDependency: hipfort not found and HIPFORT_AUTO_FETCH is OFF.\n" "Options:\n"
      "  - Set hipfort_DIR to existing installation\n" "  - Add hipfort location to CMAKE_PREFIX_PATH\n"
      "  - Set HIPFORT_AUTO_FETCH=ON to download automatically"
  )
endif()

message(STATUS "Fetching hipfort ${HIPFORT_GIT_TAG}...")

if(HIPFORT_SOURCE_URL)
  if(NOT HIPFORT_SOURCE_URL MATCHES "^[a-zA-Z][a-zA-Z0-9+.-]*://")
    cmake_path(ABSOLUTE_PATH HIPFORT_SOURCE_URL OUTPUT_VARIABLE _source_url)
    if(NOT EXISTS "${_source_url}")
      message(FATAL_ERROR "HipfortDependency: Source path not found: ${_source_url}")
    endif()
  else()
    set(_source_url "${HIPFORT_SOURCE_URL}")
  endif()

  set(_fc_args URL "${_source_url}")
  if(HIPFORT_SOURCE_URL_HASH)
    list(APPEND _fc_args URL_HASH "${HIPFORT_SOURCE_URL_HASH}")
  endif()
else()
  set(_fc_args
      GIT_REPOSITORY
      "${HIPFORT_GIT_REPO}"
      GIT_TAG
      "${HIPFORT_GIT_TAG}"
      GIT_SHALLOW
      TRUE
      GIT_PROGRESS
      TRUE
  )
endif()

include(ExternalProject)

# Build hipfort compiler flags
set(_hipfort_flags "${CMAKE_Fortran_FLAGS}")
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  string(APPEND _hipfort_flags " -ffree-form -cpp -ffree-line-length-none")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(APPEND _hipfort_flags " -free -fpp")
endif()

# Build optional arguments list - only add if defined
set(_optional_args "")

# GPU_TARGETS
if(GPU_TARGETS)
  string(REPLACE ";" "|" _hipfort_gpu_targets "${GPU_TARGETS}")
  list(APPEND _optional_args "-DGPU_TARGETS=${_hipfort_gpu_targets}")
elseif(OFFLOAD_ARCH)
  string(REPLACE ";" "|" _hipfort_gpu_targets "${OFFLOAD_ARCH}")
  list(APPEND _optional_args "-DGPU_TARGETS=${_hipfort_gpu_targets}")
endif()

# AR/RANLIB
if(CMAKE_Fortran_COMPILER_AR)
  list(APPEND _optional_args "-DHIPFORT_AR=${CMAKE_Fortran_COMPILER_AR}")
endif()
if(CMAKE_Fortran_COMPILER_RANLIB)
  list(APPEND _optional_args "-DHIPFORT_RANLIB=${CMAKE_Fortran_COMPILER_RANLIB}")
endif()

# ROCm path - hipfort needs this to find HIP headers
if(HIPFORT_ROCM_ROOT)
  list(APPEND _optional_args "-DROCM_PATH=${HIPFORT_ROCM_ROOT}")
endif()

# Pass HIP_PLATFORM to ensure ExternalProject builds the correct platform library
list(APPEND _optional_args "-DHIP_PLATFORM=${HIP_PLATFORM}")
if(HIP_PLATFORM STREQUAL "nvidia" AND HIPFORT_GIT_TAG MATCHES "^rocm-[67]\\.")
  # These releases always build both Fortran archives. Their optional component
  # exports are AMD-only and may try to load incompatible ROCm math packages;
  # none are needed to compile or use the nvptx archive with CUDA directly.
  foreach(_component hip rocblas hipblas rocfft hipfft rocrand hiprand rocsolver hipsolver rocsparse hipsparse)
    list(APPEND _optional_args "-DCMAKE_DISABLE_FIND_PACKAGE_${_component}=ON")
  endforeach()
endif()
foreach(_input CMAKE_PREFIX_PATH CUDA_PATH CUDAToolkit_ROOT CMAKE_C_COMPILER CMAKE_CXX_COMPILER
               CMAKE_Fortran_COMPILER)
  if(DEFINED ${_input} AND NOT "${${_input}}" STREQUAL "")
    string(REPLACE ";" "|" _value "${${_input}}")
    list(APPEND _optional_args "-D${_input}=${_value}")
  endif()
endforeach()

externalproject_add(
  hipfort_external
  ${_fc_args}
  PREFIX "${FETCHCONTENT_BASE_DIR}/hipfort-ep"
  INSTALL_DIR "${_hipfort_install_dir}"
  LIST_SEPARATOR "|"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             -DHIPFORT_INSTALL_DIR=<INSTALL_DIR>
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DHIPFORT_COMPILER=${HIPFORT_COMPILER}
             "-DHIPFORT_COMPILER_FLAGS=${_hipfort_flags}"
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DBUILD_TESTING=OFF
             ${_optional_args}
  BUILD_BYPRODUCTS "${_hipfort_install_dir}/lib/libhipfort-${_hipfort_platform}.a"
)

set(_hipfort_generated_stamp "${FETCHCONTENT_BASE_DIR}/hipfort-cache-stamp.cmake")
_hipfort_write_cache_stamp("${_hipfort_generated_stamp}")
ExternalProject_Add_Step(
  hipfort_external
  write_cache_stamp
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_hipfort_generated_stamp}" "${_hipfort_stamp_file}"
  DEPENDEES install
)

# Create include directories now so CMake doesn't complain about non-existent paths (ExternalProject builds during build
# step, not configure step)
set(_hipfort_include_dir "${_hipfort_install_dir}/include/hipfort/${_hipfort_platform}")
set(_hipfort_include_dir_base "${_hipfort_install_dir}/include")
set(_hipfort_imported_location "${_hipfort_install_dir}/lib/libhipfort-${_hipfort_platform}.a")
file(MAKE_DIRECTORY "${_hipfort_include_dir}")

# Create or update the platform-specific target (e.g., hipfort::hipfort-amdgcn) If the target already exists (from an
# incompatible system hipfort), update it to point to our freshly-built version instead
if(TARGET ${_hipfort_platform_target})
  message(STATUS "Updating hipfort::hipfort-${_hipfort_platform} target")
else()
  add_library(${_hipfort_platform_target} STATIC IMPORTED GLOBAL)
endif()

add_dependencies(${_hipfort_platform_target} hipfort_external)
set_target_properties(
  ${_hipfort_platform_target}
  PROPERTIES IMPORTED_LOCATION "${_hipfort_imported_location}"
             INTERFACE_INCLUDE_DIRECTORIES "${_hipfort_include_dir};${_hipfort_include_dir_base}"
)
if(TARGET quop_gpu_runtime)
  set_property(TARGET ${_hipfort_platform_target} PROPERTY INTERFACE_LINK_LIBRARIES quop_gpu_runtime)
endif()

foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(
    ${_hipfort_platform_target}
    PROPERTIES IMPORTED_LOCATION_${_cfg} "${_hipfort_imported_location}"
               IMPORTED_LINK_INTERFACE_LANGUAGES_${_cfg} "Fortran"
  )
endforeach()

# Create hipfort::hipfort as an alias-like target that links to the platform target This provides a convenient
# platform-agnostic target name
if(NOT TARGET hipfort::hipfort)
  add_library(hipfort::hipfort INTERFACE IMPORTED GLOBAL)
  set_property(
    TARGET hipfort::hipfort
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES hipfort::hipfort-${_hipfort_platform}
  )
endif()

set(HIPFORT_PROVIDER
    "fetched"
    CACHE INTERNAL ""
)
set(HIPFORT_VERSION
    "${HIPFORT_GIT_TAG}"
    CACHE INTERNAL ""
)

message(STATUS "hipfort ${HIPFORT_GIT_TAG} will be installed to ${_hipfort_install_dir}")
