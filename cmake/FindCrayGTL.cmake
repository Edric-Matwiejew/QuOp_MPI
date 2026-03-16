#[=======================================================================[.rst:
FindCrayGTL
---------------

Finds the Cray GPU Transport Layer (GTL) library for GPU-aware MPI.

This module supports both AMD and NVIDIA GPUs on Cray systems. The GTL library
enables GPU-aware MPI operations (direct GPU memory transfers) with Cray MPICH.

Result Variables
^^^^^^^^^^^^^^^^
``CrayGTL_FOUND``
  True if GTL was found
``CrayGTL_LIBRARIES``
  Libraries to link
``CrayGTL_TYPE``
  Architecture type (e.g., "amd_gfx90a", "nvidia80")

Imported Targets
^^^^^^^^^^^^^^^^
``CrayGTL::CrayGTL``
  The GTL library target

Cache Variables
^^^^^^^^^^^^^^^
``CRAY_GTL_LIBRARY``
  Path to the GTL library

Hints
^^^^^
Set ``GTL_ARCH`` to override automatic architecture detection.

#]=======================================================================]

include_guard(GLOBAL)

#=============================================================================
# Check if this is a Cray system
#=============================================================================

if(NOT DEFINED ENV{CRAY_CPU_TARGET})
  if(NOT CrayGTL_FIND_QUIETLY)
    message(STATUS "FindCrayGTL: Not a Cray system (CRAY_CPU_TARGET not set)")
  endif()
  set(CrayGTL_FOUND FALSE)
  return()
endif()

#=============================================================================
# Find available GTL architectures
#=============================================================================

# Try to find GTL architecture from PE_MPICH_GTL_DIR_* environment variables
# Common architectures on Cray systems:
#   - amd_gfx90a (MI250X)
#   - amd_gfx908 (MI100)
#   - nvidia80 (A100)
#   - nvidia90 (H100)

set(_gtl_found_archs "")

# Check common AMD architectures
foreach(_arch IN ITEMS "amd_gfx90a" "amd_gfx908" "amd_gfx942" "amd_gfx940")
  if(DEFINED ENV{PE_MPICH_GTL_DIR_${_arch}})
    list(APPEND _gtl_found_archs "${_arch}")
  endif()
endforeach()

# Check common NVIDIA architectures
foreach(_arch IN ITEMS "nvidia80" "nvidia90" "nvidia70")
  if(DEFINED ENV{PE_MPICH_GTL_DIR_${_arch}})
    list(APPEND _gtl_found_archs "${_arch}")
  endif()
endforeach()

if(NOT _gtl_found_archs)
  if(NOT CrayGTL_FIND_QUIETLY)
    message(STATUS "FindCrayGTL: No GTL support detected (no PE_MPICH_GTL_DIR_* variables found)")
  endif()
  set(CrayGTL_FOUND FALSE)
  return()
endif()

message(VERBOSE "FindCrayGTL: Available GTL architectures: ${_gtl_found_archs}")

#=============================================================================
# Determine target GPU architecture
#=============================================================================

if(DEFINED GTL_ARCH AND GTL_ARCH)
  # User override
  set(_gtl_arch "${GTL_ARCH}")
  message(VERBOSE "FindCrayGTL: Using user-specified GTL_ARCH=${_gtl_arch}")
elseif(CMAKE_HIP_ARCHITECTURES)
  # AMD GPU - use CMake's HIP architecture
  list(GET CMAKE_HIP_ARCHITECTURES 0 _first_arch)
  set(_gtl_arch "amd_${_first_arch}")
  message(VERBOSE "FindCrayGTL: Detected AMD GPU from CMAKE_HIP_ARCHITECTURES: ${_gtl_arch}")
elseif(CMAKE_CUDA_ARCHITECTURES)
  # NVIDIA GPU - map CUDA compute capability to GTL naming
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _first_arch)
  # GTL uses "nvidia80" for sm_80, "nvidia90" for sm_90, etc.
  string(REGEX REPLACE "^([0-9]+).*" "\\1" _cuda_major "${_first_arch}")
  set(_gtl_arch "nvidia${_cuda_major}")
  message(VERBOSE "FindCrayGTL: Detected NVIDIA GPU from CMAKE_CUDA_ARCHITECTURES: ${_gtl_arch}")
elseif(DEFINED ENV{HIP_PLATFORM})
  # Fall back to HIP_PLATFORM environment variable
  if("$ENV{HIP_PLATFORM}" STREQUAL "nvidia")
    # Use first available nvidia GTL
    foreach(_arch IN LISTS _gtl_found_archs)
      if(_arch MATCHES "^nvidia")
        set(_gtl_arch "${_arch}")
        break()
      endif()
    endforeach()
  elseif("$ENV{HIP_PLATFORM}" STREQUAL "amd")
    # Try ROCM_GPU or HCC_AMDGPU_TARGET (legacy)
    if(DEFINED ENV{ROCM_GPU})
      set(_gtl_arch "amd_$ENV{ROCM_GPU}")
    elseif(DEFINED ENV{HCC_AMDGPU_TARGET})
      set(_gtl_arch "amd_$ENV{HCC_AMDGPU_TARGET}")
    else()
      # Use first available amd GTL
      foreach(_arch IN LISTS _gtl_found_archs)
        if(_arch MATCHES "^amd_")
          set(_gtl_arch "${_arch}")
          break()
        endif()
      endforeach()
    endif()
  endif()
endif()

# Final fallback: use first available GTL architecture
if(NOT _gtl_arch)
  list(GET _gtl_found_archs 0 _gtl_arch)
  message(VERBOSE "FindCrayGTL: Using first available GTL architecture: ${_gtl_arch}")
endif()

#=============================================================================
# Verify the GTL environment variables exist for chosen architecture
#=============================================================================

set(_gtl_dir_env "PE_MPICH_GTL_DIR_${_gtl_arch}")
set(_gtl_lib_env "PE_MPICH_GTL_LIBS_${_gtl_arch}")

if(NOT DEFINED ENV{${_gtl_dir_env}})
  if(NOT CrayGTL_FIND_QUIETLY)
    message(STATUS "FindCrayGTL: GTL not available for architecture '${_gtl_arch}'")
    message(STATUS "  Available GTL architectures: ${_gtl_found_archs}")
  endif()
  set(CrayGTL_FOUND FALSE)
  return()
endif()

#=============================================================================
# Extract library path and name
#=============================================================================

set(_gtl_dir_raw "$ENV{${_gtl_dir_env}}")
set(_gtl_lib_raw "$ENV{${_gtl_lib_env}}")

# Remove linker flags (-L, -l) if present
string(REGEX REPLACE "^-L" "" _gtl_dir "${_gtl_dir_raw}")
string(REGEX REPLACE "^-l" "" _gtl_lib_name "${_gtl_lib_raw}")

# If library name wasn't provided by environment, use vendor-specific defaults
# AMD GPUs use HSA (Heterogeneous System Architecture): libmpi_gtl_hsa.so
# NVIDIA GPUs use CUDA: libmpi_gtl_cuda.so
if(NOT _gtl_lib_name)
  if(_gtl_arch MATCHES "^nvidia")
    set(_gtl_lib_name "mpi_gtl_cuda")
    message(VERBOSE "FindCrayGTL: Using default NVIDIA library name: ${_gtl_lib_name}")
  elseif(_gtl_arch MATCHES "^amd_")
    set(_gtl_lib_name "mpi_gtl_hsa")
    message(VERBOSE "FindCrayGTL: Using default AMD library name: ${_gtl_lib_name}")
  else()
    # Unknown vendor - try both
    set(_gtl_lib_name "mpi_gtl_hsa;mpi_gtl_cuda")
    message(VERBOSE "FindCrayGTL: Unknown vendor, will try: ${_gtl_lib_name}")
  endif()
endif()

#=============================================================================
# Find the library
#=============================================================================

# Build list of search paths
set(_gtl_search_paths ${_gtl_dir})

# Add common Cray MPICH library paths as fallbacks
if(DEFINED ENV{CRAY_MPICH_DIR})
  list(APPEND _gtl_search_paths "$ENV{CRAY_MPICH_DIR}/lib")
  list(APPEND _gtl_search_paths "$ENV{CRAY_MPICH_DIR}/gtl/lib")
endif()

find_library(CRAY_GTL_LIBRARY NAMES ${_gtl_lib_name} PATHS ${_gtl_search_paths} NO_DEFAULT_PATH)

#=============================================================================
# Set result variables and create target
#=============================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CrayGTL
  REQUIRED_VARS CRAY_GTL_LIBRARY
  REASON_FAILURE_MESSAGE "GTL library '${_gtl_lib_name}' not found in ${_gtl_dir}"
)

if(CrayGTL_FOUND)
  set(CrayGTL_LIBRARIES "${CRAY_GTL_LIBRARY}")
  set(CrayGTL_TYPE "${_gtl_arch}")

  if(NOT TARGET CrayGTL::CrayGTL)
    add_library(CrayGTL::CrayGTL UNKNOWN IMPORTED)
    set_target_properties(CrayGTL::CrayGTL PROPERTIES IMPORTED_LOCATION "${CRAY_GTL_LIBRARY}")
  endif()
endif()

mark_as_advanced(CRAY_GTL_LIBRARY)
