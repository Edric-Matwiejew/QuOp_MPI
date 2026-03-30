#!/usr/bin/env bash
# =============================================================================
# Profile: ubuntu-24
#
# Ubuntu 24.04 LTS workstation or server.
# Supports both 'mpi' (CPU-only) and 'wavefront' (GPU / HIP) backends.
#
# Prerequisites (apt):
#   sudo apt install gcc g++ gfortran cmake pkg-config \
#       libopenmpi-dev openmpi-bin \
#       libhdf5-openmpi-dev libfftw3-dev libfftw3-mpi-dev \
#       python3-dev python3-venv python3-pip
#
# For wavefront (GPU) support, install ROCm from AMD's repository:
#   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
#   The profile expects ROCm at /opt/rocm (default install path).
#
# >>> SITE-SPECIFIC: adjust compilers or paths via config.toml or env vars. <<<
# =============================================================================
apply_profile_defaults

# ---- Build type defaults -----------------------------------------------------
MPI_BUILD_TYPE="${MPI_BUILD_TYPE:-${CFG_MPI_BUILD_TYPE:-Release}}"
WAVEFRONT_BUILD_TYPE="${WAVEFRONT_BUILD_TYPE:-${CFG_WAVEFRONT_BUILD_TYPE:-Release}}"
OFFLOAD_ARCH="${OFFLOAD_ARCH:-${CFG_OFFLOAD_ARCH:-gfx90a}}"

# ---- Site-specific discovery extensions --------------------------------------
# These extend the shared helpers in common.sh with Ubuntu-specific fallbacks.

profile_find_rocm_path() {
    # Try the shared helper first, then fall back to the default ROCm install
    # location on Ubuntu.
    find_rocm_path && return 0
    if [[ -d /opt/rocm ]]; then
        echo /opt/rocm
        return 0
    fi
    return 1
}

profile_find_hdf5_dir() {
    # Ubuntu installs parallel HDF5 under /usr with openmpi flavour.
    find_hdf5_dir && return 0
    [[ -f /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so ]] && echo /usr
}

profile_find_fftw_root() {
    find_fftw_root && return 0
    [[ -f /usr/lib/x86_64-linux-gnu/libfftw3.so ]] && echo /usr
}

# Detect the Debian/Ubuntu multiarch library directory.
# Returns e.g. /usr/lib/x86_64-linux-gnu or empty string.
profile_multiarch_libdir() {
    local triplet
    triplet="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || gcc -dumpmachine 2>/dev/null || true)"
    if [[ -n "$triplet" && -d "/usr/lib/${triplet}" ]]; then
        printf '%s\n' "/usr/lib/${triplet}"
    fi
}

# ---- Post-module environment -------------------------------------------------
profile_post_modules_env() {
    export CC="${CC:-$(resolve_command "${CFG_CC:-gcc}")}"
    export CXX="${CXX:-$(resolve_command "${CFG_CXX:-g++}")}"
    export FC="${FC:-$(resolve_command "${CFG_FC:-gfortran}")}"

    require_command "$CC" || return 1
    require_command "$CXX" || return 1
    require_command "$FC" || return 1

    export HDF5_DIR="${HDF5_DIR:-$(profile_find_hdf5_dir || true)}"
    if [[ -z "${HDF5_DIR:-}" ]]; then
        echo "Error: could not locate parallel HDF5; install libhdf5-openmpi-dev or set HDF5_DIR"
        return 1
    fi

    export FFTW_ROOT="${FFTW_ROOT:-$(profile_find_fftw_root || true)}"
    if [[ -z "${FFTW_ROOT:-}" ]]; then
        echo "Error: could not locate FFTW; install libfftw3-dev or set FFTW_ROOT"
        return 1
    fi

    if [[ "$BACKEND" == "wavefront" ]]; then
        export ROCM_PATH="${ROCM_PATH:-$(profile_find_rocm_path || true)}"
        if [[ -z "${ROCM_PATH:-}" ]]; then
            echo "Error: could not locate ROCm; install from https://rocm.docs.amd.com or set ROCM_PATH"
            return 1
        fi
        export PATH="${ROCM_PATH}/bin${PATH:+:${PATH}}"
        export SHAFFT_PATH="${SHAFFT_PATH:-$(resolve_shafft_path || true)}"

        local -a extra_ld_paths=("${ROCM_PATH}/lib")
        local extra_ld_path
        extra_ld_path="$(IFS=:; echo "${extra_ld_paths[*]}")"
        export LD_LIBRARY_PATH="${extra_ld_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

# ---- mpi4py ------------------------------------------------------------------
profile_install_mpi4py() {
    info "Installing mpi4py via pip"
    python -m pip install --no-cache-dir mpi4py
}

# ---- h5py --------------------------------------------------------------------
profile_install_h5py() {
    info "Building h5py with parallel HDF5 from ${HDF5_DIR}"

    local mpi_inc=""
    if command -v mpicc >/dev/null 2>&1; then
        mpi_inc="$(mpicc --showme:compile 2>/dev/null || true)"
    fi

    # Ubuntu installs the parallel HDF5 libraries under a multiarch
    # subdirectory (e.g. /usr/lib/x86_64-linux-gnu/hdf5/openmpi/) rather
    # than $HDF5_DIR/lib.  Tell h5py to use pkg-config so it finds the
    # correct include/library paths automatically.
    local hdf5_pkg=""
    if pkg-config --exists hdf5-openmpi 2>/dev/null; then
        hdf5_pkg="hdf5-openmpi"
    elif pkg-config --exists hdf5 2>/dev/null; then
        hdf5_pkg="hdf5"
    fi

    if [[ -n "$hdf5_pkg" ]]; then
        (
            unset HDF5_DIR
            CC="${CC}" \
            CFLAGS="${CFLAGS:-} ${mpi_inc}" \
            HDF5_MPI="ON" \
            HDF5_PKGCONFIG_NAME="${hdf5_pkg}" \
                python -m pip install --no-binary=h5py --no-cache-dir --force-reinstall h5py
        )
    else
        (
            unset HDF5_PKGCONFIG_NAME
            CC="${CC}" \
            CFLAGS="${CFLAGS:-} ${mpi_inc}" \
            HDF5_MPI="ON" \
            HDF5_DIR="${HDF5_DIR}" \
                python -m pip install --no-binary=h5py --no-cache-dir --force-reinstall h5py
        )
    fi
}

# ---- CMake flags -------------------------------------------------------------
profile_cmake_args() {
    local cmake_prefix_path
    local shafft_path

    # On multiarch systems (Debian/Ubuntu) FFTW libraries live under
    # /usr/lib/<triplet>/ rather than ${FFTW_ROOT}/lib/.  The downloaded
    # FindFFTW module uses NO_DEFAULT_PATH when FFTW_ROOT is set, so it
    # only checks ${FFTW_ROOT}/lib and ${FFTW_ROOT}/lib64.  If the MPI
    # sub-library is not there we must *omit* FFTW_ROOT and instead pass
    # CMAKE_LIBRARY_PATH so FindFFTW falls through to its pkg-config path.
    local fftw_root_arg="-DFFTW_ROOT=${FFTW_ROOT}"
    local fftw_libdir_arg=""
    if [[ ! -f "${FFTW_ROOT}/lib/libfftw3_mpi.so" && ! -f "${FFTW_ROOT}/lib64/libfftw3_mpi.so" ]]; then
        # Multiarch layout -- do not set FFTW_ROOT so FindFFTW uses pkg-config.
        fftw_root_arg=""
        local ma_libdir
        ma_libdir="$(profile_multiarch_libdir)"
        if [[ -n "$ma_libdir" ]]; then
            fftw_libdir_arg="-DCMAKE_LIBRARY_PATH=${ma_libdir}"
        fi
    fi

    if [[ "$BACKEND" == "wavefront" ]]; then
        shafft_path="$(resolve_shafft_path || true)"
        cmake_prefix_path="$(join_by_delimiter ';' \
            "${shafft_path}" "${ROCM_PATH}" "${HDF5_DIR}" "${FFTW_ROOT}")"
        cat <<ARGS
-DCMAKE_C_COMPILER=${CC}
-DCMAKE_Fortran_COMPILER=${FC}
-DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc
-DHDF5_ROOT=${HDF5_DIR}
-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE
-DHDF5_PREFER_PARALLEL=TRUE
${fftw_root_arg}
${fftw_libdir_arg}
-DCMAKE_PREFIX_PATH=${cmake_prefix_path}
-DFETCHCONTENT_BASE_DIR=${FETCHCONTENT_BASE_DIR}
-DWAVEFRONT_BACKEND=ON
-DSHAFFT_PATH=${shafft_path}
-DOFFLOAD_ARCH=${OFFLOAD_ARCH}
-DROCM_PATH=${ROCM_PATH}
-DGPU_AWARE_MPI=OFF
ARGS
    else
        cmake_prefix_path="$(join_by_delimiter ';' "${HDF5_DIR}" "${FFTW_ROOT}")"
        cat <<ARGS
-DCMAKE_C_COMPILER=${CC}
-DCMAKE_Fortran_COMPILER=${FC}
-DHDF5_ROOT=${HDF5_DIR}
-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE
-DHDF5_PREFER_PARALLEL=TRUE
${fftw_root_arg}
${fftw_libdir_arg}
-DCMAKE_PREFIX_PATH=${cmake_prefix_path}
-DWAVEFRONT_BACKEND=OFF
-DWITH_SHAFFT=OFF
-DGPU_AWARE_MPI=OFF
ARGS
    fi
}
