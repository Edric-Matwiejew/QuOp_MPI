#!/usr/bin/env bash
# =============================================================================
# Profile: macos
#
# macOS workstation with dependencies installed via Homebrew.
# Assumes open-mpi (or mpich), hdf5-mpi, fftw, and gcc are available through
# Homebrew.  Supports only the 'mpi' backend (no GPU / wavefront).
#
# Compiler defaults (gcc-15 / gfortran-15) can be overridden in config.toml
# or via environment variables CC, CXX, FC.
# =============================================================================
apply_profile_defaults

# ---- Homebrew helpers -------------------------------------------------------

_brew_prefix() {
    # Return the Homebrew prefix, honouring an explicit override.
    if [[ -n "${HOMEBREW_PREFIX:-}" ]]; then
        printf '%s\n' "$HOMEBREW_PREFIX"
        return 0
    fi
    if [[ -n "${CFG_HOMEBREW_PREFIX:-}" ]]; then
        printf '%s\n' "$CFG_HOMEBREW_PREFIX"
        return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        brew --prefix
        return 0
    fi
    # Sensible defaults for Apple Silicon / Intel Macs.
    if [[ -d /opt/homebrew ]]; then
        printf '/opt/homebrew\n'
    else
        printf '/usr/local\n'
    fi
}

_brew_formula_prefix() {
    local formula="$1"
    local prefix
    prefix="$(_brew_prefix)"
    if [[ -d "$prefix/opt/$formula" ]]; then
        printf '%s/opt/%s\n' "$prefix" "$formula"
        return 0
    fi
    echo "Error: Homebrew formula '$formula' not found at $prefix/opt/$formula" >&2
    return 1
}

# ---- Post-module environment ------------------------------------------------
profile_post_modules_env() {
    local prefix
    prefix="$(_brew_prefix)"

    # Compilers -- default to Homebrew GCC; override via config or env.
    export CC="${CC:-${CFG_CC:-gcc-15}}"
    export CXX="${CXX:-${CFG_CXX:-g++-15}}"
    export FC="${FC:-${CFG_FC:-gfortran-15}}"

    # Resolve full paths when the compiler lives under the Homebrew prefix.
    for var in CC CXX FC; do
        local val="${!var}"
        if ! command -v "$val" >/dev/null 2>&1; then
            if [[ -x "$prefix/bin/$val" ]]; then
                export "$var=$prefix/bin/$val"
            fi
        fi
    done

    # Validate that the resolved compilers are actually available.
    require_command "$CC" || return 1
    require_command "$CXX" || return 1
    require_command "$FC" || return 1

    # HDF5 (parallel)
    local hdf5_formula="${CFG_HOMEBREW_HDF5_FORMULA:-hdf5-mpi}"
    export HDF5_DIR="${HDF5_DIR:-$(_brew_formula_prefix "$hdf5_formula")}"
    if [[ -z "${HDF5_DIR:-}" ]]; then
        echo "Error: could not locate parallel HDF5; install with: brew install $hdf5_formula"
        return 1
    fi

    # FFTW
    local fftw_formula="${CFG_HOMEBREW_FFTW_FORMULA:-fftw}"
    export FFTW_ROOT="${FFTW_ROOT:-$(_brew_formula_prefix "$fftw_formula")}"
    if [[ -z "${FFTW_ROOT:-}" ]]; then
        echo "Error: could not locate FFTW; install with: brew install $fftw_formula"
        return 1
    fi

    # Ensure Homebrew HDF5 tools are on PATH (needed by h5py build).
    export PATH="${HDF5_DIR}/bin:$PATH"
}

# ---- mpi4py -----------------------------------------------------------------
profile_install_mpi4py() {
    info "Installing mpi4py via pip"
    python -m pip install mpi4py
}

# ---- h5py -------------------------------------------------------------------
profile_install_h5py() {
    info "Building h5py with parallel HDF5 from ${HDF5_DIR}"

    # Resolve the MPI include directory so that gcc can find mpi.h.
    local mpi_inc=""
    if command -v mpicc >/dev/null 2>&1; then
        mpi_inc="$(mpicc --showme:compile 2>/dev/null || mpic++ --showme:compile 2>/dev/null || true)"
    fi
    if [[ -z "$mpi_inc" ]]; then
        local mpi_prefix
        mpi_prefix="$(_brew_formula_prefix open-mpi 2>/dev/null || _brew_formula_prefix mpich 2>/dev/null || true)"
        if [[ -n "$mpi_prefix" && -d "$mpi_prefix/include" ]]; then
            mpi_inc="-I${mpi_prefix}/include"
        fi
    fi

    CC="${CC}" \
    CFLAGS="${CFLAGS:-} ${mpi_inc}" \
    HDF5_MPI="ON" \
    HDF5_DIR="${HDF5_DIR}" \
        python -m pip install --no-binary=h5py --no-cache-dir --force-reinstall h5py
}

# ---- CMake flags ------------------------------------------------------------
profile_cmake_args() {
    cat <<ARGS
-DCMAKE_C_COMPILER=${CC}
-DCMAKE_Fortran_COMPILER=${FC}
-DHDF5_ROOT=${HDF5_DIR}
-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE
-DHDF5_PREFER_PARALLEL=TRUE
-DFFTW_ROOT=${FFTW_ROOT}
-DWAVEFRONT_BACKEND=OFF
-DWITH_SHAFFT=OFF
-DGPU_AWARE_MPI=OFF
ARGS
}
