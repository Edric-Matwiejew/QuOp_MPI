#!/usr/bin/env bash
# =============================================================================
# Profile: pawsey-setonix
#
# Pawsey Setonix -- Cray EX with AMD MI250X (gfx90a) GPUs.
# Supports both 'mpi' (CPU-only) and 'wavefront' (GPU / HIP) backends.
#
# >>> SITE-SPECIFIC: adjust module lists or build defaults via config.toml. <<<
# >>> SHAFFT defaults to the installer-managed .deps tree unless overridden. <<<
# =============================================================================
apply_profile_defaults

# ---- Site-specific paths (override via environment if needed) ----------------
OFFLOAD_ARCH="${OFFLOAD_ARCH:-${CFG_OFFLOAD_ARCH:-gfx90a}}"
MPI_BUILD_TYPE="${MPI_BUILD_TYPE:-${CFG_MPI_BUILD_TYPE:-Release}}"
WAVEFRONT_BUILD_TYPE="${WAVEFRONT_BUILD_TYPE:-${CFG_WAVEFRONT_BUILD_TYPE:-Debug}}"

# ---- Site-specific helpers ---------------------------------------------------

profile_wrapper_opts() {
    local wrapper="$1"
    local option="$2"
    if command -v "$wrapper" >/dev/null 2>&1; then
        "$wrapper" --cray-print-opts="$option" 2>/dev/null || true
    fi
}

profile_sanitize_path_flags() {
    local flags="$1"
    local -a sanitized=()
    local token
    local path

    for token in ${flags}; do
        case "$token" in
            -I*)
                path="${token#-I}"
                if [[ -d "$path" ]]; then
                    sanitized+=("$token")
                fi
                ;;
            -L*)
                path="${token#-L}"
                if [[ -d "$path" ]]; then
                    sanitized+=("$token")
                fi
                ;;
            *)
                sanitized+=("$token")
                ;;
        esac
    done

    printf '%s\n' "${sanitized[*]}"
}

profile_write_mpi_cfg() {
    local output="$1"
    local -a include_dirs=()
    local -a library_dirs=()
    local -a libraries=()
    local -a extra_compile_args=()
    local -a extra_link_args=()
    local token

    for token in ${MPI_CFLAGS}; do
        case "$token" in
            -I*) include_dirs+=("${token#-I}") ;;
            *) extra_compile_args+=("$token") ;;
        esac
    done

    for token in ${MPI_LDFLAGS}; do
        case "$token" in
            -L*) library_dirs+=("${token#-L}") ;;
            -l*) libraries+=("${token#-l}") ;;
            *) extra_link_args+=("$token") ;;
        esac
    done

    cat >"$output" <<EOF
[mpi]
mpicc = ${CC}
mpicxx = ${CXX}
include_dirs = $(join_by_delimiter ':' "${include_dirs[@]}")
library_dirs = $(join_by_delimiter ':' "${library_dirs[@]}")
libraries = ${libraries[*]}
runtime_library_dirs = $(profile_wrapper_opts "$MPI_CC_WRAPPER" cray_ld_library_path)
EOF

    if ((${#extra_compile_args[@]})); then
        printf 'extra_compile_args = %s\n' "${extra_compile_args[*]}" >>"$output"
    fi
    if ((${#extra_link_args[@]})); then
        printf 'extra_link_args = %s\n' "${extra_link_args[*]}" >>"$output"
    fi
}

# ---- Module loading ---------------------------------------------------------
profile_load_modules() {
    # Module lists are defined in config.toml under [modules].common.
    # No inline fallbacks -- config.toml is the single source of truth.
    load_module_list CFG_PROFILE_MODULES_COMMON || return 1
}

profile_load_modules_gpu() {
    # Module lists are defined in config.toml under [modules].wavefront.
    load_module_list CFG_PROFILE_MODULES_WAVEFRONT || return 1
}

# ---- Post-module environment ------------------------------------------------
profile_post_modules_env() {
    export MPI_CC_WRAPPER="${MPI_CC_WRAPPER:-${CFG_MPI_CC_WRAPPER:-cc}}"
    export MPI_CXX_WRAPPER="${MPI_CXX_WRAPPER:-${CFG_MPI_CXX_WRAPPER:-CC}}"
    export MPI_FC_WRAPPER="${MPI_FC_WRAPPER:-${CFG_MPI_FC_WRAPPER:-ftn}}"

    require_command "$MPI_CC_WRAPPER" || return 1
    require_command "$MPI_CXX_WRAPPER" || return 1
    require_command "$MPI_FC_WRAPPER" || return 1

    export CC="${CC:-$(resolve_command "${CFG_CC:-gcc}")}"
    export CXX="${CXX:-$(resolve_command "${CFG_CXX:-g++}")}"
    export FC="${FC:-$(resolve_command "${CFG_FC:-gfortran}")}"

    require_command "$CC" || return 1
    require_command "$CXX" || return 1
    require_command "$FC" || return 1

    if [[ "$BACKEND" == "wavefront" ]]; then
        # Some module stacks advertise stale ROCm paths in wrapper flags
        # (for example rocprofiler include dirs that no longer exist). CMake
        # rejects imported MPI targets that reference non-existent includes.
        if [[ -n "${CRAY_ROCM_INCLUDE_OPTS:-}" ]]; then
            CRAY_ROCM_INCLUDE_OPTS="$(profile_sanitize_path_flags "${CRAY_ROCM_INCLUDE_OPTS}")"
            export CRAY_ROCM_INCLUDE_OPTS
        fi
        if [[ -n "${CRAY_ROCM_POST_LINK_OPTS:-}" ]]; then
            CRAY_ROCM_POST_LINK_OPTS="$(profile_sanitize_path_flags "${CRAY_ROCM_POST_LINK_OPTS}")"
            export CRAY_ROCM_POST_LINK_OPTS
        fi
    fi

    MPI_CFLAGS="${MPI_CFLAGS:-$(profile_wrapper_opts "$MPI_CC_WRAPPER" cflags)}"
    MPI_LDFLAGS="${MPI_LDFLAGS:-$(profile_wrapper_opts "$MPI_CC_WRAPPER" libs)}"
    MPI_FCFLAGS="${MPI_FCFLAGS:-$(profile_wrapper_opts "$MPI_FC_WRAPPER" cflags)}"
    MPI_FLIBS="${MPI_FLIBS:-$(profile_wrapper_opts "$MPI_FC_WRAPPER" libs)}"

    MPI_CFLAGS="$(profile_sanitize_path_flags "${MPI_CFLAGS}")"
    MPI_LDFLAGS="$(profile_sanitize_path_flags "${MPI_LDFLAGS}")"
    MPI_FCFLAGS="$(profile_sanitize_path_flags "${MPI_FCFLAGS}")"
    MPI_FLIBS="$(profile_sanitize_path_flags "${MPI_FLIBS}")"

    export MPI_CFLAGS MPI_LDFLAGS MPI_FCFLAGS MPI_FLIBS

    export HDF5_DIR="${HDF5_DIR:-$(find_hdf5_dir || true)}"
    if [[ -z "${HDF5_DIR:-}" ]]; then
        echo "Error: could not locate the parallel HDF5 prefix from h5pcc/h5cc/pkg-config"
        return 1
    fi

    export FFTW_ROOT="${FFTW_ROOT:-$(find_fftw_root || true)}"
    if [[ -z "${FFTW_ROOT:-}" ]]; then
        echo "Error: could not locate the FFTW prefix from fftw-wisdom/pkg-config"
        return 1
    fi

    if [[ "$BACKEND" == "wavefront" ]]; then
        export SHAFFT_PATH="${SHAFFT_PATH:-$(resolve_shafft_path || true)}"
        export ROCM_PATH="${ROCM_PATH:-$(find_rocm_path || true)}"
        if [[ -z "${ROCM_PATH:-}" ]]; then
            echo "Error: could not locate ROCm from hipcc; set ROCM_PATH explicitly"
            return 1
        fi

        export MPICH_GPU_SUPPORT_ENABLED=1

        local -a extra_ld_paths=("${ROCM_PATH}/lib")
        local wrapper_ld_path
        wrapper_ld_path="$(profile_wrapper_opts "$MPI_CC_WRAPPER" cray_ld_library_path)"

        # SHAFFT is bundled into repaired Linux wheels via auditwheel, so it
        # does not need to be on LD_LIBRARY_PATH at runtime. It is still
        # exported via SHAFFT_PATH for CMake and auditwheel to find at build time.

        local extra_ld_path
        extra_ld_path="$(IFS=:; echo "${extra_ld_paths[*]}")"
        export LD_LIBRARY_PATH="${extra_ld_path}${wrapper_ld_path:+:${wrapper_ld_path}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

# ---- Venv configuration ----------------------------------------------------
profile_configure_venv() {
    if [[ "$BACKEND" == "mpi" ]]; then
        # Reuse the system-installed mpi4py provided by the py-mpi4py module.
        # Module name is defined in config.toml under [modules].mpi_python.
        load_module_list CFG_PROFILE_MODULES_MPI_PYTHON || return 1
        VENV_ARGS+=(--system-site-packages)
    fi
}

# ---- mpi4py -----------------------------------------------------------------
profile_install_mpi4py() {
    if [[ "$BACKEND" == "mpi" ]]; then
        info "Using system mpi4py (loaded via module)"
        return
    fi

    local mpi_cc_wrapper
    if [[ -n "${MPI_CC_WRAPPER:-}" ]] && command -v "${MPI_CC_WRAPPER}" >/dev/null 2>&1; then
        mpi_cc_wrapper="$(command -v "${MPI_CC_WRAPPER}")"
    elif [[ -n "${CFG_MPI_CC_WRAPPER:-}" ]] && command -v "${CFG_MPI_CC_WRAPPER}" >/dev/null 2>&1; then
        mpi_cc_wrapper="$(command -v "${CFG_MPI_CC_WRAPPER}")"
    else
        mpi_cc_wrapper="$(command -v mpicc 2>/dev/null || true)"
    fi

    if [[ -n "$mpi_cc_wrapper" ]]; then
        info "Building mpi4py from source with ${CC} via ${mpi_cc_wrapper}"
        MPICH_CC="${CC}" \
        MPICH_CXX="${CXX}" \
        MPI4PY_BUILD_MPICC="${mpi_cc_wrapper}" \
        MPI4PY_BUILD_MPILD="${mpi_cc_wrapper}" \
            python -m pip install --no-cache-dir --no-binary=mpi4py \
                --force-reinstall mpi4py
        return
    fi

    # Fallback when the Cray MPI compiler wrapper is unavailable: build with
    # the direct host compiler and wrapper-derived MPI flags.
    info "Building mpi4py from source with ${CC} and wrapper-derived MPI flags"
    local mpi_cfg
    local build_deps_dir="${BUILD_DEPS_DIR:-$PROJECT_ROOT/.cache/quop/${PROFILE_ID:-${PROFILE:-site}}/${BACKEND:-mpi}/deps}"
    mkdir -p "$build_deps_dir"
    mpi_cfg="$(mktemp "${build_deps_dir}/mpi4py.XXXXXX.cfg")"
    profile_write_mpi_cfg "$mpi_cfg"

    MPI4PY_BUILD_MPICC="${CC}" \
    MPI4PY_BUILD_MPILD="${CC}" \
    MPI4PY_BUILD_MPICFG="${mpi_cfg}" \
        python -m pip install --no-cache-dir --no-binary=mpi4py \
            --force-reinstall mpi4py

    rm -f "$mpi_cfg"
}

# ---- h5py -------------------------------------------------------------------
profile_install_h5py() {
    info "Building h5py from source with parallel HDF5 from ${HDF5_DIR}"
    CC="${CC}" \
    CFLAGS="${MPI_CFLAGS}${CFLAGS:+ ${CFLAGS}}" \
    LDFLAGS="${MPI_LDFLAGS}${LDFLAGS:+ ${LDFLAGS}}" \
    HDF5_MPI="ON" \
    HDF5_DIR="${HDF5_DIR}" \
        python -m pip install -vvv --no-deps --force-reinstall \
            --no-binary=h5py --no-cache-dir h5py
}

# ---- CMake flags ------------------------------------------------------------
profile_cmake_args() {
    local cmake_prefix_path
    local shafft_path

    if [[ "$BACKEND" == "wavefront" ]]; then
        shafft_path="$(resolve_shafft_path || true)"
        cmake_prefix_path="$(join_by_delimiter ';' \
            "${shafft_path}" "${ROCM_PATH}" "${HDF5_DIR}" "${FFTW_ROOT}")"
        cat <<ARGS
-DCMAKE_C_COMPILER=${CC}
-DCMAKE_Fortran_COMPILER=${FC}
-DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc
-DHDF5_ROOT=${HDF5_DIR}
-DFFTW_ROOT=${FFTW_ROOT}
-DCMAKE_PREFIX_PATH=${cmake_prefix_path}
-DFETCHCONTENT_BASE_DIR=${FETCHCONTENT_BASE_DIR}
-DWAVEFRONT_BACKEND=ON
-DSHAFFT_PATH=${shafft_path}
-DOFFLOAD_ARCH=${OFFLOAD_ARCH}
-DROCM_PATH=${ROCM_PATH}
-DGPU_AWARE_MPI=ON
ARGS
    else
        cmake_prefix_path="$(join_by_delimiter ';' "${HDF5_DIR}" "${FFTW_ROOT}")"
        cat <<ARGS
-DCMAKE_C_COMPILER=${CC}
-DCMAKE_Fortran_COMPILER=${FC}
-DHDF5_ROOT=${HDF5_DIR}
-DFFTW_ROOT=${FFTW_ROOT}
-DCMAKE_PREFIX_PATH=${cmake_prefix_path}
-DWAVEFRONT_BACKEND=OFF
-DWITH_SHAFFT=OFF
-DGPU_AWARE_MPI=OFF
ARGS
    fi
}
