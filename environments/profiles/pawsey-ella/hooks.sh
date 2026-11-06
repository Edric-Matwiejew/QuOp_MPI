#!/usr/bin/env bash
# Ella GH200: GCC/GFortran host code, NVCC device code, CUDA-aware Open MPI.
# The 2026.11 module lists are defined in the backend's config.toml and are
# loaded by the shared installer and activation hooks.
apply_profile_defaults

MPI_BUILD_TYPE="${MPI_BUILD_TYPE:-${CFG_MPI_BUILD_TYPE:-Release}}"
WAVEFRONT_BUILD_TYPE="${WAVEFRONT_BUILD_TYPE:-${CFG_WAVEFRONT_BUILD_TYPE:-Release}}"
OFFLOAD_ARCH="${OFFLOAD_ARCH:-${CFG_OFFLOAD_ARCH:-90}}"

profile_post_modules_env() {
    export CC="${CC:-$(resolve_command "${CFG_CC:-gcc}")}"
    export CXX="${CXX:-$(resolve_command "${CFG_CXX:-g++}")}"
    export FC="${FC:-$(resolve_command "${CFG_FC:-gfortran}")}"
    export MPI_CC_WRAPPER="${MPI_CC_WRAPPER:-${CFG_MPI_CC_WRAPPER:-mpicc}}"
    export MPI_CXX_WRAPPER="${MPI_CXX_WRAPPER:-${CFG_MPI_CXX_WRAPPER:-mpicxx}}"
    export MPI_FC_WRAPPER="${MPI_FC_WRAPPER:-${CFG_MPI_FC_WRAPPER:-mpifort}}"
    local compiler
    for compiler in "$CC" "$CXX" "$FC" "$MPI_CC_WRAPPER" "$MPI_CXX_WRAPPER" "$MPI_FC_WRAPPER"; do
        require_command "$compiler" || return 1
    done
    export HDF5_DIR="${HDF5_DIR:-${CFG_HDF5_DIR:-${PAWSEY_HDF5_HOME:-$(find_hdf5_dir || true)}}}"
    export FFTW_ROOT="${FFTW_ROOT:-${CFG_FFTW_ROOT:-${PAWSEY_FFTW_HOME:-$(find_fftw_root || true)}}}"
    if [[ -z "$HDF5_DIR" || -z "$FFTW_ROOT" ]]; then
        echo "Error: load parallel HDF5 (C and Fortran) and FFTW3 MPI, or set HDF5_DIR and FFTW_ROOT"
        return 1
    fi

    if [[ "$BACKEND" == "wavefront" ]]; then
        export HIP_PLATFORM="${HIP_PLATFORM:-${CFG_HIP_PLATFORM:-nvidia}}"
        if [[ "$HIP_PLATFORM" != "nvidia" ]]; then
            echo "Error: the pawsey-ella wavefront profile requires HIP_PLATFORM=nvidia"
            return 1
        fi
        export CUDA_PATH="${CUDA_PATH:-${CFG_CUDA_PATH:-${CUDA_HOME:-${PAWSEY_CUDA_HOME:-$(prefix_from_executable nvcc || true)}}}}"
        export HIP_PATH="${HIP_PATH:-${CFG_HIP_PATH:-${PAWSEY_HIP_HOME:-$(find_rocm_path || true)}}}"
        export HIPFORT_PATH="${HIPFORT_PATH:-${CFG_HIPFORT_PATH:-${PAWSEY_HIPFORT_HOME:-}}}"
        QUOP_HIP_COMPILER="${QUOP_HIP_COMPILER:-${CFG_HIP_COMPILER:-${CUDA_PATH}/bin/nvcc}}"
        export SHAFFT_PATH="${SHAFFT_PATH:-${PAWSEY_SHAFFT_HOME:-$(resolve_shafft_path || true)}}"
        export GPU_AWARE_MPI="${GPU_AWARE_MPI:-${CFG_GPU_AWARE_MPI:-false}}"
        if [[ -z "$CUDA_PATH" || -z "$HIP_PATH" ]]; then
            echo "Error: load CUDA and HIP for NVIDIA, or set CUDA_PATH and HIP_PATH"
            return 1
        fi
        require_command "$QUOP_HIP_COMPILER" || return 1

        # Retain the module's CUDA math-library paths (separate from the toolkit
        # in the NVIDIA HPC SDK) while adding the HIP and toolkit runtimes.
        local prefix
        local libdir
        for prefix in "$HIP_PATH" "$CUDA_PATH" "$HIPFORT_PATH" "${PAWSEY_HIPFFT_HOME:-}"; do
            [[ -n "$prefix" ]] || continue
            for libdir in lib lib64 targets/sbsa-linux/lib; do
                if [[ -d "${prefix}/${libdir}" ]]; then
                    export LD_LIBRARY_PATH="${prefix}/${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
                fi
            done
        done
    fi
}

profile_install_mpi4py() {
    info "Building mpi4py against the loaded Open MPI"
    MPI4PY_BUILD_MPICC="$MPI_CC_WRAPPER" \
        python -m pip install --no-cache-dir --no-binary=mpi4py --no-deps mpi4py
}

profile_install_h5py() {
    info "Building h5py against the loaded parallel HDF5"
    CC="$MPI_CC_WRAPPER" HDF5_MPI=ON HDF5_DIR="$HDF5_DIR" \
        python -m pip install --no-cache-dir --no-binary=h5py --no-deps h5py
}

profile_cmake_args() {
    local prefix_path
    local module_prefix_path="${CMAKE_PREFIX_PATH:-}"
    # Preserve module-provided prefixes, including HPC SDK math libraries.
    prefix_path="$(join_by_delimiter ';' "$HDF5_DIR" "$FFTW_ROOT" \
        "${CFG_CMAKE_PREFIX_PATH:-}" "${module_prefix_path//:/;}")"
    cat <<ARGS
-DCMAKE_C_COMPILER=${CC}
-DCMAKE_CXX_COMPILER=${CXX}
-DCMAKE_Fortran_COMPILER=${FC}
-DMPI_C_COMPILER=${MPI_CC_WRAPPER}
-DMPI_CXX_COMPILER=${MPI_CXX_WRAPPER}
-DMPI_Fortran_COMPILER=${MPI_FC_WRAPPER}
-DHDF5_ROOT=${HDF5_DIR}
-DHDF5_PREFER_PARALLEL=TRUE
-DFFTW_ROOT=${FFTW_ROOT}
-DFETCHCONTENT_BASE_DIR=${FETCHCONTENT_BASE_DIR}
ARGS
    if [[ "$BACKEND" == "wavefront" ]]; then
        prefix_path="$(join_by_delimiter ';' "$prefix_path" "$HIP_PATH" "$HIPFORT_PATH" \
            "${PAWSEY_HIPFFT_HOME:-}" "$SHAFFT_PATH")"
        cat <<ARGS
-DCMAKE_BUILD_TYPE=${WAVEFRONT_BUILD_TYPE}
-DWAVEFRONT_BACKEND=ON
-DWITH_SHAFFT=ON
-DHIP_PLATFORM=nvidia
-DCMAKE_HIP_PLATFORM=nvidia
-DCMAKE_HIP_COMPILER=${QUOP_HIP_COMPILER}
-DCMAKE_HIP_HOST_COMPILER=${CXX}
-DOFFLOAD_ARCH=${OFFLOAD_ARCH}
-DCMAKE_HIP_ARCHITECTURES=${OFFLOAD_ARCH}
-DCUDA_PATH=${CUDA_PATH}
-DHIP_PATH=${HIP_PATH}
-DHIPFORT_PATH=${HIPFORT_PATH}
-DSHAFFT_PATH=${SHAFFT_PATH}
-DGPU_AWARE_MPI=${GPU_AWARE_MPI}
ARGS
    else
        cat <<ARGS
-DCMAKE_BUILD_TYPE=${MPI_BUILD_TYPE}
-DWAVEFRONT_BACKEND=OFF
-DWITH_SHAFFT=OFF
-DGPU_AWARE_MPI=OFF
ARGS
    fi
    printf '%s\n' "-DCMAKE_PREFIX_PATH=${prefix_path}"
}
