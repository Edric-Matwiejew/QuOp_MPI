#!/usr/bin/env bash
# =============================================================================
# Profile: generic
#
# Standard Linux workstation or cluster with MPI already installed.
# Assumes mpicc, HDF5 (parallel), FFTW, and gfortran are available in $PATH
# or discoverable via pkg-config / environment variables.
# =============================================================================
apply_profile_defaults

# No environment modules required.
# Override HDF5_DIR if the parallel HDF5 installation is not auto-detected.

profile_install_mpi4py() {
    info "Installing mpi4py via pip"
    python -m pip install mpi4py
}

profile_install_h5py() {
    info "Installing h5py with MPI support (building from source)"
    HDF5_MPI="ON" \
        python -m pip install --no-binary=h5py h5py
}

profile_cmake_args() {
    cat <<ARGS
-DWAVEFRONT_BACKEND=OFF
-DWITH_SHAFFT=OFF
-DGPU_AWARE_MPI=OFF
-DCMAKE_BUILD_TYPE=Release
ARGS
}
