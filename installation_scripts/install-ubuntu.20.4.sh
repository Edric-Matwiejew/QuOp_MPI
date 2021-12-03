#!/bin/bash +x

# A QuOp_MPI installation script for QuOp_MPI on Ubuntu 20.04.
# This script requires super-user access - use at your own risk!
# Execute the script from within the 'QuOp_MPI' directory.

apt-get update && apt-get -y  --no-install-recommends install \
	build-essential \
	gfortran \
	pkg-config \
	git \
	python3-pip \
	python3-dev \
	libopenmpi-dev \
	libhdf5-openmpi-dev \
	libfftw3-dev \
	libfftw3-mpi-dev

python3 -m pip install setuptools
python3 -m pip install wheel numpy scipy mpi4py nlopt pandas

export CC="mpicc" 
export MPI="ON" 
export HDF5_PKGCONFIG_NAME="hdf5-openmpi" 
python3 -m pip -v install --no-cache --no-binary=h5py h5py

python3 setup.py sdist bdist_wheel
cd dist
python3 -m pip install quop_mpi-1.0.1.tar.gz

cd ../
cd examples/maxcut
python3 maxcut.py
cd ../
