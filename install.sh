#!/bin/bash

# This script is an example of how to install and test
# QuOp_MPI on a Debian-based system with mpich chosen
# as the MPI implementation.

# Install required packages from distro repositories:
sudo apt-get update -qq && apt-get -y  --no-install-recommends install \
build-essential \
pkg-config \
python3-pip \
python3-dev \
mpich \
libhdf5-mpich-dev \
libfftw3-dev \
libfftw3-mpi-dev

# Install the required python modules:
sudo python3 -m pip install setuptools
sudo python3 -m pip install wheel numpy scipy mpi4py nlopt pandas

# Importing a version of H5Py which is built against a different version of
# HDF5 with cause a QuOp_MPI to crash when writing simulation results to disk.
# Build h5py against the local version of HDF5:
sudo CC="mpicc" HDF5_PKGCONFIG_NAME="hdf5-mpich" python3 -m pip -v install --no-cache --no-binary=h5py h5py

# Install optional python modules needed to run the examples:
sudo python3 -m pip install pandas-datareader networkx

# Build QuOp_MPI:
sudo python3 setup.py sdist bdist_wheel

# Install QuOp_MPI:
cd dist
sudo python3 -m pip install quop_mpi-1.0.0.tar.gz

# Test the installation by running an example:
cd ../
cd examples/maxcut
python3 maxcut.py
