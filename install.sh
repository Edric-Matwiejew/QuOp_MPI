#!/bin/bash

# This script is an example of how to install and test
# QuOp_MPI on a Debian-based system.

sudo apt-get update -qq && apt-get -y  --no-install-recommends install \
build-essential \
pkg-config \
python3-pip \
python3-dev \
mpich \
libhdf5-mpich-dev \
libfftw3-dev \
libfftw3-mpi-dev

# required
sudo python3 -m pip install setuptools
sudo python3 -m pip install wheel numpy scipy mpi4py
sudo CC=mpicc HDF5_MPI="ON" python3 -m pip -v install --no-binary=h5py h5py

# recommended
sudo python3 -m pip install nlopt pandas

# optional (to run examples)
sudo python3 -m pip pandas-datareader networkx
sudo python3 setup.py sdist bdist_wheel

# build the package and install QuOp_MPI
cd dist
sudo python3 -m pip install quop_mpi-1.0.0.tar.gz
cd ../

# try out an example
cd examples/maxcut
python3 maxcut.py
cd ../../
