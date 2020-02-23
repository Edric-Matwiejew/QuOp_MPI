[![Documentation Status](https://readthedocs.org/projects/quop-mpi/badge/?version=latest)](https://quop-mpi.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/233372703.svg)](https://zenodo.org/badge/latestdoi/233372703)


# QuOp_MPI

## Introduction

Python 3 module for parallel distributed memory simulation of Quantum Approximate Optimization Algorithms on arbitrary mixing graphs. See https://arxiv.org/abs/1804.08227 and https://arxiv.org/abs/1912.07353 for the theoretical background.

QuOp_MPI's complete documenation can be found at "quop-mpi.readthedocs.io".

## General Dependencies

+ An MPI implementation configured with --enabled-shared.
+ FFTW configured with --enable-fortran, --enable-mpi and --enable-shared.
+ HDF5 configured with --enable-fortran, --enable-parallel, and --enable-shared.

src/Makefile assumes that the include and lib directories contaning the FTW and HDf5 shared object libraries are located in /usr/local, this file may need to be modified for your system.

## Python Dependencies

+ numpy
+ scipy
+ h5py
+ Networkx (To run included example programs.)

## Runninig and Installing QuOp_MPI on Clusters

Please consult the section covering containerisation in the QuOp_MPI docs.

## Installation on Unix-Like Systems

If the general and python depencies are statisfied, QuOp_MPI can be installed by downloading or cloning the program from https://github.com/Edric-Matwiejew/QuOp_MPI. Then:

    cd QuOp_MPI/src
    make
    cd ../
    python3 setup.py sdist bdist_wheel
    pip3 install qsw_mpi-0.0.1.tar.gz

Before importing QuOp_MPI in a python script, ensure that the path to the FFTW and HDF5 libraries is present in the LD_LIBRARY_PATH vaiable. If they are not present:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to HDF5 lib>:<path to FFTW lib>

## Documentaion
To generate a local copy of the documentaion, if not already present, install sphinx, sphinx-rtd-theme and m2r. On systems using PIP:

    pip3 install sphinx sphinx-rtd-theme m2r

Navigate to QuOp_MPI/docs and build the documentaion:

    make html

Documentaion will then be present in QuOp_MPI/docs/build/html.

## Detailed installation on Ubuntu 18.04.4

The following processes successfully installed QuOP_MPI on Ubuntu 18.04.4, this as not been tested on other systems, but the processes should be generally applicable with minor modifications.

Install MPICH and build applications:

    sudo apt-get update
    sudo apt-get install build-essential cython python3-dev python3-pip python3-setuptools wget git mpich

FFTW and HDF5 as provided by the Ubuntu app repository have not been built with the required options. These must be built from source.

    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
    tar -xvf hdf5-1.10.6.tar.gz
    cd hdf5-1.10.6
    ./configure --enable-fortran --enable-shared --enable-parallel --prefix=/usr/local
    make && make install
    cd

    wget http://www.fftw.org/fftw-3.3.8.tar.gz
    tar -xvf fftw-3.3.8.tar.gz
    cd fftw-3.3.8
    ./configure --enable-mpi --enable-fortran --enable-shared --prefix=/usr/local
    make && make install
    cd

Install the python dependencies:

    pip3 install wheel h5py mpi4py numpy networkx scipy

Clone, build and install QuOp_MPI:

    git clone https://github.com/Edric-Matwiejew/QuOP_mpi
    cd QuOp_mpi/src
    make
    cd ../
    python3 setup.py sdist bdist_wheel
    cd dist
    pip3 install quop_mpi*.tar.gz
    cd

Alternatively:

    git clone https://github.com/Edric-Matwiejew/QuOP_mpi
    cd QuOp_mpi/src
    make
    cd ../
    python3 setup.py develop

Will install QuOp_MPI with reference to the QuOp_MPI source folder. This is useful if you wish to debug or modify the package.

## Contact Information

QuOa_MPI is under development. If you encounter an issue please submit an issue via github, or email me at edric.matwiejew@research.uwa.edu.au.

