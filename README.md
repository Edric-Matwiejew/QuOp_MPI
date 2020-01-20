# qwao_mpi
Python 3 module for parallel distributed memory simulation of the QWAO (Quantum Walk Assisted Optimization) algorithm on circulant mixing graphs. See https://arxiv.org/abs/1912.07353 for the theoretical background.

## Build Requirements

+ FFTW compiled with the --enable-fortran, --enable-mpi and --enable-shared options.
+ HDF5 compiled with the --enable-fortran, --enable-parallel, and --enable-shared options.

src/Makefile assumes that the include and lib directories contaning the FTW and HDf5 shared object libraries are located in /usr/.

## Python Dependencies

+ numpy
+ scipy
+ h5py

## Installation on Unix-Like Systems
    cd qwao_mpi/src
    make
    cd ../
    python3 setup.py sdist bdist_wheel
    pip3 install qsw_mpi-0.0.1.tar.gz

Before importing qwao_mpi in a python script, ensure that the path to the FFTW and HDF5 libraries is present in the LD_LIBRARY_PATH vaiable. If they are not present:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to HDF5 lib>:<path to FFTW lib>

## Documentaion
If not already present, install sphinx, sphinx-rtd-theme and m2r. On systems using PIP:

    pip3 install sphinx sphinx-rtd-theme m2r

Navigate to qwao_mpi/docs and build the documentaion:

    make html

Documentaion will then be present in qwao_mpi/docs/build/html.

## Todo

+ Handle case of user saving a configuration with the same name as one already existing in a .h5 file.
+ Outline structure of saved .h5 file.
