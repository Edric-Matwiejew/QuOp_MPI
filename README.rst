========
QuOp_MPI
========

Introduction
============

QuOp_MPI is a Python 3 module for parallel distributed-memory simulation of Quantum Variational Algorithms (QVAs) with arbitrary phase-shift and mixing operators. The design, usage and performance of QuOp_MPI are covered in an `article which is accessible as a preprint on arXiv <https://arxiv.org/abs/2110.03963>`_.

Related Publications
--------------------

Package Development:

#. Matwiejew, E. & Wang, J. B. QuOp_MPI: A framework for parallel simulation of quantum variational algorithms. Journal of Computational Science 62, 101711 (2022).
#. Matwiejew, E. & Wang, J. QSW_MPI: A framework for parallel simulation of quantum stochastic walks. Computer Physics Communications 107724 (2020)

QuOp_MPI has provided numerical results for:

#. Bennett, T., Matwiejew, E., Marsh, S. & Wang, J. B. Quantum Walk-Based Vehicle Routing Optimisation. Frontiers in Physics 9, (2021).
#. Slate, N., Matwiejew, E., Marsh, S. & Wang, J. B. Quantum walk-based portfolio optimisation. Quantum 5, 513 (2021).
#. Matwiejew, E., Pye J. & Wang J. B. Quantum Optimisation for Continuous Multivariable Functions by a Structured Search. arXiv:2210.06227, (2022).

Installation
============

Build Dependencies
------------------

Building QuOp_MPI requires:

* GCC 7+.
* MPI (Open-MPI or MPICH).
* HDF5 (configured with --enable-fortran --enable-shared --enable-parallel).
* FFTW3 (configured --enable-fortran --enable-shard --enable-mpi).
* Python 3.11+.
* Cmake 3.9+.

These can be installed through the package manager of most Linux distributions, or the Homebrew third-party package manager elsewhere. Example install scripts are included in :code:`installation_scripts`.

Package Installation
--------------------

Build and install:

::
        
    FC=mpifort python -m setup bdist_wheel
    python -m pip install dist/QuOp_MPI-*.whl

Test the installation by running an example:

::

    cd ../
    cd examples/maxcut
    mpiexec -N 2 python3 maxcut.py
    
Documentation
=============

Install the documentation build dependencies:

::

    pip install .[docs]

Then:

::

    python3 setup.py build_sphinx

Building FFTW3 and HDF5 From Source
===================================

If parallel versions of FFTW3 and HDF5 packages are not available on your system, these packages can be built from source. For a comprehensive overview of their installation, please consult the documentation provided by the FFTW and HDF5 projects. The below commands should work with most Unix-like systems:

::

    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
    tar -xvf hdf5-1.10.6.tar.gz
    cd hdf5-1.10.6
    ./configure --enable-fortran --enable-shared --enable-parallel --prefix=/usr/local
    make && sudo make install
    cd

    wget http://www.fftw.org/fftw-3.3.8.tar.gz
    tar -xvf fftw-3.3.8.tar.gz
    cd fftw-3.3.8
    ./configure --enable-mpi --enable-fortran --enable-shared --prefix=/usr/local
    make && sudo make install
    cd

Editing .bashrc
===============

If QuOp_MPI is unable to find the HDF5 or FFTW shared object libraries.

::

    nano ~/.bashrc

Move to the bottom of the file and add:

::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

Then exit Nano (saving changes) and finally,

::

    source ~/.bashrc

Contact Information
===================

If you encounter a bug, please submit a
report via Github. If you would like to get in touch, email me at edric.matwiejew@research.uwa.edu.au.
