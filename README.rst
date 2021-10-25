|Documentation_Status| |DOI| |Builds|

========
QuOp_MPI
========

Introduction
============

QuOp_MPI is a Python 3 module for parallel distributed-memory simulation of quantum variational algorithms with arbitrary phase-shift and mixing operators. The design, usage and performance of QuOp_MPI are covered in an `article which is accessible as a preprint on arXiv <https://arxiv.org/abs/2110.03963>`_. QuOp_MPI’s `documentation is hosted on Read the Docs <https://quop-mpi.readthedocs.io>`_.

Publications
------------

Preprint article:

#. Matwiejew, E. & Wang, J. B. QuOp_MPI: a framework for parallel simulation of quantum variational algorithms. (2021).

QuOp_MPI has provided numerical results for:

#. Bennett, T., Matwiejew, E., Marsh, S. & Wang, J. B. Quantum walk-based vehicle routing optimisation. arXiv:2109.14907 [physics, physics:quant-ph] (2021).
#. Slate, N., Matwiejew, E., Marsh, S. & Wang, J. B. Quantum walk-based portfolio optimisation. Quantum 5, 513 (2021).

Installation
============

1. Install Dependencies
-----------------------

**Debian-Based Systems**

::

    sudo apt-get update -qq && apt-get -y  --no-install-recommends install \
    build-essential \
    pkg-config \
    git \
    python3-pip \
    python3-dev \
    open-mpi\
    libhdf5-openmpi-dev \
    libfftw3-dev \
    libfftw3-mpi-dev


**Debian-Based Systems**

::

    sudo python3 -m pip install setuptools
    sudo python3 -m pip install wheel numpy scipy mpi4py nlopt pandas

Install `h5py built against parallel HDF5 <https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5>`_:

::

    sudo CC="mpicc" MPI="ON" HDF5_PKGCONFIG_NAME="hdf5-openmpi" python3 -m pip -v install --no-cache --no-binary=h5py h5py

.. warning::
    Importing an h5py installation built against a different, or non-parallel, version of HDF5 will cause QuOp_MPI to crash when attempting to save simulation results.

Install optional Python dependancies needed to run the example programs:

::

    sudo python3 -m pip install pandas-datareader networkx

2. Install QuOp_MPI
-------------------

**Debian-Based-Systems**

::

    cd ~/
    git clone https://github.com/Edric-Matwiejew/QuOp_MPI
    cd QuOp_MPI
    python3 setup.py sdist bdist_wheel
    cd dist
    python3 -m pip install quop_mpi-1.0.1.tar.gz


Test the installation by running an example:

::

    cd ../
    cd examples/maxcut
    python3 maxcut.py

Building QuOp_MPI's Documentation
=================================

Install the documentation build dependencies:

::

    python3 -m pip install sphinx sphinx-rtd-theme m2r

And in ~/QuOp_MPI:

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

.. |Documentation_Status| image:: https://readthedocs.org/projects/quop-mpi/badge/?version=latest
   :target: https://quop-mpi.readthedocs.io/en/latest/?badge=latest

.. |DOI| image:: https://zenodo.org/badge/233372703.svg
   :target: https://zenodo.org/badge/latestdoi/233372703
   
.. |Builds| image:: https://github.com/Edric-Matwiejew/QuOp_MPI/actions/workflows/ci_ubuntu.yaml/badge.svg
    :target: https://github.com/Edric-Matwiejew/QuOp_MPI/actions/workflows/ci_ubuntu.yaml
