|Documentation_Status| |DOI|

QuOp_MPI
========

Introduction
------------

QuOp_MPI is a Python 3 module for parallel distributed memory simulation
of quantum variational algorithms on arbitrary mixing
graphs. See https://arxiv.org/abs/1804.08227 and
https://arxiv.org/abs/1912.07353 for the theoretical background.

QuOp_MPI’s complete documentation is hosted at
https://quop-mpi.readthedocs.io.

General Dependencies
--------------------

-  An MPI implementation configured with –enabled-shared.
-  FFTW configured with –enable-fortran, –enable-mpi and –enable-shared.
-  HDF5 configured with –enable-fortran, –enable-parallel, and
   –enable-shared.

Durring installation, setup.py assumes that the include and lib directories
containing the FTW and HDf5 shared object libraries are located in /usr/local,
or /usr this file may need to be modified for your system.

Python Dependencies
-------------------

-  numpy
-  scipy
-  h5py
-  nlopt
-  networkx (To run included example programs.)

Installation on Unix-Like Systems
---------------------------------

If the general and python dependencies are satisfied, QuOp_MPI can be
installed by downloading or cloning the program from
https://github.com/Edric-Matwiejew/QuOp_MPI. Then:

::

    python3 setup.py sdist bdist_wheel
    pip3 install quop_mpi-0.0.1.tar.gz

Before importing QuOp_MPI in a python script, ensure that the path to
the FFTW and HDF5 libraries is present in the LD_LIBRARY_PATH environment variable.
If they are not present:

::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to HDF5 lib>:<path to FFTW lib>

Documentation
-------------

To generate a local copy of the documentation, if not already present,
install sphinx, sphinx-rtd-theme and m2r. On systems using pip:

::

    pip3 install sphinx sphinx-rtd-theme m2r

Navigate to QuOp_MPI/docs and build the documentation:

::

    make html

Documentaion will then be present in QuOp_MPI/docs/build/html.

Detailed installation on Windows
--------------------------------

QuOp_MPI has been developed for Unix-like systems. While, in principle,
it is perfectly possible to install QuOp_MPI on a Windows system, this
is not currently supported. If you wish to run QuOp_MPI on Windows 10 it
is advised that the user install the Linux Subsystem for windows, choose
Ubuntu as the installed Linux distribution and proceed with the
installation method detailed below.

Detailed installation on Ubuntu 18.04.4
---------------------------------------

The following processes successfully installed QuOP_MPI on Ubuntu
18.04.4, this as not been tested on other Linux distros, but the
processes should generally be applicable with minor modifications.

Install dependancies. Note: 'openmpi' may be used in place of 'mpich'.

::

    sudo apt-get update
    sudo apt-get install build-essential cython python3-dev python3-pip python3-setuptools wget git mpich octave

FFTW and HDF5, as provided by the Ubuntu app repository, have not been
built with the required options. These must be built from source.

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

Install the python dependencies:

::

    pip3 install wheel h5py mpi4py numpy networkx scipy

Clone, build and install QuOp_MPI:

::

    git clone https://github.com/Edric-Matwiejew/QuOP_mpi
    cd QuOp_MPI/src
    make
    cd ../
    python3 setup.py sdist bdist_wheel
    cd dist
    pip3 install quop_mpi*.tar.gz
    cd

Alternatively:

::

    git clone https://github.com/Edric-Matwiejew/QuOP_mpi
    cd QuOp_MPI/src
    make
    cd ../
    python3 setup.py develop

Will install QuOp_MPI with reference to the QuOp_MPI source folder. This
is useful if you wish to debug or modify the package.

Next, test the installation by running one of the included examples.

If QuOp_MPI is unable to find the HDF5 or FFTW shared object libraries.

::

    nano ~/.bashrc

Move to the bottom of the file and add:

::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

Then exit Nano (saving changes) and finally,

::

    source ~/.bashrc   

Detailed Installation on MacOS X
--------------------------------

The following installation method uses the ‘Homebrew’ package manager.
This can be installed via the following terminal command:

::

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

You will be prompted for your user password on installing the homebrew
dependencies and on installing homebrew itself.

Next, install the GNU compiler collection, python3 + pip3, MPI, and
utilities required to download and configure QuOp_MPI’s dependencies.

::

    brew install gcc python wget pkg-config mpich swing guile octave

Download, extract and install parallel-HDF5.

::

    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
    tar -xvf hdf5-1.10.6.tar.gz
    cd hdf5-1.10.6
    export CC=mpicc
    export FC=mpif90
    ./configure --enable-fortran --enable-shared --enable-parallel --prefix=/usr/local
    make
    sudo make install
    cd

Download, extract and install FFTW.

::

    wget http://www.fftw.org/fftw-3.3.8.tar.gz
    tar -xvf fftw-3.3.8.tar.gz
    cd fftw-3.3.8
    ./configure --enable-mpi --enable-fortran --enable-shared --prefix=/usr/local
    make
    sudo make install
    cd

Finally, we can clone and install QuOp_mpi.

::

    git clone https://github.com/Edric-Matwiejew/QuOP_mpi
    cd QuOp_mpi/src
    make
    (Note: entered into makefile and altered LIB and INCLUDE to go to /usr/local/libor /usr/local/include. I think is can be done in the terminal however)
    cd ../
    python3 setup.py sdist bdist_wheel
    cd dist
    pip3 install quop_mpi*.tar.gz
    cd

Alternatively:

::

    git clone https://github.com/Edric-Matwiejew/QuOP_mpi
    cd QuOp_mpi/src
    make
    cd ../
    python3 setup.py develop

Will install QuOp_MPI with reference to the QuOp_MPI source folder. This
is useful if you wish to debug or modify the package.

Contact Information
-------------------

If you encounter a bug, please submit a
report via github. If you would like to get in touch, email me at edric.matwiejew@research.uwa.edu.au.

.. |Documentation_Status| image:: https://readthedocs.org/projects/quop-mpi/badge/?version=latest
   :target: https://quop-mpi.readthedocs.io/en/latest/?badge=latest

.. |DOI| image:: https://zenodo.org/badge/233372703.svg
   :target: https://zenodo.org/badge/latestdoi/233372703
