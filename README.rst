QuOp_MPI
========

A Parallel Framework for Quantum Variational Algorithms
-------------------------------------------------------

QuOp_MPI is a Python 3 module designed for parallel, distributed-memory simulation of Quantum Variational Algorithms (QVAs) with arbitrary phase-shift and mixing operators.

**Current Version:** 1.3.0

For an in-depth discussion on design, usage, and performance, please refer to the `preprint on arXiv <https://arxiv.org/abs/2110.03963>`_.

Related Publications
--------------------

**Package Development:**

1. Matwiejew, E. & Wang, J. B. *QuOp_MPI: A framework for parallel simulation of quantum variational algorithms.* Journal of Computational Science 62, 101711 (2022).
2. Matwiejew, E. & Wang, J. *QSW_MPI: A framework for parallel simulation of quantum stochastic walks.* Computer Physics Communications 107724 (2020).

**Numerical Results Provided by QuOp_MPI:**

1. Bennett, T., Matwiejew, E., Marsh, S. & Wang, J. B. *Quantum Walk-Based Vehicle Routing Optimisation.* Frontiers in Physics 9, (2021).
2. Slate, N., Matwiejew, E., Marsh, S. & Wang, J. B. *Quantum walk-based portfolio optimisation.* Quantum 5, 513 (2021).
3. Matwiejew, E., Pye J. & Wang J. B. *Quantum Optimisation for Continuous Multivariable Functions by a Structured Search.* arXiv:2210.06227 (2022).

Installation
============

Prerequisites
-------------

Before installing QuOp_MPI, ensure that the following system dependencies are met:

- **Compiler:** GCC 7+ with Fortran support (e.g., using `mpifort`).
- **MPI:** Open-MPI or MPICH.
- **HDF5:** Configured with `--enable-fortran --enable-shared --enable-parallel`.
- **FFTW3:** Configured with `--enable-fortran --enable-shared --enable-mpi`.
- **Python:** 3.11+

You can install these prerequisites using your Linux package manager or Homebrew on macOS. Instructions for building HDF5 and FFTW3 from source are provided later in this README.

Package Installation
--------------------

First, install the following build dependencies:

.. code-block:: bash

    python -m pip install --upgrade pip setuptools
    python -m pip install scikit-build cmake ninja

Next, choose one of the following build methods:

**Standard Build:**

To install from source (ensure that all build prerequisites are set), run:

.. code-block:: bash

    python -m pip install .

.. note::

    If you encounter installation issues on a repeated build, try removing the `_skbuild` directory:

    .. code-block:: bash

       rm -rf _skbuild

**Development Build:**

For development or modifying QuOp_MPI, use the following steps:

.. code-block:: bash

    cmake -B build -S .
    cmake --build build --target install
    python -m pip install -e .

Optional Dependencies
---------------------

QuOp_MPI provides optional dependencies for documentation and examples.  

- **Install all optional dependencies at once:**

  .. code-block:: bash

     python -m pip install '.[all]'

- **Manual installation:**  
  If you prefer, install dependencies manually:

  .. code-block:: bash

     python -m pip install numpydoc==1.5.0 sphinxcontrib-bibtex==2.5.0 sphinx-rtd-theme==1.2.0
     python -m pip install pandas-datareader==0.10.0 yfinance==0.2.12 matplotlib>=3.6.0 seaborn>=0.11.2 jupyter-client>=6.1.2 jupyter-core>=4.6.3

Usage Examples
--------------

After installation, you can test the package using one of the provided examples. For instance, to run the maxcut example:

.. code-block:: bash

    cd examples/maxcut
    mpiexec -N 2 python3 maxcut.py

Documentation
=============

The documentation is built using Sphinx. After installing the optional documentation dependencies, generate the docs with:

.. code-block:: bash

    python setup.py build_sphinx

Building FFTW3 and HDF5 From Source
===================================

If pre-built parallel versions of FFTW3 and HDF5 are not available on your system, compile them from source as follows:

.. code-block:: bash

    # HDF5
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
    tar -xvf hdf5-1.10.6.tar.gz
    cd hdf5-1.10.6
    ./configure --enable-fortran --enable-shared --enable-parallel --prefix=/usr/local
    make && sudo make install
    cd ..

    # FFTW3
    wget http://www.fftw.org/fftw-3.3.8.tar.gz
    tar -xvf fftw-3.3.8.tar.gz
    cd fftw-3.3.8
    ./configure --enable-mpi --enable-fortran --enable-shared --prefix=/usr/local
    make && sudo make install
    cd ..

Environment Setup
=================

If QuOp_MPI is unable to locate the HDF5 or FFTW shared libraries, update your library path. Add the following line to your ~/.bashrc:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

Then, reload your environment:

.. code-block:: bash

    source ~/.bashrc

License
=======

QuOp_MPI is distributed under the GNU General Public License v3.0 (GPLv3). The full license text is available in the LICENSE file.

Contact Information
===================

For bug reports or inquiries, please submit an issue on GitHub or contact:

Edric Matwiejew  
Email: edric.matwiejew@research.uwa.edu.au
