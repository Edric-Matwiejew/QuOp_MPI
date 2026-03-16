QuOp_MPI
========

A Parallel Framework for Quantum Variational Algorithms
-------------------------------------------------------

QuOp_MPI is a Python 3 module designed for parallel, distributed-memory simulation of Quantum Variational Algorithms (QVAs) with arbitrary phase-shift and mixing operators.

<<<<<<< HEAD
**Current Version:** 1.5.0
=======
**Current Version:** 1.6.0
>>>>>>> quop_quisa/main

For an in-depth discussion on design, usage, and performance, see the `Journal of Computational Science paper <https://doi.org/10.1016/j.jocs.2022.101711>`_ (also available on `arXiv <https://arxiv.org/abs/2110.03963>`_).

Related Publications
--------------------

**Package Development:**

1. Matwiejew, E. & Wang, J. B. *QuOp_MPI: A framework for parallel simulation of quantum variational algorithms.* Journal of Computational Science 62, 101711 (2022).
2. Matwiejew, E. & Wang, J. *QSW_MPI: A framework for parallel simulation of quantum stochastic walks.* Computer Physics Communications 107724 (2020).

**Numerical Results Provided by QuOp_MPI:**

1. Bennett, T., Matwiejew, E., Marsh, S. & Wang, J. B. *Quantum Walk-Based Vehicle Routing Optimisation.* Frontiers in Physics 9, (2021).
2. Slate, N., Matwiejew, E., Marsh, S. & Wang, J. B. *Quantum walk-based portfolio optimisation.* Quantum 5, 513 (2021).
3. Matwiejew, E., Pye, J. & Wang, J. B. *Quantum optimisation for continuous multivariable functions by a structured search.* Quantum Science and Technology 8, 045012 (2023).
4. Qu, D., Matwiejew, E., Wang, K., Wang, J. & Xue, P. *Experimental implementation of quantum-walk-based portfolio optimization.* Quantum Science and Technology 9, 025014 (2024).
5. Pye, J., Matwiejew, E., Smith, A., Maylanda Kovalam, M. & Finn, L. S. *Gravitational-wave matched filtering with variational quantum algorithms.* arXiv:2408.13177 (2024).
6. Freeland, A. & Wang, J. B. *Quantum optimisation applied to the Quadratic Assignment Problem.* arXiv:2601.01104 (2026).

Citing QuOp_MPI
---------------

If you use QuOp_MPI in your research, please cite:

    Matwiejew, E. & Wang, J. B. QuOp_MPI: A framework for parallel simulation of quantum variational algorithms. *Journal of Computational Science* 62, 101711 (2022).

BibTeX entry:

.. code-block:: bibtex

    @article{matwiejew2022quop,
      title={QuOp\_MPI: A framework for parallel simulation of quantum variational algorithms},
      author={Matwiejew, Edric and Wang, Jingbo B},
      journal={Journal of Computational Science},
      volume={62},
      pages={101711},
      year={2022},
      publisher={Elsevier}
    }

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
    python -m pip install scikit-build-core cmake ninja

Next, choose one of the following build methods:

**Standard Build:**

To install from source (ensure that all build prerequisites are set), run:

.. code-block:: bash

    python -m pip install .

**Development Build:**

For development or modifying QuOp_MPI, use the following steps:

.. code-block:: bash

    cmake -B build -S .
    cmake --build build --target install
    python -m pip install -e .

**Environment Build:**

For a fully automated setup including virtual-environment creation, on specific platforms,
use the installer script with the appropriate profile and backend options:

.. code-block:: bash

    bash setup/install.sh -p macos -b mpi --prefix ./.quop-install              # macOS with Homebrew
    bash setup/install.sh -p generic -b mpi --prefix ./.quop-install            # generic Linux
    bash setup/install.sh -p pawsey-setonix -b wavefront --prefix /scratch/$USER/quop  # Pawsey Setonix GPU

Available profiles live under ``setup/sites/``.
Run ``./setup/install.sh --help`` for the full list of options.
Use ``--clean`` to clear the install cache under the chosen prefix, or
``--veryclean`` to clear both the cache and the fetched dependency tree.

Optional Dependencies
---------------------

For a full development environment with all optional dependencies:

.. code-block:: bash

   python -m pip install '.[dev]'

Alternatively, install only what you need:

.. code-block:: bash

   python -m pip install '.[examples]'  # Run example notebooks and scripts
   python -m pip install '.[docs]'      # Build documentation
   python -m pip install '.[test]'      # Run the test suite
   python -m pip install '.[nlopt]'     # Enable NLopt optimizer support

Backend Selection
-----------------

QuOp_MPI selects its compiled backend at process startup through the
``QUOP_BACKEND`` environment variable.

- Supported values are ``mpi`` and ``wavefront``.
- If ``QUOP_BACKEND`` is unset or invalid, QuOp_MPI defaults to ``mpi``.
- ``QUOP_BACKEND`` is read once, when ``quop_mpi`` is first imported.
- Changing ``QUOP_BACKEND`` after import has no effect.
- Mixed-backend use in a single Python process is unsupported.

Set the backend before importing ``quop_mpi``:

.. code-block:: bash

    QUOP_BACKEND=mpi python -m pytest tests/
    QUOP_BACKEND=wavefront python -m pytest tests/ --backend wavefront

The wavefront backend must also be built into the current installation. A
default build only guarantees the MPI backend; wavefront extensions are only
available when QuOp_MPI is configured and built with ``-DWAVEFRONT_BACKEND=ON``
and its required GPU dependencies.

Running Tests
=============

Install test dependencies with ``python -m pip install '.[test]'``, then:

.. code-block:: bash

    ./run_tests.sh              # all tests (unit + MPI + integration), 2 MPI processes
    ./run_tests.sh 4            # same, with 4 MPI processes
    ./run_tests.sh 1 unit       # unit tests only (serial)
    ./run_tests.sh 2 mpi        # MPI tests only
    ./run_tests.sh 2 mpi-full   # MPI + parallel-Jacobian tests (12 processes)

The ``all`` mode (default) also runs integration tests, which execute the
example scripts and check that optimisation results match the expected bounds
in ``tests/integration/expected_results.json``.

To select the wavefront backend, set ``QUOP_BACKEND=wavefront`` before
running.  On Cray systems the script auto-detects ``srun``.

Usage Examples
--------------

After installation, you can test the package using one of the provided examples. For instance, to run the maxcut example:

.. code-block:: bash

    cd examples/maxcut
    mpiexec -N 2 python3 maxcut.py

Documentation
=============

After installing ``.[docs]``, build the documentation with:

.. code-block:: bash

    python -m sphinx -b html docs/source docs/build/html

Building FFTW3 and HDF5 From Source
===================================

To compile FFTW3 and HDF5 from source:

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
Email: edric_matwiejew@CSIRO.au
