.. _changelog:

=========
Changelog
=========

All notable changes to QuOp_MPI are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
==========

Added
-----

- Comprehensive MPI-enabled test suite with pytest-mpi (380+ tests)
- Unit tests for toolkit module (kronecker, pauli, string functions)
- Unit tests for NLopt wrapper module
- Performance profiling support via ``QUOP_PROFILE=1`` environment variable
- ``set_objective`` method for custom objective functions
- Documentation restructure with navigable API reference
- Quick Start example in documentation
- Changelog documentation

Changed
-------

- **Major refactor of Ansatz.py**: Extracted cohesive subsystems into mixins for improved maintainability

  - ``_sampling.py``: Simulated quantum measurement functionality
  - ``_logging.py``: CSV logging and HDF5 parallel I/O
  - ``_communicator.py``: MPI subcommunicator management
  - ``_optimization/parallel_jacobian.py``: Parallel gradient computation
  - ``_optimization/finite_differences.py``: Numerical gradient approximation methods
  - ``_benchmark.py``: Systematic depth study workflow

- Reorganized internal module structure with consistent naming conventions

  - Renamed ``__lib/`` → ``_lib/``, ``__utils/`` → ``_utils/``, ``__profile/`` → ``_profile/``
  - Renamed internal modules from ``__*.py`` to ``_*.py`` (single underscore for private modules)
  - Moved ``_utils/_nlopt_wrap.py`` → ``_optimization/nlopt_wrap.py``

- Removed orphaned ``algorithm/multivariable.py`` (duplicate of ``algorithm/multivariable/``)
- Refactored optional dependencies in pyproject.toml (``[dev]`` replaces ``[all]``)
- Updated documentation sidebar with grouped navigation
- NumPy 2.x compatibility

Fixed
-----

- FFTW MPI crash with system_size=1 (edge case in circulant propagator)
- Deadlock in ``Ansatz.__post()`` when MPI ranks are excluded from subcomm
- Profiler crash during Sphinx doc builds (outside MPI context)
- OpenMP thread contention causing apparent deadlock with >2 MPI processes
- Circulant eigenvalue generation bug when graph is complete
- Parallel Jacobian implementation compatibility with parameter maps
- Docstring formatting warnings (unescaped asterisks)

Version 1.2.1 (2025-03-07)
==========================

Changed
-------

- Build system update with modern GitHub Actions workflow
- Bumped dawidd6/action-download-artifact from 2 to 6
- Updated dependency version ranges
- Removed outdated test directory

Version 1.2.0 (2024-08-13)
==========================

Added
-----

- Parameter mapping functionality for flexible variational parameter control (#11, #12)
- Parallel gradient evaluation with MPI subcommunicators
- Broadcast of variational parameters from root of MPI_COMM_WORLD
- Basic profiling functionality with OpenMP support in release builds
- Interleaved local spMV with parallel communication for improved performance
- Documentation build GitHub workflow

Changed
-------

- CMake build requires cmake>=3.5,<4
- Removed self.Ns from operator dict

Version 1.1.0 (2022-10-12)
==========================

Added
-----

- Composite Ansatz class for multivariable optimization problems
- Multivariable optimization examples
- QMOA (Quantum Multivariable Optimization Algorithm) experiments
- Link to preprint article

Changed
-------

- Evaluate returns results to all calling MPI ranks
- Support for Python 3.6 and 3.7 builds

Version 1.0.0 (2021-09-30)
==========================

This is the first major release, coinciding with the `Journal of Computational Science publication <https://doi.org/10.1016/j.jocs.2022.101711>`_.

Added
-----

- Complete API redesign with modular Ansatz architecture
- QAOA and QWOA algorithm implementations  
- Circulant, diagonal, and sparse propagator classes
- Observable and state modules
- Toolkit for common quantum operators (Pauli matrices, Kronecker products)
- MPI-parallel state vector simulation
- Parallel HDF5 I/O for saving and loading simulation data
- Support for NLopt optimizers alongside SciPy
- Benchmark infrastructure for performance evaluation
- Comprehensive documentation with Sphinx
- GitHub Actions CI/CD pipeline
- Installation scripts for Ubuntu 20.04
- Singularity container support
- License (GPLv3) and contributor acknowledgements

Changed
-------

- Refactored pre-post execution cycle for unitaries
- Communicator resizing for optimal MPI process utilization
- Default optimizer changed to L-BFGS-B
- Parallel post methods with total_params tracking

Fixed
-----

- Circulant operators FFTW compatibility
- Deadlock issues in communicator shrinking
- Various MPI edge cases for ranks with zero-length local arrays
- evolve_state converts list parameters into ndarray
- Check for pre or post on cold call to evolve_state

Version 0.0.2 (2020-02-23)
==========================

Added
-----

- Zenodo DOI for citation
- Support for multiple mixing operators in QAOA class
- Custom optimizer support beyond default SciPy minimize
- Basin-hopping optimization option
- Process-independent quality generation using integers or random floats
- Updated sparse matrix exponentiation subroutines
- Benchmark construct with parameter reuse option
- Parallel HDF5 write functionality
- MPI hypercube mixer
- Objective function mapping to custom scalar values
- MacOS installation support

Changed
-------

- QuOp_MPI now minimizes the objective function (was maximizing)
- Optimizer result 'success' field renamed to 'optimizer_success'
- 'success' metric replaced by 'quality cutoff'
- Method ``log_success`` replaced by ``log_results`` (**breaking change from 0.0.1**)
- Switched to L-BFGS-B optimizer
- Split QAOA and QWOA into separate classes
- Networkx added as required dependency

Fixed
-----

- Bug fix in MPI.py (#1)
- Handling of zero-length quality arrays at local MPI process
- Initial state handling when rank 0 has local_i = 0
- Fourier transform ordering
- Initial state defaults to equal superposition if undefined

Version 0.0.1 (2020-01-12)
==========================

Initial release.

- QWAO_MPI: Quantum Walk-Assisted Optimizer with MPI parallelization
- Basic CTQW (Continuous-Time Quantum Walk) implementation
- Parallel eigenvalue computation with FFTW
- Parallel HDF5 output support
- Draft documentation
