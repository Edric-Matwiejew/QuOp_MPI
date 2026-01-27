.. _changelog:

=========
Changelog
=========

All notable changes to QuOp_MPI are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Version 1.3.1 (2026-01-27)
==========================

Fixed
-----

- Fixed FFTW MPI crash when system_size is 1 in circulant propagator
- Fixed deadlock in ``Ansatz.__post()`` when MPI ranks are excluded from subcommunicator
- Fixed profiler crash when importing module outside MPI context (e.g., Sphinx docs)
- Fixed docstring formatting warnings (unescaped asterisks in ``*.h5``)

Added
-----

- Added ``nlopt`` as optional dependency (``pip install QuOp_MPI[nlopt]``)
- Added comprehensive MPI test suite (384+ tests across 21 test files)
- Added unit tests for toolkit module (kronecker, pauli, string)
- Added unit tests for NLopt wrapper
- Reorganized API documentation into navigable multi-page structure

Changed
-------

- Renamed ``[all]`` optional dependency group to ``[dev]``
- Updated optional dependencies to use self-referencing extras
- Improved README documentation structure and clarity

Version 1.3.0 (2024-XX-XX)
==========================

Added
-----

- Parameter mapping support for custom variational parameter structures
- Job suspension and resumption for long-running benchmarks
- Profiler for performance analysis (``QUOP_PROFILE=1``)

Changed
-------

- Improved MPI subcommunicator handling for swarm meta-algorithm

Version 1.2.0 (2023-XX-XX)
==========================

Added
-----

- Multivariable optimization algorithms (QMOA)
- Momentum-space propagator
- Composite propagator for Cartesian sums

Version 1.1.0 (2022-XX-XX)
==========================

Added
-----

- QWOA algorithm implementation
- Sparse matrix propagator
- HDF5 save/load functionality

Version 1.0.0 (2021-XX-XX)
==========================

Initial release.

- Core Ansatz class for quantum variational algorithms
- QAOA algorithm implementation
- Circulant and diagonal propagators
- MPI-parallel distributed memory simulation
