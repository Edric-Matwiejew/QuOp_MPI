.. _api-propagators:

===========
Propagators
===========

Propagators implement the unitary time evolution for different types of operators. Choose the appropriate propagator based on your operator's structure.

.. automodule:: quop_mpi.propagator

.. list-table:: Propagator Selection Guide
   :header-rows: 1
   :widths: 20 40 40

   * - Propagator
     - Use When
     - Example Applications
   * - :doc:`circulant`
     - Operator is circulant (translation-invariant)
     - QWOA, constrained combinatorial optimisation
   * - :doc:`diagonal`
     - Operator is diagonal in computational basis
     - Phase operators, cost Hamiltonians
   * - :doc:`sparse`
     - Operator is a general sparse matrix
     - QAOA, custom Hamiltonians
   * - :doc:`composite`
     - Operator is a Cartesian sum of circulant operators mixing over subregisters
     - Discretised continuous multivariable optimisation, combinatorial problems with decision variables of arity >= 3
   * - :doc:`momentum`
     - State evolves in momentum space
     - Gradient-descent search inspired by continuous-variable photonic quantum computing

.. toctree::
   :maxdepth: 1

   circulant
   diagonal
   sparse
   composite
   momentum
