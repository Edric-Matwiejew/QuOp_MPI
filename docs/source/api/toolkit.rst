.. _api-toolkit:

=======
Toolkit
=======

Utility functions for constructing quantum operators and states.

.. currentmodule:: quop_mpi.toolkit

Module Reference
================

.. automodule:: quop_mpi.toolkit
   :members:

Pauli Matrices
--------------

Functions for generating Pauli operators on multi-qubit systems:

- :func:`I` - Identity operator
- :func:`X` - Pauli-X operator on specified qubit
- :func:`Y` - Pauli-Y operator on specified qubit  
- :func:`Z` - Pauli-Z operator on specified qubit

Kronecker Products
------------------

Functions for tensor products:

- :func:`kron` - Kronecker product of a list of matrices
- :func:`kron_power` - n-fold tensor product of a matrix

State Preparation
-----------------

- :func:`string` - Convert bit-string to computational basis state
