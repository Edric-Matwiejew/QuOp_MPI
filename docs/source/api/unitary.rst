.. _api-unitary:

=======
Unitary
=======

The :class:`~quop_mpi.unitary` class is the base class for all unitary operators in QuOp_MPI. Custom propagators should inherit from this class.

Built-in propagators evolve through backend-owned native context buffers. The
``initial_state`` and ``final_state`` attributes exposed on
:class:`~quop_mpi.unitary` remain available only for custom legacy subclasses
that explicitly manage Python-side state buffers.

.. currentmodule:: quop_mpi

Class Reference
===============

.. autoclass:: Unitary
   :members:
   :exclude-members: gen_initial_params, gen_operator, parse_operator_function, parse_parameter_function
