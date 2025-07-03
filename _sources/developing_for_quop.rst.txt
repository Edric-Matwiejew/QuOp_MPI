Developing for QuOp_MPI
=======================

Propagator Modules
------------------

A :mod:`~quop_mpi.propagator` module consists of an :class:`~quop_mpi.Unitary` that implements the methods:

* :meth:`~quop_mpi.Unitary.plan`
* :meth:`~quop_mpi.Unitary.copy_plan`
* :meth:`~quop_mpi.Unitary.propagate`
* :meth:`~quop_mpi.Unitary.destroy`

The :meth:`~quop_mpi.Unitary.propagate` method computes the action of a :term:`phase-shift <phase-shift unitary>` or :term:`mixing <mixing unitary>` using a method that is compatible with QuOp_MPI's :ref:`parallel partitioning scheme for Quantum simulation <parallel-QVA>`. The :meth:`~quop_mpi.Unitary.plan` and :meth:`~quop_mpi.unitary.copy_plan` handle the generation of parallel partitioning schemes and the copying of parallel partitioning schemes from other ``unitary`` instances. The :meth:`~quop_mpi.Unitary.plan` and :meth:`~quop_mpi.Unitary.destroy` methods handle the allocation and freeing of memory that is managed by Python extension modules that are called by :meth:`~quop_mpi.Unitary.propagate`.

A :meth:`~quop_mpi.propagator` module should also implement an ``operator`` submodule containing :term:`Operator Functions <Operator Function>` that are compatible with the implemented :meth:`~quop_mpi.Unitary.propagate` method.