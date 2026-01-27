.. _api-ansatz:

======
Ansatz
======

The :class:`~quop_mpi.Ansatz` class is the core building block for quantum variational algorithms in QuOp_MPI. It manages the quantum state, unitary operators, observables, and classical optimization.

.. currentmodule:: quop_mpi

Class Reference
===============

.. autoclass:: Ansatz
   :members:
   :exclude-members: setup, destroy
   :show-inheritance:

   .. rubric:: Key Methods

   **Configuration**

   - :meth:`set_unitaries` - Define the sequence of unitary operators
   - :meth:`set_observables` - Set the observable for computing expectation values
   - :meth:`set_initial_state` - Define the initial quantum state

   **Execution**

   - :meth:`execute` - Run the variational algorithm
   - :meth:`benchmark` - Study performance across ansatz depths

   **Results**

   - :meth:`save` - Save results to HDF5
   - :meth:`print_result` - Display optimization results
