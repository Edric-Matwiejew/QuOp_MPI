.. _backend_architecture:

==========================================
QuOp_MPI Backend & Software Architecture
==========================================

This section documents the backend architecture and implementation-level
details for QuOp_MPI. The pages below are aligned with the Fortran sources in
``native/comm_info`` and ``native/wavefront``.

Implementation details are aligned with the current Fortran sources in:

- ``native/comm_info/comm_info_module.f90``
- ``native/mpi/mpi_context.f90``
- ``native/wavefront/context/wavefront_context.f90``
- ``native/wavefront/context/gpu_transfer.f90``

Backend Overview
================

QuOp_MPI supports two execution backends:

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - Backend
     - Summary
     - Primary hardware
   * - ``mpi``
     - Host-only state representation with MPI collectives over ``SUBCOMM``.
     - CPU clusters
   * - ``wavefront``
     - Device-resident state representation with hierarchical communicators and
       host/device transfer collectives.
     - HIP-capable GPU clusters

Both backends use the same negotiated layout object (``quop_mpi_layout_t``),
created in ``comm_info_module``.

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - MPI backend
     - Wavefront backend
   * - Primary state location
     - Host memory
     - Device memory (host staging for transfers/reductions)
   * - Runtime communicators
     - ``SUBCOMM`` (+ ``NODECOMM`` available from layout)
     - ``SUBCOMM``, ``NODECOMM``, ``DEVCOMM``, ``DEVCOMM_NODE``
   * - Expectation/norm reduction
     - Host local compute + ``MPI_Allreduce(SUBCOMM)``
     - GPU kernel + host partial sum + ``MPI_Allreduce(SUBCOMM)``
   * - State I/O path
     - Host buffers only
     - ``gpu_allscatterv_htod`` / ``gpu_allgatherv_dtoh``
   * - Device ownership logic
     - Not applicable
     - ``has_device`` from ``DEVCOMM`` membership

Runtime Environment Variables
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``QUOP_BACKEND``
     - Select Python backend path (``mpi`` or ``wavefront``) at import time.
   * - ``QUOP_RANKS_PER_GPU``
     - Controls topology assignment density for wavefront GPU ranks.
   * - ``QUOP_FORCE_STAGED_HTOD``
     - In GPU-aware builds, force staged host->device transfers.
   * - ``QUOP_PROFILE``
     - Enable runtime profiling output when set to ``1``.

Build-time controls are documented in :doc:`../build_system`.

Implementation Details
======================

.. toctree::
   :maxdepth: 1

   communicator_structure
   quop_mpi_layout_flowchart
   statevector_lifecycle
   gpu_transfer
   wavefront_layout_256_16ranks_2gpus
