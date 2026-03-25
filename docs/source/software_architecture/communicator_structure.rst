QuOp MPI Communicator Structure (Current)
=========================================

Overview
--------

The communicator hierarchy is implemented in
``native/comm_info/comm_info_module.f90`` with two core objects:

- ``split_info_t``: pre/post-negotiate worker metadata (``SUBCOMM``,
  ``ROOTCOMM``, ``JACCOMM``, worker ids)
- ``quop_mpi_layout_t``: active execution layout (partitioning + owned runtime
  communicators)

The current code path is backend-aware:

- MPI backend: ``NODECOMM`` is created, ``DEVCOMM``/``DEVCOMM_NODE`` stay
  ``MPI_COMM_NULL``
- Wavefront backend: ``NODECOMM``, ``DEVCOMM``, and ``DEVCOMM_NODE`` are
  created and can be rebuilt during negotiate

Communicator Set
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Name
     - Owner
     - Meaning
   * - ``MPI_COMM``
     - Borrowed by both objects
     - Root communicator passed in by caller. Never freed by QuOp.
   * - ``SUBCOMM``
     - ``split_info_t`` then ``quop_mpi_layout_t``
     - Per-worker communicator used by negotiate and runtime collectives.
   * - ``NODECOMM``
     - ``quop_mpi_layout_t``
     - Shared-memory node-local communicator derived from ``SUBCOMM``.
   * - ``DEVCOMM``
     - ``quop_mpi_layout_t``
     - All active GPU ranks across nodes (wavefront only).
   * - ``DEVCOMM_NODE``
     - ``quop_mpi_layout_t``
     - Active GPU ranks on the local node (wavefront only).
   * - ``ROOTCOMM``
     - ``split_info_t``
     - Rank-0 leader from each post-negotiate ``SUBCOMM``.
   * - ``JACCOMM``
     - ``split_info_t``
     - Optional Jacobian communicator (optimizer rank + Jacobian worker ranks);
       typically created only when ``n_workers > 1``.

Ownership Model
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Object
     - Owns
     - Destroy behavior
   * - ``split_info_t``
     - ``SUBCOMM`` (until negotiate), ``ROOTCOMM``, ``JACCOMM``
     - ``split_info_destroy`` frees owned communicators.
   * - ``quop_mpi_layout_t``
     - ``SUBCOMM``, ``NODECOMM``, ``DEVCOMM``, ``DEVCOMM_NODE``
     - ``layout_destroy`` frees owned communicators.

``MPI_COMM`` is never freed by either object.

Lifecycle
---------

1. ``discover_topology(MPI_COMM, backend_flag)`` detects node/global topology,
   with wavefront additionally building hardware GPU topology metadata.
2. ``split_workers(...)`` creates worker ``SUBCOMM`` (single-worker case uses
   ``MPI_Comm_dup``), creates an initial ``ROOTCOMM`` from pre-negotiate
   subcomm leaders, and stores ``worker_id``/``n_workers`` in ``split_info_t``.
3. ``negotiate(...)`` allocates ``quop_mpi_layout_t``, transfers ``SUBCOMM``
   ownership from ``split_info_t``, creates ``NODECOMM`` (all backends), creates
   ``DEVCOMM``/``DEVCOMM_NODE`` on wavefront, and may shrink ``SUBCOMM`` with
   child-communicator rebuilds.
4. ``create_rootcomm(MPI_COMM, split_ptr, layout_ptr)`` rebuilds ``ROOTCOMM``
   from rank 0 of each post-negotiate ``SUBCOMM``.
5. ``create_jaccomm(MPI_COMM, split_ptr, layout_ptr)`` builds ``JACCOMM`` with
   current worker-role semantics (``worker_id == 0`` joins only on ``SUBCOMM``
   rank 0, while ``worker_id > 0`` joins on all ranks). In the current Python
   ``Ansatz`` path, this step is skipped when ``n_workers == 1``.

Current Flow
------------

.. mermaid::

   flowchart TD
       A[MPI_COMM] --> B[discover_topology]
       B --> C[split_workers]
       C --> D[split_info_t: SUBCOMM and ROOTCOMM]
       D --> E[negotiate]
       E --> F[quop_mpi_layout_t: SUBCOMM, NODECOMM, DEVCOMM on wavefront]
       F --> G[create_rootcomm]
       F --> H[create_jaccomm optional]
       G --> I[split_info_t ROOTCOMM refreshed]
       H --> J[split_info_t JACCOMM ready]

Backend-Specific Notes
----------------------

- ``layout_shrink`` always rebuilds ``SUBCOMM`` and ``NODECOMM``; wavefront
  also rebuilds ``DEVCOMM`` and ``DEVCOMM_NODE``.
- ``layout_rebuild_communicators`` is wavefront-only in practice (MPI backend
  is a no-op for device communicators).
- ``device_n_processes`` in the layout tracks active device ranks on
  ``NODECOMM`` (node-local count), not global ``DEVCOMM`` size.

Collective Requirements
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Operation
     - Collective communicator
   * - ``discover_topology``
     - ``MPI_COMM``
   * - ``split_workers``
     - ``MPI_COMM``
   * - ``negotiate``
     - Active ranks on ``SUBCOMM``
   * - ``create_rootcomm``
     - ``MPI_COMM`` (all ranks call)
   * - ``create_jaccomm``
     - ``MPI_COMM`` (all ranks call)

Destruction Order
-----------------

For safe teardown, destroy in this order:

1. Propagator/context users that borrow layout communicators
2. ``quop_mpi_layout_t`` (frees runtime communicators)
3. ``split_info_t`` (frees ``ROOTCOMM``/``JACCOMM`` and any remaining ``SUBCOMM``)

This order avoids dangling communicator handles in borrowed objects.
