``quop_mpi_layout_t`` Flow (Current Implementation)
===================================================

This page documents the current Fortran flow in ``native/comm_info`` and the
f2py-facing wrappers in ``native/comm_info/comm_info_wrapper.f90``.

Phase Flow
----------

.. mermaid::

   flowchart TD
       A[wrapper_discover_topology] --> B[wrapper_split_workers]
       B --> C[wrapper_negotiate]
       C --> D{negotiate status}
       D -->|active or excluded| E[wrapper_create_rootcomm]
       D -->|active or excluded and multiple workers| F[wrapper_create_jaccomm]
       D -->|error status| G[caller cleanup and raise]
       E --> I[post-negotiate communicator refresh complete]
       F --> I
       I --> J{status is zero}
       J -->|yes| H[active runtime path]
       J -->|no excluded| K[excluded-rank path]

Top-Level Calls
---------------

1. ``wrapper_discover_topology(topo_ptr, MPI_COMM, backend_flag, error_code)``
   is collective on ``MPI_COMM`` and builds ``gpu_topology_t`` (full on
   wavefront, defaults on MPI backend).
2. ``wrapper_split_workers(split_ptr, MPI_COMM, topo_ptr, n_jacobian_workers, backend_flag, worker_id, status)``
   is collective on ``MPI_COMM`` and creates worker ``SUBCOMM`` plus an initial
   ``ROOTCOMM`` in ``split_info_t``.
3. ``wrapper_negotiate(layout_ptr, split_ptr, topo_ptr, system_size, backend_flag, n_propagators, propagator_ptrs, n_callbacks, callback_ptrs, status)``
   is collective on active worker ``SUBCOMM`` and allocates
   ``quop_mpi_layout_t``; lock is reached only on ``status == 0``.
4. ``wrapper_create_rootcomm(MPI_COMM, split_ptr, layout_ptr)`` is collective
   on ``MPI_COMM`` and rebuilds ``ROOTCOMM`` from post-negotiate ``SUBCOMM``
   leaders. In the current ``Ansatz`` path, this is called for both
   ``status == 0`` and ``status == -1``.
5. ``wrapper_create_jaccomm(MPI_COMM, split_ptr, layout_ptr)`` is collective on
   ``MPI_COMM`` and builds the Jacobian communicator with current worker-role
   rules. In the current ``Ansatz`` path, this is called only when
   ``n_workers > 1`` (and for both ``status == 0`` and ``status == -1``).

Both ``wrapper_create_rootcomm`` and ``wrapper_create_jaccomm`` have guards for
``layout_ptr == 0`` and return null communicator membership in that case.

Inside ``negotiate``
--------------------

Phase 1: create
~~~~~~~~~~~~~~~

- Allocate ``quop_mpi_layout_t``.
- Copy ``MPI_COMM``, ``system_size``, backend flag, and topology.
- Transfer ``SUBCOMM`` ownership from ``split_info_t``.
- Create ``NODECOMM`` (all backends).
- Wavefront: create ``DEVCOMM``/``DEVCOMM_NODE`` from topology.

Phase 2: negotiate loop
~~~~~~~~~~~~~~~~~~~~~~~

- Start from block distribution.
- For each propagator, call callback pointer (C trampoline) with
  ``(prop_ptr, ci_ptr, error_code)``.
- Callback may request smaller communicator (by lowering ``ci%n_processes``)
  or modify partition fields.
- If ``n_processes`` shrinks: ``layout_shrink`` + redistribute + continue loop.
- Stability uses a collective check and a confirmation pass before convergence.

Phase 3: finalise
~~~~~~~~~~~~~~~~~

- Ensure ``alloc_local >= local_i``.
- Build partition table.

Phase 4: validate
~~~~~~~~~~~~~~~~~

- Run collective validation checks (non-negative, completeness, ordering,
  contiguity, plus device checks when applicable).

Phase 5: lock
~~~~~~~~~~~~~

- Lock layout to prevent further mutation.

Status Codes from ``negotiate``
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Status
     - Meaning
   * - ``0``
     - Success, rank is active, layout locked.
   * - ``-1``
     - Rank excluded (``SUBCOMM`` became ``MPI_COMM_NULL`` after shrink).
   * - ``1``
     - Invalid ``system_size`` (<= 0).
   * - ``3``
     - Negotiation did not converge within iteration cap.
   * - ``4``
     - Shrink/finalization failure during negotiate.
   * - ``5``
     - Block distribution failure.
   * - ``100 + x``
     - Validation failure.
   * - ``200 + x``
     - Partition table finalization failure.
   * - ``300 + x``
     - Lock failure.
   * - ``1000 + x``
     - Propagator callback returned ``x``.

Status Codes from ``split_workers``
-----------------------------------

- ``0``: success
- ``1``: invalid ``n_jacobian_workers``

Communicator Mutation Points
----------------------------

- ``split_workers``: creates pre-negotiate ``SUBCOMM`` and ``ROOTCOMM``.
- ``negotiate``: may shrink ``SUBCOMM`` and rebuild child communicators.
- ``create_rootcomm``: refreshes ``ROOTCOMM`` for post-negotiate leaders.
- ``create_jaccomm``: refreshes ``JACCOMM`` based on worker role (typically
  only when ``n_workers > 1``).

Signature References
--------------------

.. code-block:: fortran

   subroutine wrapper_negotiate(layout_ptr, split_ptr, topo_ptr, &
                                system_size, backend_flag, &
                                n_propagators, propagator_ptrs, &
                                n_callbacks, callback_ptrs, status)

.. code-block:: fortran

   subroutine negotiate(layout_ptr, split_ptr, topo_ptr, &
                        system_size, backend_flag, &
                        n_propagators, propagator_ptrs, &
                        callback_ptrs, status)

.. code-block:: fortran

   subroutine create_rootcomm(MPI_COMM, split_ptr, layout_ptr)

.. code-block:: fortran

   subroutine create_jaccomm(MPI_COMM, split_ptr, layout_ptr)
