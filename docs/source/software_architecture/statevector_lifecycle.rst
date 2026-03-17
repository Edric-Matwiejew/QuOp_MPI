Statevector Lifecycle (Wavefront Backend)
=========================================

Scope
-----

This page describes the current wavefront state/observable lifecycle implemented
in:

- ``src/wavefront/context/wavefront_context.f90``
- ``src/wavefront/sparse/wavefront_sparse.f90``
- ``src/wavefront/circulant/wavefront_circulant.f90``

Key Arrays and Ownership
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 18 18 40

   * - Array
     - Location
     - Owner
     - Notes
   * - ``context%state``
     - Device
     - ``wavefront_context``
     - Main device state buffer, size ``ci%get_device_alloc_local()``.
   * - ``context%work``
     - Device
     - ``wavefront_context``
     - Optional scratch buffer; allocated only if
       ``ci%get_requires_device_work_buffer()`` is true.
   * - ``context%observables``
     - Device
     - ``wavefront_context``
     - Device observable slice, size ``ci%get_device_local_i()``.
   * - ``context%reduction_dout``
     - Device
     - ``wavefront_context``
     - Persistent reduction output buffer (``reduction_num_blocks = 1200``).
   * - ``context%reduction_host_out``
     - Host
     - ``wavefront_context``
     - Host staging array for reduction buffer copies.
   * - ``NODECOMM_counts/displs``
     - Host
     - ``wavefront_context``
     - Per-NODECOMM-rank host partition metadata.
   * - ``DEVCOMM_NODE_counts/displs``
     - Host
     - ``wavefront_context``
     - Per-NODECOMM-rank device partition metadata.
   * - Sparse CSR device data
     - Device
     - ``wavefront_sparse``
     - Managed by sparse graph communication setup/cleanup.
   * - ``circulant%dev_eigenvalues``
     - Device
     - ``wavefront_circulant``
     - Reallocated on each ``gen_operator``.

Setup and Transfer Metadata
---------------------------

During ``context%setup(ci, error_code)``:

1. ``ci`` pointer is borrowed and cached.
2. ``has_device`` is set from ``ci%get_DEVCOMM()`` membership.
3. Transfer metadata arrays are allocated and filled with ``MPI_Allgather`` over
   ``NODECOMM`` using:
   - host partition: ``local_i``, ``local_i_offset``
   - device partition: ``device_local_i``, ``device_local_i_offset``
4. Device allocations occur only on ``has_device`` ranks.
5. ``state``, ``work`` (if allocated), and ``observables`` are zeroed with
   ``hipMemset``.
6. Reduction buffers are allocated once and reused by
   ``get_expectation_value`` and ``get_state_norm``.

State/Observable I/O
--------------------

The context exposes four collective transfer operations:

- ``set_state``: host -> device via ``gpu_allscatterv_htod``
- ``get_state``: device -> host via ``gpu_allgatherv_dtoh``
- ``set_observables``: host -> device via ``gpu_allscatterv_htod``
- ``get_observables``: device -> host via ``gpu_allgatherv_dtoh``

Current behavior:

- Collective error checks are synchronized on ``SUBCOMM``.
- Each routine brackets transfer calls with ``MPI_Barrier(SUBCOMM)``.
- Non-GPU ranks pass ``C_NULL_PTR`` for device pointers.

Propagation Paths
-----------------

Sparse path
~~~~~~~~~~~

- ``wavefront_sparse%propagate`` calls ``Chebyshev_Multiply`` on ``DEVCOMM``.
- Output is written to ``context%work``.
- ``context%state`` and ``context%work`` are pointer-swapped after each call.
- A ``MPI_Barrier(SUBCOMM)`` follows the sparse propagate call.

Circulant path
~~~~~~~~~~~~~~

- ``wavefront_circulant%propagate`` uses SHAFFT over ``DEVCOMM``.
- The code calls ``shafftSetBuffers`` before forward FFT.
- After forward FFT and normalization, ``shafftGetBuffers`` refreshes
  ``context%state``/``context%work`` pointers.
- Phase shift kernel runs on ``context%state``.
- ``shafftSetBuffers`` is called again before backward FFT to ensure SHAFFT uses
  the updated pointers.
- Final ``shafftGetBuffers`` refreshes pointers again.

Reductions (No Full State Transfer)
-----------------------------------

- ``get_expectation_value`` launches a GPU reduction kernel, copies
  ``reduction_dout`` to host, sums locally, then ``MPI_Allreduce`` on ``SUBCOMM``.
- ``get_state_norm`` follows the same pattern with a norm kernel.
- Ranks without devices contribute ``0`` to the ``SUBCOMM`` allreduce.

Pointer Stability Rules
-----------------------

- Treat ``context%state`` as a logical handle, not a stable address.
- Sparse propagation explicitly swaps ``state`` and ``work`` pointers.
- Circulant propagation may reorder buffers through SHAFFT and then rebinds
  pointers via ``shafftGetBuffers``.
- Do not cache ``c_loc(context%state)`` across propagate calls.

Destruction
-----------

``context%destroy()`` currently:

- Re-selects ``device_ID`` on GPU ranks.
- Synchronizes device.
- Frees ``work``, ``state``, ``observables``, and reduction buffers if allocated.
- Deallocates transfer metadata arrays.
- Nullifies borrowed ``ci`` pointer.

Important current behavior:

- No ``MPI_Barrier`` is executed in ``context_destroy``.
- This avoids destructor deadlocks when called through Python GC timing.

Collective Participation Summary
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Operation
     - Required participants
   * - ``set_state/get_state``
     - All active ``SUBCOMM`` ranks (transfer core runs on ``NODECOMM``).
   * - ``set_observables/get_observables``
     - All active ``SUBCOMM`` ranks (transfer core runs on ``NODECOMM``).
   * - ``get_expectation_value/get_state_norm``
     - All active ``SUBCOMM`` ranks.
   * - Sparse propagate
     - Device-owning ranks compute on ``DEVCOMM``; routine still barriers on
       ``SUBCOMM``.
   * - Circulant propagate
     - Device-owning ranks in ``DEVCOMM`` participate in SHAFFT collectives.

Error Handling Notes
--------------------

- Most context/circulant API-level failures are surfaced via ``error_code``.
- Circulant SHAFFT calls return non-zero ``error_code`` on failure.
- Some lower-level checks still abort collectively (for example invalid GPU
  topology/binding configuration), and HIP utility behavior depends on the
  linked ``hipfort_check`` implementation.
