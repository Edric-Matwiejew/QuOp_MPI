GPU Transfer Layer (``gpu_transfer.f90``)
=========================================

The active transfer implementation is in
``src/wavefront/context/gpu_transfer.f90``.

It replaces the older ``hipMPIMemcpy`` helper model and exposes two public,
layout-aware collectives used by ``wavefront_context``.

Public APIs
-----------

.. code-block:: fortran

   subroutine gpu_allscatterv_htod(host_counts, dev_counts, host_displs, &
                                   dev_displs, host_ptr, dev_ptr, mpi_type, &
                                   NODECOMM)

   subroutine gpu_allgatherv_dtoh(dev_counts, host_counts, dev_displs, &
                                  host_displs, dev_ptr, host_ptr, mpi_type, &
                                  NODECOMM)

Argument Semantics
------------------

- ``*_counts`` and ``*_displs`` are per-rank global partitions over
  ``NODECOMM`` (element units, not bytes).
- Arrays are indexed by ``NODECOMM`` rank and must have length
  ``MPI_Comm_size(NODECOMM)``.
- ``mpi_type`` defines element type/size (for example ``MPI_DOUBLE_COMPLEX`` or
  ``MPI_DOUBLE``).
- ``host_ptr`` and ``dev_ptr`` are raw C pointers; non-GPU ranks may pass
  ``C_NULL_PTR`` for ``dev_ptr``.

Execution Modes
---------------

Compile-time selection:

- If ``QUOP_GPU_AWARE_MPI`` is defined, use the GPU-aware MPI path (device
  pointers participate directly in MPI).
- If ``QUOP_GPU_AWARE_MPI`` is not defined, use the host-staged path (MPI on
  host staging buffers with host<->device copies as required by direction).

Runtime override (GPU-aware builds only):

- ``QUOP_FORCE_STAGED_HTOD=1`` forces staged ``host -> device`` path for
  diagnostics.

Algorithm Details
-----------------

GPU-aware MPI path
~~~~~~~~~~~~~~~~~~

For each rank:

1. Compute local source/destination overlap.
2. Copy that local overlap directly with ``hipMemcpy`` (no MPI).
3. Build per-partner send/recv schedule from interval overlap
   (``build_transfer_schedule``).
4. Exchange non-local segments with MPI using device pointers.

Host-staged path
~~~~~~~~~~~~~~~~

For each rank:

1. Stage local source payload between device and host when needed:
   - DtoH: copy full local device chunk to host staging buffer.
   - HtoD: allocate host staging buffer for incoming device chunk.
2. Build per-partner send/recv schedule from interval overlap
   (``build_transfer_schedule``).
3. Run ``MPI_Alltoallv`` on host staging buffers (byte-derived MPI datatype).
4. HtoD only: copy received staging payload from host to device.

Safety/Correctness Details
--------------------------

- Device synchronization is used at key points before exposing GPU buffers to
  MPI and after receiving device-target traffic.
- MPI count/displacement arguments are 32-bit; conversion uses ``safe_int32``
  with overflow abort protection.
- Staged path uses an MPI byte-derived datatype to move raw staged payloads
  while preserving element sizing from ``mpi_type``.

Where the Count/Displ Arrays Come From
--------------------------------------

``wavefront_context%setup`` builds four arrays with ``MPI_Allgather`` on
``NODECOMM``:

- ``NODECOMM_counts``
- ``NODECOMM_displs``
- ``DEVCOMM_NODE_counts``
- ``DEVCOMM_NODE_displs``

These arrays are then passed directly to ``gpu_allscatterv_htod`` and
``gpu_allgatherv_dtoh`` for state and observable transfers.
