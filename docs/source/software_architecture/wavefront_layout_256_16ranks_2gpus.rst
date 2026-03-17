Wavefront Layout Example: ``N=256``, ``16`` Ranks, ``2`` GPUs
==============================================================

This example matches the current wavefront layout logic in:

- ``src/comm_info/comm_info_module.f90`` (``device_block_distribute``)
- ``src/wavefront/context/partitions.f90``
- ``src/wavefront/context/wavefront_context.f90``

Assumptions
-----------

- One worker subcommunicator with ``|SUBCOMM| = 16``.
- One node, so ``NODECOMM == SUBCOMM``.
- Exactly two ranks are active GPU ranks in ``DEVCOMM``.
- Partitioning remains block-based (for example sparse/diagonal path, no
  SHAFFT override).

Step 1: Device Partition (``DEVCOMM``)
--------------------------------------

``device_block_distribute`` first block-distributes ``system_size`` over
``DEVCOMM`` ranks.

With ``N = 256`` and ``|DEVCOMM| = 2``:

- Device rank 0: ``device_local_i = 128``, ``device_local_i_offset = 0``
- Device rank 1: ``device_local_i = 128``, ``device_local_i_offset = 128``

So device-resident state chunks are ``[0,127]`` and ``[128,255]``.

Step 2: Host Partition (``NODECOMM`` / ``SUBCOMM``)
----------------------------------------------------

Host-side partitioning is then derived bottom-up from the device layout.

With ``N = 256`` and ``|NODECOMM| = 16``:

- ``local_i = 16`` on every rank
- ``local_i_offset(r) = 16 * r``

So rank ``r`` owns ``[16*r, 16*r + 15]`` on the host view.

Transfer Metadata Built in ``context_setup``
--------------------------------------------

``context_setup`` gathers per-rank metadata on ``NODECOMM``:

- ``NODECOMM_counts = [16, 16, ..., 16]`` (length 16)
- ``NODECOMM_displs = [0, 16, 32, ..., 240]``
- ``DEVCOMM_NODE_counts`` has ``128`` at the two GPU ranks, ``0`` elsewhere
- ``DEVCOMM_NODE_displs`` has ``0`` and ``128`` at the two GPU ranks

These arrays are consumed directly by ``gpu_allscatterv_htod`` and
``gpu_allgatherv_dtoh``.

Overlap Interpretation
----------------------

Given the partitions above:

- Device chunk ``[0,127]`` overlaps host ranks 0..7
- Device chunk ``[128,255]`` overlaps host ranks 8..15

Each host rank contributes 16 elements to exactly one device chunk in this
balanced case.

Notes on GPU Rank Placement
---------------------------

The two GPU ranks do not have to be ``NODECOMM`` ranks 0 and 1. Their positions
are determined by topology assignment (binding mode, NUMA policy,
``QUOP_RANKS_PER_GPU``). The layout math above is unchanged; only which ranks
carry the non-zero ``DEVCOMM_NODE_*`` entries changes.

When This Example Does Not Apply
--------------------------------

If a propagator overrides partitioning during negotiate (notably circulant via
SHAFFT distribution queries), host/device counts may become non-uniform and this
specific 16/16/128/128 pattern will change.
