Modules
=======

MPI
---
.. note::
    All methods contained in :mod:`~quop_mpi.MPI` must be called collectively (by each MPI process).

.. automodule:: quop_mpi.MPI
    :members:
    :undoc-members:
    :show-inheritance:

graph_array
------------

Methods to generate arrays corresponding to the first row of a circulat graph adjacency matrix. These may be passed to :meth:`~quop_mpi.MPI.qwao.graph` to produce an array of graph eigenvalues distributed over the MPI communicator associated with the :class:`~quop_mpi.MPI.qwao` object.

.. automodule:: quop_mpi.graph_array
    :members:
    :undoc-members:
    :show-inheritance:

qualities
---------

When passed to :meth:`~quop_mpi.MPI.qwao.set_qualities`, these methods create an array of qualities distributed over the MPI communicator associated with the :class:`~quop_mpi.MPI.qwao` object.

.. automodule:: quop_mpi.qualities
    :members:
    :undoc-members:
    :show-inheritance:
