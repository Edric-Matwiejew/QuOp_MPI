"""Predefined :term:`Operator Functions <Operator Function>` for
:class:`quop_mpi.propagator.circulant.unitary`.

An Operator Function for :literal:`'circulant'` :literal:`unitary` instances return a
:literal:`local_i` sized partition of the :term:`operator` eigenvalues with global index offset
:literal:`local_i_offset`.
"""
from .eigenvalues import complete, graph

__all__ = ["complete", "graph"]
