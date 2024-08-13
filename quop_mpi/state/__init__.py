"""Predefined :term:`Initial State Functions <Initial State Function>` .

See :meth:`~quop_mpi.Ansatz.set_initial_state`.
"""
from .standard import equal, basis, serial, array, position_grid

__all__ = ["equal", "basis", "serial", "array", "position_grid"]