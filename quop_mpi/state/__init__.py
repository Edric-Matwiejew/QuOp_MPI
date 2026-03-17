"""Predefined :term:`Initial State Functions <Initial State Function>` .

See :meth:`~quop_mpi.ansatz.set_initial_state`.
"""

from .standard import array, basis, equal, position_grid, serial

__all__ = ["equal", "basis", "serial", "array", "position_grid"]
