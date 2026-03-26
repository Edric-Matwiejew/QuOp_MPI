"""Predefined :term:`Observable Functions <Observables Function>`.

See also: :meth:`quop_mpi.ansatz.set_observables`.
"""

from . import rand
from .standard import array, csv, hdf5, serial

__all__ = ["rand", "serial", "csv", "hdf5", "array"]
