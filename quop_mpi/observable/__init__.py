"""Predefined :term:`Observable Functions <Observables Function>`.

See also: :meth:`quop_mpi.Ansatz.set_observables`.
"""

from .standard import serial, csv, hdf5, array
from . import rand

__all__ = ["rand", "serial", "csv", "hdf5", "array"]
