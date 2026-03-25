"""Predefined :term:`QVAs <QVA>` for combinatorial optimisation problems.

..  note::
    The following compatible :term:`Operator Functions <Operator Function>`
    may be imported from the
    :mod:`~quop_mpi.algorithm.combinatorial`:

        * :func:`~quop_mpi.observable.serial`
        * :func:`~quop_mpi.observable.csv`
        * :func:`~quop_mpi.observable.hdf5`
        * :func:`~quop_mpi.observable.array`
        * :func:`~quop_mpi.observable.rand.uniform`
"""

from ..._utils._deprecation import deprecated_alias_getattr
from ...observable import array as array
from ...observable import csv as csv
from ...observable import hdf5 as hdf5
from ...observable import serial as serial
from ...observable.rand import uniform as uniform
from .qaoa import QAOA
from .qwoa import QWOA

__all__ = ["QWOA", "QAOA", "qwoa", "qaoa", "serial", "csv", "hdf5", "array", "rand"]

__getattr__ = deprecated_alias_getattr(__name__, globals(), {"qaoa": "QAOA", "qwoa": "QWOA"})
