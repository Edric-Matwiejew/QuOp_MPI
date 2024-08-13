"""Predefined :term:`QVAs <QVA>` for combinatorial optimisation problems.

..  note::
    The following compatible :term:`Operator Functions <Operator Function>` may be imported from the :mod:`~quop_mpi.algorithm.combinatorial`:

        * :meth:`~quop_mpi.propagator.diagonal.operator.serial`
        * :meth:`~quop_mpi.propagator.diagonal.operator.csv`
        * :meth:`~quop_mpi.propagator.diagonal.operator.hdf5`
        * :meth:`~quop_mpi.propagator.diagonal.operator.array`
        * :meth:`~quop_mpi.propagator.diagonal.operator.uniform`
"""
from .qwoa import qwoa
from .qaoa import qaoa
from ...observable import serial as serial
from ...observable import csv as csv
from ...observable import hdf5 as hdf5
from ...observable import array as array
from ...observable.rand import uniform as uniform

__all__ = ["qwoa", "qaoa", "serial", "csv", "hdf5", "array", "rand"]
