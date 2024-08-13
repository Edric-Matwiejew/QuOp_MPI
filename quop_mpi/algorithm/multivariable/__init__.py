"""Predefined :term:`QVAs <QVA>` for the optimisation of continuous multivariable
functions.

..  note::

    The following compatible :term:`Operator Functions <Operator Function>` may
    be imported from the :mod:`~quop_mpi.algorithm.multivariable` :

    * :meth:`~quop_mpi.diagonal.operator.setup_cartesian`
    * :meth:`~quop_mpi.diagonal.operator.cartesian`
    * :meth:`~quop_mpi.diagonal.operator.cartesian_scaled`
"""
from .multivariable import qmoa, qowe
from ...propagator.diagonal.operator import setup_cartesian as setup_cartesian
from ...propagator.diagonal.operator import cartesian as cartesian
from ...propagator.diagonal.operator import cartesian_scaled as cartesian_scaled

__all__ = ["qmoa", "qowe", "setup_cartesian", "cartesian", "cartesian_scaled"]