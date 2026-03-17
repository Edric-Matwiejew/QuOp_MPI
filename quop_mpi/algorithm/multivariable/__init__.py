"""Predefined :term:`QVAs <QVA>` for the optimisation of continuous multivariable
functions.

..  note::

    The following compatible :term:`Operator Functions <Operator Function>` may
    be imported from the :mod:`~quop_mpi.algorithm.multivariable` :

    * :func:`~quop_mpi.propagator.diagonal.operator.setup_cartesian`
    * :func:`~quop_mpi.propagator.diagonal.operator.cartesian`
    * :func:`~quop_mpi.propagator.diagonal.operator.cartesian_scaled`
"""

from ..._utils._deprecation import deprecated_alias_getattr
from ...propagator.diagonal.operator import cartesian as cartesian
from ...propagator.diagonal.operator import cartesian_scaled as cartesian_scaled
from ...propagator.diagonal.operator import setup_cartesian as setup_cartesian
from .multivariable import QMOA, QOWE

__all__ = ["QMOA", "QOWE", "qmoa", "qowe", "setup_cartesian", "cartesian", "cartesian_scaled"]

__getattr__ = deprecated_alias_getattr(__name__, globals(), {"qmoa": "QMOA", "qowe": "QOWE"})
