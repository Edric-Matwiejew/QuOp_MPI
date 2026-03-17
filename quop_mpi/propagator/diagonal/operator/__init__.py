"""Predefined :term:`Operator Functions <Operator Function>` and related utility
for :class:`quop_mpi.propagator.diagonal.unitary`.

**Return Format**

An Operator Function for :literal:`'diagonal'` :literal:`unitary` instances returns:

    diagonal : ndarray[float64] or list[ndarray[float64]]
        A 1-D real array of size :literal:`local_i` containing the diagonal
        elements of the operator for global indices :literal:`local_i_offset`
        to :literal:`local_i_offset + local_i - 1`.

        Alternatively, a list of such arrays for multiple diagonal operators
        (requires :literal:`unitary_n_params` to match the list length).

If the Operator function returns :literal:`list[ndarray[float64]]`, the :literal:`unitary`
instance must be initialised with :literal:`unitary_n_parameters` equal to the length
of returned list. The resulting :literal:`unitary` is then equivalent to a sequence of
:term:`phase-shift unitaries<phase-shift unitary>` with independently
parameterised :term:`unitary parameters<unitary parameter>`.

**Propagation Method**

The diagonal propagator computes :math:`e^{-itH}|\\psi\\rangle` via direct
element-wise multiplication:

.. math::

    |\\psi'\\rangle_k = e^{-it H_{kk}} |\\psi\\rangle_k

where :math:`H_{kk}` are the diagonal elements. This is the most efficient
propagation method as it requires only :math:`O(N)` operations with no
communication between MPI ranks.
"""

from . import rand
from .standard import (
    array,
    cartesian,
    cartesian_scaled,
    csv,
    hdf5,
    observables,
    serial,
    setup_cartesian,
)

__all__ = [
    "serial",
    "csv",
    "hdf5",
    "array",
    "rand",
    "setup_cartesian",
    "cartesian",
    "cartesian_scaled",
    "observables",
]
