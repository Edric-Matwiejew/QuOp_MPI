"""Momentum propagator subpackage.

The momentum propagator does not accept user-defined :term:`Operator Functions
<Operator Function>`.  The momentum-space eigenvalues are computed internally
by the Fortran backend from the grid parameters (:literal:`Ns`, :literal:`minsq`,
:literal:`minsk`, :literal:`deltasq`, :literal:`deltask`) passed when
constructing :class:`quop_mpi.propagator.momentum.Unitary`.
"""

__all__: list[str] = []
