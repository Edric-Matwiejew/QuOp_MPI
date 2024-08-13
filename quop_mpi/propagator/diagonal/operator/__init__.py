"""Predefined :term:`Operator Functions <Operator Function>` and related utility
for :class:`quop_mpi.propagator.diagonal.unitary`.

An Operator Function for :literal:`'diagonal'` :literal:`unitary` instances returns an
:literal:`ndarray[float64]` of size :literal:`local_i`, or a :literal:`list[ndarray[float64]]` with
:literal:`local_i` sized elements, which correspond to partition(s) of diagonal
:term:`operator(s)<operator>` with global index offset :literal:`local_i_offset`.

If the Operator function returns :literal:`list[ndarray[float64]]`, the :literal:`unitary`
instance must be initialised with :literal:`unitary_n_parameters` equal to the length
of returned list. The resulting :literal:`unitary` is then equivalent to a sequence of
:term:`phase-shift unitaries<phase-shift unitary>` with independently
parameterised :term:`unitary parameters<unitary parameter>`.

"""
from .standard import serial, csv, hdf5, array, setup_cartesian, cartesian, cartesian_scaled, observables
from . import rand

__all__ = ["serial", "csv", "hdf5", "array", "rand", "setup_cartesian", "cartesian", "cartesian_scaled", "observables"]
