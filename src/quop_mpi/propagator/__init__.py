"""Predefined ``unitary`` classes for simulation of the action of :term:`QVA`
:term:`phase-shift <phase-shift unitary>` and :term:`mixing <mixing unitary>`
unitaries with compatible :term:`Operator Functions <Operator Function>`."""

from . import circulant, composite, diagonal, momentum, sparse

__all__ = ["diagonal", "circulant", "sparse", "composite", "momentum"]
