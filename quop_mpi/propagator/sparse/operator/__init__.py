"""Predefined :term:`Operator Functions <Operator Function>` for
:class:`quop_mpi.propagator.sparse.unitary`.

Operator Functions for :literal:`unitary`  instances of type :literal:`'sparse'`  return CSR
partitions of one or more matrices. More than one CSR partition defines a
sequence of :term:`mixing unitaries <mixing unitary>` with independent
:term:`unitary parameters <unitary parameter>`.

**Partitioned CSR Matrix Format**

.. glossary::

    lb : int
        lower index of the :term:`system state` and :term:`observables` partition, :class:`quop_mpi.Unitary` attribute

    ub : int
        upper index of the system state and observables partition, :class:`quop_mpi.Unitary` attribute

    W_col_index : ndarray[int[]]
        a 1-D integer array containing non-zero column indexes for rows :literal:`lb` 
        to :literal:`ub` , grouped by ascending row index

    W_values : ndarray[float] 
        a 1-D real array containing non-zero values for rows :literal:`lb`  to :literal:`ub` ,
        grouped by ascending row index in the same order as :literal:`W_col_index` 

    W_row_start : ndarray[int] 
        an 1-D integer array of length :literal:`ub - lb + 1` , a cumulative sum of the
        number of non-zero elements in each row such that
        :literal:`W_row_start[row_index + 1] - W_row_start[row_index]`  is equal to the
        number of non-zero elements in the row with index :literal:`row_index`  and
        :literal:`W_rows_start[row] - local_i_offset`  gives the local starting index
        for the non-zero column indexes and values in :literal:`W_col_index`  and
        :literal:`W_values`  for the row with index :literal:`row_index` 

These are returned by the Operator Function as 
:literal:`list[list[W_row_start], list[W_col_indexes], list[W_values]]`.
"""
from .standard import serial, hypercube, qmoa_mixer

__all__ = ["serial", "hypercube", "qmoa_mixer"]
