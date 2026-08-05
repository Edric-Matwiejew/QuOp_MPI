"""Predefined :term:`Operator Functions <Operator Function>` for
:class:`quop_mpi.propagator.sparse.unitary`.

Operator Functions for :literal:`unitary`  instances of type :literal:`'sparse'`  return CSR
partitions of one or more matrices. More than one CSR partition defines a
sequence of :term:`mixing unitaries <mixing unitary>` with independent
:term:`unitary parameters <unitary parameter>`.

**Partitioned CSR Matrix Format**

.. glossary::

    lb : int
        lower index of the :term:`system state` and :term:`observables`
        partition, :class:`quop_mpi.unitary` attribute

    ub : int
        upper index of the system state and observables partition,
        :class:`quop_mpi.unitary` attribute

    W_col_index : ndarray[int64]
        a 1-D, C-contiguous integer array containing non-zero column indexes
        for rows :literal:`lb` to :literal:`ub`, grouped by ascending row
        index. Column indexes are 0-based and must be sorted in ascending
        order within each row.

    W_values : ndarray[complex128] or None
        a 1-D, C-contiguous complex array containing non-zero values for rows
        :literal:`lb` to :literal:`ub`, in the same order as :literal:`W_col_index`.

        For **unit-valued matrices** (where all non-zero entries are 1.0), this
        may be :literal:`None`. When :literal:`W_values` is :literal:`None`, the propagator
        skips value storage and uses an optimized code path, reducing memory
        usage and improving performance. This is automatically detected for
        adjacency matrices such as those used by the hypercube mixer.

    W_row_start : ndarray[int64]
        a 1-D, C-contiguous integer array of length :literal:`ub - lb + 2`
        giving 0-based cumulative offsets into :literal:`W_col_index` and
        :literal:`W_values` for each local row. The contract is:

        - :literal:`W_row_start[0] == 0`,
        - :literal:`W_row_start[k + 1] - W_row_start[k]` equals the number
          of non-zeros in the :literal:`k`-th local row, and
        - :literal:`W_row_start[-1]` equals the total local nnz.

**Legacy compatibility**

Older sparse operator functions may return 1-based CSR (with
:literal:`W_row_start[0]` carrying a global nnz offset and column indexes
offset by 1) or columns that are not sorted within each row. The sparse
propagator detects these inputs, converts them to the canonical form above,
and emits a :class:`DeprecationWarning`. Future releases will require the
canonical contract.

These are returned by the Operator Function as
:literal:`list[list[W_row_start], list[W_col_indexes], list[W_values]]`.

**Propagation Method**

The sparse propagator uses **Chebyshev polynomial expansion** to compute the
matrix exponential :math:`e^{-itH}`. This method:

- Automatically estimates the spectral radius of the operator
- Expands the matrix exponential as a sum of Chebyshev polynomials
- Is numerically stable and efficient for sparse Hermitian matrices

This replaces the previous scaling-and-squaring approach.
"""

from .standard import hypercube, qmoa_mixer, serial

__all__ = ["serial", "hypercube", "qmoa_mixer"]
