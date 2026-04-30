"""Unit tests for the sparse CSR normalizer used by ``Propagator.gen_operator``.

These tests cover the column-sorting and 1-based -> 0-based conversion paths
formerly exercised through the removed ``_sort_csr_columns`` helper.
"""

import numpy as np

from quop_mpi._lib.propagator import _normalize_sparse_csr_operator_args


def _normalize(row_starts, col_indexes, values=None):
    args = [row_starts, col_indexes] if values is None else [row_starts, col_indexes, values]
    normalized, compliant = _normalize_sparse_csr_operator_args(args)
    return normalized, compliant


class TestNormalizeSparseCSR:
    """Sortedness and 1-based -> 0-based legacy normalization."""

    def test_compliant_input_is_returned_unchanged(self):
        row_starts = np.array([0, 3, 6], dtype=np.int64)
        col_indexes = np.array([0, 2, 4, 1, 3, 5], dtype=np.int64)
        values = np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6], dtype=np.complex128)

        (out_row, out_col, out_vals), compliant = _normalize(row_starts, col_indexes, values)

        assert compliant is True
        # Compliant fast path: zero-copy, same objects returned.
        assert out_row is row_starts
        assert out_col is col_indexes
        assert out_vals is values

    def test_legacy_one_based_unsorted_is_normalized(self):
        # Rows [5, 1, 3] and [6, 2, 4], 1-based, two rows of three nnz.
        row_starts = np.array([1, 4, 7], dtype=np.int64)
        col_indexes = np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)
        values = np.array([0.5, 0.1, 0.3, 0.6, 0.2, 0.4], dtype=np.complex128)

        (out_row, out_col, out_vals), compliant = _normalize(row_starts, col_indexes, values)

        assert compliant is False
        np.testing.assert_array_equal(out_row, [0, 3, 6])
        np.testing.assert_array_equal(out_col[:3], [0, 2, 4])
        np.testing.assert_array_equal(out_vals[:3], [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(out_col[3:], [1, 3, 5])
        np.testing.assert_array_equal(out_vals[3:], [0.2, 0.4, 0.6])
        assert out_row.dtype == np.int64
        assert out_col.dtype == np.int64
        assert out_vals.dtype == np.complex128

    def test_unit_valued_legacy_is_normalized(self):
        row_starts = np.array([1, 4, 7], dtype=np.int64)
        col_indexes = np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)

        (out_row, out_col), compliant = _normalize(row_starts, col_indexes, None)

        assert compliant is False
        np.testing.assert_array_equal(out_row, [0, 3, 6])
        np.testing.assert_array_equal(out_col[:3], [0, 2, 4])
        np.testing.assert_array_equal(out_col[3:], [1, 3, 5])

    def test_single_element_rows_preserve_order(self):
        # 1-based input, one nnz per row; sort must be a no-op.
        row_starts = np.array([1, 2, 3], dtype=np.int64)
        col_indexes = np.array([5, 2], dtype=np.int64)
        values = np.array([0.5, 0.2], dtype=np.complex128)

        (out_row, out_col, out_vals), _ = _normalize(row_starts, col_indexes, values)

        np.testing.assert_array_equal(out_row, [0, 1, 2])
        np.testing.assert_array_equal(out_col, [4, 1])
        np.testing.assert_array_equal(out_vals, [0.5, 0.2])

    def test_empty_rows_are_preserved(self):
        row_starts = np.array([1, 1, 4], dtype=np.int64)
        col_indexes = np.array([3, 1, 2], dtype=np.int64)
        values = np.array([0.3, 0.1, 0.2], dtype=np.complex128)

        (out_row, out_col, out_vals), _ = _normalize(row_starts, col_indexes, values)

        np.testing.assert_array_equal(out_row, [0, 0, 3])
        np.testing.assert_array_equal(out_col, [0, 1, 2])
        np.testing.assert_array_equal(out_vals, [0.1, 0.2, 0.3])

    def test_partitioned_legacy_input_with_global_offset(self):
        """Non-root MPI rank: 1-based row_starts carry a cumulative global nnz offset."""
        row_starts = np.array([101, 104, 107], dtype=np.int64)
        col_indexes = np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)
        values = np.array([0.5, 0.1, 0.3, 0.6, 0.2, 0.4], dtype=np.complex128)

        (out_row, out_col, out_vals), compliant = _normalize(row_starts, col_indexes, values)

        assert compliant is False
        np.testing.assert_array_equal(out_row, [0, 3, 6])
        np.testing.assert_array_equal(out_col[:3], [0, 2, 4])
        np.testing.assert_array_equal(out_vals[:3], [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(out_col[3:], [1, 3, 5])
        np.testing.assert_array_equal(out_vals[3:], [0.2, 0.4, 0.6])

    def test_caller_buffers_are_not_mutated(self):
        row_starts = np.array([1, 4, 7], dtype=np.int64)
        col_indexes = np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)
        values = np.array([0.5, 0.1, 0.3, 0.6, 0.2, 0.4], dtype=np.complex128)
        original_row = row_starts.copy()
        original_col = col_indexes.copy()
        original_vals = values.copy()

        _normalize(row_starts, col_indexes, values)

        np.testing.assert_array_equal(row_starts, original_row)
        np.testing.assert_array_equal(col_indexes, original_col)
        np.testing.assert_array_equal(values, original_vals)
