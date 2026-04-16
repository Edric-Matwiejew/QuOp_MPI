import numpy as np
import pytest

from quop_mpi.propagator.sparse.unitary import _sort_csr_columns


class TestSortCSRColumns:
    """Unit tests for _sort_csr_columns."""

    def test_already_sorted_is_noop(self):
        row_starts = [np.array([1, 4, 7], dtype=np.int64)]
        col_indexes = [np.array([1, 3, 5, 2, 4, 6], dtype=np.int64)]
        values = [np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])]

        original_cols = col_indexes[0].copy()
        original_vals = values[0].copy()

        _sort_csr_columns(row_starts, col_indexes, values)

        np.testing.assert_array_equal(col_indexes[0], original_cols)
        np.testing.assert_array_equal(values[0], original_vals)

    def test_unsorted_columns_are_sorted(self):
        # Two rows: [5, 1, 3] and [6, 2, 4] — both unsorted
        row_starts = [np.array([1, 4, 7], dtype=np.int64)]
        col_indexes = [np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)]
        values = [np.array([0.5, 0.1, 0.3, 0.6, 0.2, 0.4])]

        _sort_csr_columns(row_starts, col_indexes, values)

        # Row 0: should be [1, 3, 5] with values [0.1, 0.3, 0.5]
        np.testing.assert_array_equal(col_indexes[0][:3], [1, 3, 5])
        np.testing.assert_array_equal(values[0][:3], [0.1, 0.3, 0.5])

        # Row 1: should be [2, 4, 6] with values [0.2, 0.4, 0.6]
        np.testing.assert_array_equal(col_indexes[0][3:], [2, 4, 6])
        np.testing.assert_array_equal(values[0][3:], [0.2, 0.4, 0.6])

    def test_unit_valued_no_values(self):
        row_starts = [np.array([1, 4, 7], dtype=np.int64)]
        col_indexes = [np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)]

        _sort_csr_columns(row_starts, col_indexes, None)

        np.testing.assert_array_equal(col_indexes[0][:3], [1, 3, 5])
        np.testing.assert_array_equal(col_indexes[0][3:], [2, 4, 6])

    def test_single_element_rows(self):
        row_starts = [np.array([1, 2, 3], dtype=np.int64)]
        col_indexes = [np.array([5, 2], dtype=np.int64)]
        values = [np.array([0.5, 0.2])]

        _sort_csr_columns(row_starts, col_indexes, values)

        np.testing.assert_array_equal(col_indexes[0], [5, 2])
        np.testing.assert_array_equal(values[0], [0.5, 0.2])

    def test_empty_rows(self):
        row_starts = [np.array([1, 1, 4], dtype=np.int64)]
        col_indexes = [np.array([3, 1, 2], dtype=np.int64)]
        values = [np.array([0.3, 0.1, 0.2])]

        _sort_csr_columns(row_starts, col_indexes, values)

        np.testing.assert_array_equal(col_indexes[0], [1, 2, 3])
        np.testing.assert_array_equal(values[0], [0.1, 0.2, 0.3])

    def test_multiple_operators(self):
        row_starts = [
            np.array([1, 3], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
        ]
        col_indexes = [
            np.array([4, 2], dtype=np.int64),
            np.array([3, 1], dtype=np.int64),
        ]
        values = [
            np.array([0.4, 0.2]),
            np.array([0.3, 0.1]),
        ]

        _sort_csr_columns(row_starts, col_indexes, values)

        np.testing.assert_array_equal(col_indexes[0], [2, 4])
        np.testing.assert_array_equal(values[0], [0.2, 0.4])
        np.testing.assert_array_equal(col_indexes[1], [1, 3])
        np.testing.assert_array_equal(values[1], [0.1, 0.3])

    def test_nonzero_row_start_offset(self):
        """Simulates a non-root MPI rank where row_starts don't begin at 1."""
        row_starts = [np.array([101, 104, 107], dtype=np.int64)]
        col_indexes = [np.array([5, 1, 3, 6, 2, 4], dtype=np.int64)]
        values = [np.array([0.5, 0.1, 0.3, 0.6, 0.2, 0.4])]

        _sort_csr_columns(row_starts, col_indexes, values)

        np.testing.assert_array_equal(col_indexes[0][:3], [1, 3, 5])
        np.testing.assert_array_equal(values[0][:3], [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(col_indexes[0][3:], [2, 4, 6])
        np.testing.assert_array_equal(values[0][3:], [0.2, 0.4, 0.6])
