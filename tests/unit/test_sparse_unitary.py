"""Unit tests for the sparse propagator Python wrapper."""

import numpy as np
import pytest

from quop_mpi._lib.propagator import Propagator
from quop_mpi.propagator.sparse.unitary import Unitary


class _FakeContext:
    """Opaque context object used to catch accidental Python attribute writes."""


class _FakePropagator:
    def __init__(self):
        self.context = _FakeContext()
        self.calls = []

    def propagate(self, t):
        self.calls.append(t)


class _FakeWrapper:
    def setup(self):
        return 1, 0

    def destroy(self, ptr):
        return None

    def gen_operator(self, ptr, ptrs, array_sizes):
        self.last_ptr = ptr
        self.last_ptrs = ptrs
        self.last_sizes = array_sizes
        return 0


class TestSparseUnitaryWrapper:
    def test_propagate_relies_on_native_context_swaps(self):
        unitary = Unitary.__new__(Unitary)
        unitary.propagators = [_FakePropagator(), _FakePropagator()]

        Unitary.propagate(unitary, np.array([-1.5, 2.5], dtype=np.float64))

        assert unitary.propagators[0].calls == [1.5]
        assert unitary.propagators[1].calls == [2.5]

        for propagator in unitary.propagators:
            assert not hasattr(propagator.context, "initial_state")
            assert not hasattr(propagator.context, "final_state")


class TestSparsePropagatorCompliance:
    def test_sparse_propagator_keeps_exact_compliant_arrays_alive(self):
        wrapper = _FakeWrapper()
        propagator = Propagator(wrapper)

        row_starts = np.array([0, 2, 4], dtype=np.int64)
        col_indexes = np.array([0, 1, 1, 2], dtype=np.int64)
        values = np.array([1, 2, 3, 4], dtype=np.complex128)

        propagator.gen_operator(
            [row_starts, col_indexes, values], prepare_sparse_csr=True
        )

        # Compliant fast path: the wrapper retains the *exact* caller arrays
        # (no copy) so the native backend's borrowed pointers remain valid.
        held = propagator._native_pinned_buffers[-1]
        assert held[0] is row_starts
        assert held[1] is col_indexes
        assert held[2] is values

    def test_sparse_propagator_normalizes_legacy_csr_and_warns(self):
        wrapper = _FakeWrapper()
        propagator = Propagator(wrapper)

        row_starts = np.array([5, 7, 9], dtype=np.int32)
        col_indexes = np.array([3, 2, 4, 1], dtype=np.int32)
        values = np.array([10, 20, 30, 40], dtype=np.float64)
        original_row_starts = row_starts.copy()
        original_col_indexes = col_indexes.copy()
        original_values = values.copy()

        with pytest.deprecated_call():
            propagator.gen_operator(
                [row_starts, col_indexes, values], prepare_sparse_csr=True
            )

        held = propagator._native_pinned_buffers[-1]
        normalized_row_starts, normalized_col_indexes, normalized_values = held
        np.testing.assert_array_equal(
            normalized_row_starts, np.array([0, 2, 4], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            normalized_col_indexes, np.array([1, 2, 0, 3], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            normalized_values, np.array([20, 10, 40, 30], dtype=np.complex128)
        )
        assert normalized_row_starts.dtype == np.int64
        assert normalized_col_indexes.dtype == np.int64
        assert normalized_values.dtype == np.complex128

        # Caller-supplied arrays must not be mutated by normalization.
        np.testing.assert_array_equal(row_starts, original_row_starts)
        np.testing.assert_array_equal(col_indexes, original_col_indexes)
        np.testing.assert_array_equal(values, original_values)

    def test_repeated_gen_operator_retains_all_buffers(self):
        wrapper = _FakeWrapper()
        propagator = Propagator(wrapper)

        first_row = np.array([0, 1], dtype=np.int64)
        first_col = np.array([0], dtype=np.int64)
        first_vals = np.array([1], dtype=np.complex128)
        second_row = np.array([0, 1], dtype=np.int64)
        second_col = np.array([0], dtype=np.int64)
        second_vals = np.array([2], dtype=np.complex128)

        propagator.gen_operator(
            [first_row, first_col, first_vals], prepare_sparse_csr=True
        )
        propagator.gen_operator(
            [second_row, second_col, second_vals], prepare_sparse_csr=True
        )

        # Both buffer sets must remain referenced; the native backend may
        # still hold raw pointers from the earlier gen_operator call.
        assert len(propagator._native_pinned_buffers) == 2
        assert propagator._native_pinned_buffers[0][0] is first_row
        assert propagator._native_pinned_buffers[1][0] is second_row

    def test_destroy_releases_pinned_buffers(self):
        wrapper = _FakeWrapper()
        propagator = Propagator(wrapper)

        row_starts = np.array([0, 1], dtype=np.int64)
        col_indexes = np.array([0], dtype=np.int64)
        values = np.array([1], dtype=np.complex128)

        propagator.gen_operator(
            [row_starts, col_indexes, values], prepare_sparse_csr=True
        )
        propagator.destroy()

        assert propagator._native_pinned_buffers is None
