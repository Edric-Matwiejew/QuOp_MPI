"""Unit tests for the sparse propagator Python wrapper."""

import numpy as np

from quop_mpi.propagator.sparse.unitary import Unitary


class _FakeContext:
    """Opaque context object used to catch accidental Python attribute writes."""


class _FakePropagator:
    def __init__(self):
        self.context = _FakeContext()
        self.calls = []

    def propagate(self, t):
        self.calls.append(t)


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
