"""Unit tests for UnitaryBase state handling."""

import numpy as np

from quop_mpi.unitary import UnitaryBase


class _FakeComm:
    def Get_rank(self):  # noqa: N802
        return 0


class _FakeLayout:
    def __init__(self):
        self.SUBCOMM = _FakeComm()
        self.local_i = 4
        self.alloc_local = 6
        self.partition_table = np.array([1, 5], dtype=np.int64)


def _make_unitary():
    return UnitaryBase(
        operator_function=lambda: None,
        parameter_function=lambda: np.zeros(1, dtype=np.float64),
    )


class TestUnitaryBaseStateHandling:
    def test_plan_does_not_allocate_legacy_state_buffers(self):
        unitary = _make_unitary()

        unitary._UnitaryBase__plan(4, _FakeLayout())

        assert unitary.initial_state is None
        assert unitary.final_state is None

    def test_legacy_state_buffers_stay_unset_for_native_context_backends(self):
        unitary = _make_unitary()
        unitary._UnitaryBase__plan(4, _FakeLayout())
        unitary.context = object()

        assert unitary.initial_state is None
        assert unitary.final_state is None

    def test_custom_subclasses_can_still_install_explicit_legacy_buffers(self):
        unitary = _make_unitary()

        initial_buffer = np.full(6, 2.0 + 0.0j, dtype=np.complex128)
        final_buffer = np.full(6, 3.0 + 0.0j, dtype=np.complex128)

        unitary.initial_state = initial_buffer
        unitary.final_state = final_buffer

        assert unitary.initial_state is initial_buffer
        assert unitary.final_state is final_buffer
