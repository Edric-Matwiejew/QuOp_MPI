import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from quop_mpi._lib.context import Context


class _FakeSubcomm:
    def __init__(self):
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1


class _FakeCommInfo:
    def __init__(self):
        self.system_size = 8
        self.local_i = 4
        self.local_i_offset = 2
        self.alloc_local = 6
        self.handle = 123
        self.subcomm = _FakeSubcomm()


class _FakeContext:
    """Stand-in for the native ``Context`` PyObject returned by ``setup``."""

    def __init__(self, handle, alloc_local, local_i):
        self.handle = handle
        self.alloc_local = alloc_local
        self.local_i = local_i


class _FakeContextWrapper:
    def __init__(
        self,
        *,
        setup_status=0,
        get_state_status=0,
        set_state_status=0,
        get_observables_status=0,
        set_observables_status=0,
        get_expectation_value_status=0,
        get_state_norm_status=0,
    ):
        self.setup_status = setup_status
        self.get_state_status = get_state_status
        self.set_state_status = set_state_status
        self.get_observables_status = get_observables_status
        self.set_observables_status = set_observables_status
        self.get_expectation_value_status = get_expectation_value_status
        self.get_state_norm_status = get_state_norm_status
        self.setup_calls = []
        self.get_state_calls = []
        self.set_state_calls = []
        self.get_observables_calls = []
        self.set_observables_calls = []
        self.get_expectation_value_calls = []
        self.get_state_norm_calls = []
        self.destroyed_ctx = None

    def setup(self, handle, alloc_local, local_i):
        self.setup_calls.append((handle, alloc_local, local_i))
        return _FakeContext(456, alloc_local, local_i), self.setup_status

    def destroy(self, ctx):
        self.destroyed_ctx = ctx

    def get_state(self, ctx):
        self.get_state_calls.append(ctx)
        return (
            (np.arange(ctx.alloc_local, dtype=np.float64) + 1j).astype(np.complex128),
            self.get_state_status,
        )

    def set_state(self, ctx, state):
        self.set_state_calls.append((ctx, state.copy()))
        return self.set_state_status

    def get_observables(self, ctx):
        self.get_observables_calls.append(ctx)
        return np.arange(ctx.local_i, dtype=np.float64), self.get_observables_status

    def set_observables(self, ctx, obs):
        self.set_observables_calls.append((ctx, obs.copy()))
        return self.set_observables_status

    def get_expectation_value(self, ctx):
        self.get_expectation_value_calls.append(ctx)
        return 1.5, self.get_expectation_value_status

    def get_state_norm(self, ctx):
        self.get_state_norm_calls.append(ctx)
        return 0.75, self.get_state_norm_status

    def get_local_probabilities(self, ctx):
        return np.zeros(ctx.local_i, dtype=np.float64)


def _make_backend(wrapper):
    return SimpleNamespace(context_wrapper=wrapper)


class TestContextStatusTranslation:
    def test_setup_success_initializes_context(self):
        wrapper = _FakeContextWrapper()
        comm_info = _FakeCommInfo()

        ctx = Context(_make_backend(wrapper), comm_info)

        assert ctx.ptr == 456
        assert wrapper.setup_calls == [(comm_info.handle, comm_info.alloc_local, comm_info.local_i)]
        assert comm_info.subcomm.barrier_calls == 1

    def test_setup_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(setup_status=12)
        comm_info = _FakeCommInfo()

        with pytest.raises(RuntimeError, match="status 12"):
            Context(_make_backend(wrapper), comm_info)

        assert wrapper.setup_calls == [(comm_info.handle, comm_info.alloc_local, comm_info.local_i)]
        assert comm_info.subcomm.barrier_calls == 0

    def test_state_and_observables_use_layout_contract(self):
        wrapper = _FakeContextWrapper()
        comm_info = _FakeCommInfo()

        ctx = Context(_make_backend(wrapper), comm_info)

        state = ctx.state
        observables = ctx.observables

        np.testing.assert_array_equal(
            state,
            (np.arange(comm_info.alloc_local, dtype=np.float64) + 1j).astype(np.complex128),
        )
        np.testing.assert_array_equal(
            observables,
            np.arange(comm_info.local_i, dtype=np.float64),
        )
        assert wrapper.get_state_calls == [ctx._ctx]
        assert wrapper.get_observables_calls == [ctx._ctx]

    def test_scalar_getters_return_values_on_zero_status(self):
        wrapper = _FakeContextWrapper()
        comm_info = _FakeCommInfo()

        ctx = Context(_make_backend(wrapper), comm_info)

        assert ctx.get_expectation_value() == 1.5
        assert ctx.get_state_norm() == 0.75
        assert wrapper.get_expectation_value_calls == [ctx._ctx]
        assert wrapper.get_state_norm_calls == [ctx._ctx]

    def test_state_getter_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(get_state_status=2)
        comm_info = _FakeCommInfo()
        ctx = Context(_make_backend(wrapper), comm_info)

        with pytest.raises(RuntimeError, match="Cannot fetch context state"):
            _ = ctx.state

        assert wrapper.get_state_calls == [ctx._ctx]

    def test_state_setter_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(set_state_status=1)
        comm_info = _FakeCommInfo()
        ctx = Context(_make_backend(wrapper), comm_info)

        with pytest.raises(RuntimeError, match="Cannot update context state"):
            ctx.state = np.ones(comm_info.alloc_local, dtype=np.complex128)

        assert len(wrapper.set_state_calls) == 1
        assert wrapper.set_state_calls[0][0] is ctx._ctx

    def test_observables_getter_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(get_observables_status=2)
        comm_info = _FakeCommInfo()
        ctx = Context(_make_backend(wrapper), comm_info)

        with pytest.raises(RuntimeError, match="Cannot fetch context observables"):
            _ = ctx.observables

        assert wrapper.get_observables_calls == [ctx._ctx]

    def test_observables_setter_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(set_observables_status=1)
        comm_info = _FakeCommInfo()
        ctx = Context(_make_backend(wrapper), comm_info)

        with pytest.raises(RuntimeError, match="Cannot update context observables"):
            ctx.observables = np.ones(comm_info.local_i, dtype=np.float64)

        assert len(wrapper.set_observables_calls) == 1
        assert wrapper.set_observables_calls[0][0] is ctx._ctx

    def test_expectation_value_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(get_expectation_value_status=1)
        comm_info = _FakeCommInfo()
        ctx = Context(_make_backend(wrapper), comm_info)

        with pytest.raises(RuntimeError, match="Cannot fetch context expectation value"):
            ctx.get_expectation_value()

        assert wrapper.get_expectation_value_calls == [ctx._ctx]

    def test_state_norm_raises_on_nonzero_status(self):
        wrapper = _FakeContextWrapper(get_state_norm_status=1)
        comm_info = _FakeCommInfo()
        ctx = Context(_make_backend(wrapper), comm_info)

        with pytest.raises(RuntimeError, match="Cannot fetch context state norm"):
            ctx.get_state_norm()

        assert wrapper.get_state_norm_calls == [ctx._ctx]

    def test_destroy_releases_borrowed_layout_and_subcomm_refs(self):
        wrapper = _FakeContextWrapper()
        comm_info = _FakeCommInfo()
        layout_ref = weakref.ref(comm_info)
        subcomm_ref = weakref.ref(comm_info.subcomm)

        ctx = Context(_make_backend(wrapper), comm_info)
        live_ctx = ctx._ctx
        ctx.destroy()

        del comm_info
        gc.collect()

        # destroy() forwarded the live Context PyObject to the wrapper.
        assert wrapper.destroyed_ctx is live_ctx
        assert ctx.ptr == 0
        assert ctx.initialised is False
        assert ctx._comm_info is None
        assert ctx.SUBCOMM is None
        assert ctx._ctx is None
        assert layout_ref() is None
        assert subcomm_ref() is None
