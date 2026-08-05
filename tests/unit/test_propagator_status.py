import gc
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

import quop_mpi._lib.propagator as propagator_module
from quop_mpi._lib.propagator import Propagator


class _FakeBackendPropagator:
    def __init__(self, *, setup_status=0, max_status=0, plan_status=0, gen_status=0):
        self.setup_status = setup_status
        self.max_status = max_status
        self.plan_status = plan_status
        self.gen_status = gen_status
        self.setup_ptr = 123
        self.setup_calls = 0
        self.max_calls = []
        self.plan_calls = []
        self.gen_calls = []

    def setup(self):
        self.setup_calls += 1
        return self.setup_ptr, self.setup_status

    def destroy(self, ptr):
        self.destroyed_ptr = ptr

    def store_constraints(self, ptr, ptrs, sizes):
        self.constraint_call = (ptr, tuple(sizes.tolist()))

    def get_negotiate_callback(self):
        return 456

    def max_comm_size(self, ptr, ci_ptr):
        self.max_calls.append((ptr, ci_ptr))
        return self.max_status

    def plan(self, ptr, context_ptr):
        self.plan_calls.append((ptr, context_ptr))
        return self.plan_status

    def gen_operator(self, ptr, ptrs, array_sizes):
        self.gen_calls.append((ptr, tuple(array_sizes.tolist())))
        return self.gen_status

    def propagate(self, ptr, ts_arr):
        self.propagate_call = (ptr, ts_arr.copy())


class _LifetimeTrackingBackend(_FakeBackendPropagator):
    def __init__(self):
        super().__init__()
        self.destroyed_tokens = None
        self.tokens_seen_during_call = None

    def store_constraints(self, ptr, ptrs, sizes):
        self.tokens_seen_during_call = list(self.destroyed_tokens)
        self.constraint_call = (ptr, tuple(sizes.tolist()))

    def gen_operator(self, ptr, ptrs, array_sizes):
        self.tokens_seen_during_call = list(self.destroyed_tokens)
        self.gen_calls.append((ptr, tuple(array_sizes.tolist())))
        return self.gen_status


class _FakeContext:
    def __init__(self, ptr):
        self.ptr = ptr


class _DummyProp:
    def __init__(self):
        self.ptr = 11
        self.negotiate_callback = 22
        self.seen_constraints = None

    def store_constraints(self, constraints):
        self.seen_constraints = constraints


class _DummyUnitary:
    def __init__(self, prop):
        self.propagators = [prop]
        self.comm_size_constraints = [np.array([4], dtype=np.int32)]


class _DummyLayout:
    def __init__(self):
        self.split_ptr = 101
        self.topo_ptr = 202
        self.handle = None
        self.destroy_calls = 0
        self.marked_excluded = False
        self.set_layout_ptr_calls = []

    @staticmethod
    def get_n_subcomms():
        return 1

    def set_layout_ptr(self, ptr):
        self.handle = ptr
        self.set_layout_ptr_calls.append(ptr)

    def mark_excluded(self):
        self.marked_excluded = True

    def destroy(self):
        self.destroy_calls += 1
        self.handle = None


class TestPropagatorStatusTranslation:
    def test_setup_raises_on_nonzero_status(self):
        backend = _FakeBackendPropagator(setup_status=100)

        with pytest.raises(RuntimeError, match="Cannot initialize propagator"):
            Propagator(backend)

        assert backend.setup_calls == 1

    def test_plan_success_passes_context_pointer(self):
        backend = _FakeBackendPropagator()
        wrapped = Propagator(backend)

        wrapped.plan(_FakeContext(999))

        assert backend.setup_calls == 1
        assert backend.plan_calls == [(backend.setup_ptr, 999)]

    def test_max_comm_size_raises_on_nonzero_status(self):
        backend = _FakeBackendPropagator(max_status=1)
        wrapped = Propagator(backend)

        with pytest.raises(RuntimeError, match="negotiate propagator layout"):
            wrapped.max_comm_size(77)

    def test_plan_raises_on_nonzero_status(self):
        backend = _FakeBackendPropagator(plan_status=2)
        wrapped = Propagator(backend)

        with pytest.raises(RuntimeError, match="kernel dimension limit"):
            wrapped.plan(_FakeContext(55))

    def test_gen_operator_raises_on_nonzero_status(self):
        backend = _FakeBackendPropagator(gen_status=1)
        wrapped = Propagator(backend)

        with pytest.raises(RuntimeError, match="layout is incompatible"):
            wrapped.gen_operator([np.arange(4, dtype=np.float64)])

    def test_store_constraints_keeps_converted_arrays_alive_during_call(self, monkeypatch):
        destroyed_tokens = []
        backend = _LifetimeTrackingBackend()
        backend.destroyed_tokens = destroyed_tokens
        wrapped = Propagator(backend)
        real_ascontiguousarray = np.ascontiguousarray

        class _TrackedArray(np.ndarray):
            def __new__(cls, array, token):
                obj = np.asarray(array).view(cls)
                obj._token = token
                return obj

            def __array_finalize__(self, obj):
                self._token = getattr(obj, "_token", None)

            def __del__(self):
                if self._token is not None:
                    destroyed_tokens.append(self._token)

        next_token = 0

        def _tracked_ascontiguousarray(array, dtype=None):
            nonlocal next_token
            token = next_token
            next_token += 1
            base = real_ascontiguousarray(array, dtype=dtype)
            return _TrackedArray(base, token)

        monkeypatch.setattr(
            propagator_module.np,
            "ascontiguousarray",
            _tracked_ascontiguousarray,
        )

        wrapped.store_constraints([np.arange(8, dtype=np.int32)[::2]])

        assert backend.tokens_seen_during_call == []
        gc.collect()
        assert destroyed_tokens == [0]

    def test_gen_operator_keeps_converted_arrays_alive_during_call(self, monkeypatch):
        destroyed_tokens = []
        backend = _LifetimeTrackingBackend()
        backend.destroyed_tokens = destroyed_tokens
        wrapped = Propagator(backend)
        real_ascontiguousarray = np.ascontiguousarray

        class _TrackedArray(np.ndarray):
            def __new__(cls, array, token):
                obj = np.asarray(array).view(cls)
                obj._token = token
                return obj

            def __array_finalize__(self, obj):
                self._token = getattr(obj, "_token", None)

            def __del__(self):
                if self._token is not None:
                    destroyed_tokens.append(self._token)

        next_token = 0

        def _tracked_ascontiguousarray(array, dtype=None):
            nonlocal next_token
            token = next_token
            next_token += 1
            base = real_ascontiguousarray(array, dtype=dtype)
            return _TrackedArray(base, token)

        monkeypatch.setattr(
            propagator_module.np,
            "ascontiguousarray",
            _tracked_ascontiguousarray,
        )

        wrapped.gen_operator([np.arange(8, dtype=np.float64)[::2]])

        assert backend.tokens_seen_during_call == []
        gc.collect()
        assert destroyed_tokens == [0]


class TestAnsatzNegotiateStatus:
    def test_check_comm_size_raises_on_callback_status(self, monkeypatch):
        ansatz_module = import_module("quop_mpi.ansatz")
        prop = _DummyProp()
        unitary = _DummyUnitary(prop)
        layout = _DummyLayout()

        alg = ansatz_module.Ansatz.__new__(ansatz_module.Ansatz)
        alg.system_size = 8
        alg.unitaries = [unitary]
        alg._layout = layout
        alg.MPI_COMM_WORLD = MPI.COMM_WORLD

        fake_ciw = SimpleNamespace(
            wrapper_negotiate=lambda *args: (333, 1001),
        )

        with monkeypatch.context() as mp:
            mp.setattr(ansatz_module, "_ciw", fake_ciw)
            with pytest.raises(RuntimeError, match="callback failed with status 1"):
                ansatz_module.Ansatz._Ansatz__check_comm_size.__wrapped__(alg)

        assert prop.seen_constraints is unitary.comm_size_constraints
        assert layout.set_layout_ptr_calls == [333]
        assert layout.destroy_calls == 1
        assert alg._layout is None

    def test_check_comm_size_cleans_layout_on_generic_error_status(self, monkeypatch):
        ansatz_module = import_module("quop_mpi.ansatz")
        prop = _DummyProp()
        unitary = _DummyUnitary(prop)
        layout = _DummyLayout()

        alg = ansatz_module.Ansatz.__new__(ansatz_module.Ansatz)
        alg.system_size = 8
        alg.unitaries = [unitary]
        alg._layout = layout
        alg.MPI_COMM_WORLD = MPI.COMM_WORLD

        fake_ciw = SimpleNamespace(
            wrapper_negotiate=lambda *args: (0, 1),
        )

        with monkeypatch.context() as mp:
            mp.setattr(ansatz_module, "_ciw", fake_ciw)
            with pytest.raises(RuntimeError, match="error status 1"):
                ansatz_module.Ansatz._Ansatz__check_comm_size.__wrapped__(alg)

        assert layout.set_layout_ptr_calls == []
        assert layout.destroy_calls == 1
        assert alg._layout is None

    def test_check_comm_size_raises_on_partition_table_status(self, monkeypatch):
        ansatz_module = import_module("quop_mpi.ansatz")
        prop = _DummyProp()
        unitary = _DummyUnitary(prop)
        layout = _DummyLayout()

        alg = ansatz_module.Ansatz.__new__(ansatz_module.Ansatz)
        alg.system_size = 8
        alg.unitaries = [unitary]
        alg._layout = layout
        alg.MPI_COMM_WORLD = MPI.COMM_WORLD

        fake_ciw = SimpleNamespace(
            wrapper_negotiate=lambda *args: (444, 201),
        )

        with monkeypatch.context() as mp:
            mp.setattr(ansatz_module, "_ciw", fake_ciw)
            with pytest.raises(RuntimeError, match="finalizing partition_table"):
                ansatz_module.Ansatz._Ansatz__check_comm_size.__wrapped__(alg)

        assert layout.set_layout_ptr_calls == [444]
        assert layout.destroy_calls == 1
        assert alg._layout is None

    def test_check_comm_size_raises_on_lock_status(self, monkeypatch):
        ansatz_module = import_module("quop_mpi.ansatz")
        prop = _DummyProp()
        unitary = _DummyUnitary(prop)
        layout = _DummyLayout()

        alg = ansatz_module.Ansatz.__new__(ansatz_module.Ansatz)
        alg.system_size = 8
        alg.unitaries = [unitary]
        alg._layout = layout
        alg.MPI_COMM_WORLD = MPI.COMM_WORLD

        fake_ciw = SimpleNamespace(
            wrapper_negotiate=lambda *args: (555, 301),
        )

        with monkeypatch.context() as mp:
            mp.setattr(ansatz_module, "_ciw", fake_ciw)
            with pytest.raises(RuntimeError, match="locking layout"):
                ansatz_module.Ansatz._Ansatz__check_comm_size.__wrapped__(alg)

        assert layout.set_layout_ptr_calls == [555]
        assert layout.destroy_calls == 1
        assert alg._layout is None
