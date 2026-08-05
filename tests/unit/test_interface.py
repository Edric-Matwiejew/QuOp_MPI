"""Unit tests for the generic callable binding helper."""

import warnings

import pytest

from quop_mpi._utils._interface import Interface


class _FakeComm:
    def Get_rank(self):  # noqa: N802
        return 0


class _BoundAttrs:
    partition_table = [0, 4]
    MPI_COMM = _FakeComm()  # noqa: N815
    local_i = 8
    local_i_offset = 4


class _DeferredBoundAttrs:
    variational_parameters = None


class TestInterfaceWarnings:
    """Warning behavior should account for explicit FunctionDict inputs."""

    def test_explicit_call_args_satisfy_required_parameters(self):
        def serial_like(partition_table, MPI_COMM, function, *args, **kwargs):  # noqa: N803
            return function(*args, **kwargs)

        def qualities(graph):
            return graph

        interface = Interface(
            [_BoundAttrs()],
            serial_like,
            "observable",
            _FakeComm(),
            call_args=[qualities, "graph"],
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            interface.update_parameters()

        assert len(record) == 0
        assert interface.call(qualities, "graph") == "graph"

    def test_explicit_call_kwargs_satisfy_required_parameters(self):
        def serial_like(partition_table, MPI_COMM, *, function):  # noqa: N803
            return function()

        interface = Interface(
            [_BoundAttrs()],
            serial_like,
            "observable",
            _FakeComm(),
            call_kwargs={"function": lambda: 7},
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            interface.update_parameters()

        assert len(record) == 0
        assert interface.call(function=lambda: 7) == 7

    def test_warns_when_required_parameter_is_still_missing(self):
        def serial_like(partition_table, MPI_COMM, function, *args, **kwargs):  # noqa: N803
            return function(*args, **kwargs)

        interface = Interface(
            [_BoundAttrs()],
            serial_like,
            "observable",
            _FakeComm(),
        )

        with pytest.warns(
            UserWarning,
            match="Interface 'observable': parameter 'function' not found",
        ):
            interface.update_parameters()

    def test_raises_when_explicit_positional_precedes_bindable_parameters(self):
        def invalid_observable(scale_factor, local_i, local_i_offset):
            return scale_factor, local_i, local_i_offset

        with pytest.raises(
            TypeError,
            match="FunctionDict\\['args'\\] must trail all bindable positional parameters",
        ):
            Interface(
                [_BoundAttrs()],
                invalid_observable,
                "observable",
                _FakeComm(),
                call_args=[0.5],
            )

    def test_raises_when_explicit_positional_splits_bindable_parameters(self):
        def invalid_observable(local_i, scale_factor, local_i_offset):
            return local_i, scale_factor, local_i_offset

        with pytest.raises(
            TypeError,
            match="FunctionDict\\['args'\\] must trail all bindable positional parameters",
        ):
            Interface(
                [_BoundAttrs()],
                invalid_observable,
                "observable",
                _FakeComm(),
                call_args=[0.5],
            )

    def test_raises_when_explicit_kwarg_targets_bindable_parameter(self):
        def invalid_observable(local_i):
            return local_i

        with pytest.raises(
            TypeError,
            match="FunctionDict\\['kwargs'\\] cannot target bindable parameter 'local_i'",
        ):
            Interface(
                [_BoundAttrs()],
                invalid_observable,
                "observable",
                _FakeComm(),
                call_kwargs={"local_i": 0.5},
            )

    def test_raises_when_explicit_kwarg_targets_positional_only_parameter(self):
        def invalid_observable(local_i, scale_factor, /):
            return local_i, scale_factor

        with pytest.raises(
            TypeError,
            match="FunctionDict\\['kwargs'\\] cannot target positional-only parameter 'scale_factor'",
        ):
            Interface(
                [_BoundAttrs()],
                invalid_observable,
                "observable",
                _FakeComm(),
                call_kwargs={"scale_factor": 0.5},
            )

    def test_raises_on_bindable_keyword_only_parameter(self):
        def invalid_observable(*, local_i):
            return local_i

        with pytest.raises(
            TypeError,
            match="bindable keyword-only parameter 'local_i' is not supported",
        ):
            Interface(
                [_BoundAttrs()],
                invalid_observable,
                "observable",
                _FakeComm(),
            )

    def test_deferred_bindable_none_raises_on_runtime_update(self):
        def deferred_function(variational_parameters):
            return variational_parameters

        attrs = _DeferredBoundAttrs()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            interface = Interface(
                [attrs],
                deferred_function,
                "objective",
                _FakeComm(),
            )

        assert len(record) == 0

        with pytest.raises(
            TypeError,
            match="bindable parameter 'variational_parameters' exists on a bound object but is None",
        ):
            interface.update_parameters()

        attrs.variational_parameters = [1.0, 2.0]
        interface.update_parameters()

        assert interface.call() == [1.0, 2.0]
