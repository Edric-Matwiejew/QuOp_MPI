"""Tests for predefined parameter maps without an ansatz or distributed state."""

import numpy as np
import pytest
from mpi4py import MPI

from quop_mpi.parameter_maps import linear_ramp


class _CountingComm:
    """Count the normalization's MPI reductions while performing them normally."""

    def __init__(self, comm):
        self.comm = comm
        self.reductions = 0

    def Allreduce(self, *args, **kwargs):  # noqa: N802
        self.reductions += 1
        return self.comm.Allreduce(*args, **kwargs)

    def allreduce(self, *args, **kwargs):
        self.reductions += 1
        return self.comm.allreduce(*args, **kwargs)


def _map(
    free_vec,
    *,
    depth=3,
    total_params=2,
    observables=None,
    state=None,
    weighted=False,
    mapper=None,
    comm=MPI.COMM_SELF,
):
    """Call the map on a self communicator with small default observables."""
    if observables is None:
        observables = np.array([1.0, 3.0, 5.0])
    if state is None:
        state = np.ones(len(observables), dtype=np.complex128)
    if mapper is None:
        mapper = linear_ramp(weight_by_initial_state=weighted)
    return mapper(
        depth,
        total_params,
        free_vec,
        observables,
        state,
        comm,
    )


def test_linear_ramp_population_normalization_and_layer_order():
    """The phase increases while every mixer time decreases, layer by layer."""
    gamma, time, beta = 0.9, 0.6, 0.2
    sigma = np.std([1.0, 3.0, 5.0], ddof=0)

    mapped = _map([gamma, time, beta], total_params=4)

    expected = [
        gamma * beta / sigma,
        time,
        time,
        time,
        gamma * (1 + beta) / (2 * sigma),
        time * (1 + beta) / 2,
        time * (1 + beta) / 2,
        time * (1 + beta) / 2,
        gamma / sigma,
        time * beta,
        time * beta,
        time * beta,
    ]
    np.testing.assert_allclose(mapped, expected)
    assert mapped.shape == (12,)
    assert mapped.dtype == np.float64


@pytest.mark.parametrize("free_vec", [[0.9, 0.6], [0.9, 0.6, -17.0]])
def test_linear_ramp_depth_one_ignores_optional_beta(free_vec):
    mapped = _map(free_vec, depth=1)

    np.testing.assert_allclose(mapped, [0.9 / np.std([1.0, 3.0, 5.0]), 0.6])


@pytest.mark.parametrize("beta", [-2.0, 0.0, 1.0, 3.0])
def test_linear_ramp_accepts_any_finite_beta(beta):
    mapped = _map([0.8, 0.3, beta], depth=2).reshape(2, 2)

    np.testing.assert_allclose(mapped[:, 0], 0.8 / np.std([1.0, 3.0, 5.0]) * np.array([beta, 1]))
    np.testing.assert_allclose(mapped[:, 1], [0.3, 0.3 * beta])


def test_linear_ramp_weights_variance_by_initial_probabilities():
    observables = np.array([-2.0, 0.0, 4.0])
    state = np.array([1j, 2.0, -3j])
    probabilities = np.abs(state) ** 2
    mean = np.average(observables, weights=probabilities)
    sigma = np.sqrt(np.average((observables - mean) ** 2, weights=probabilities))

    mapped = _map([0.9, 0.6], depth=1, observables=observables, state=state, weighted=True)

    np.testing.assert_allclose(mapped, [0.9 / sigma, 0.6])


def test_linear_ramp_unweighted_variance_is_independent_of_initial_state():
    state = np.array([1.0, 0.0, 0.0])

    mapped = _map([0.9, 0.6], depth=1, state=state)

    np.testing.assert_allclose(mapped, [0.9 / np.std([1.0, 3.0, 5.0]), 0.6])


@pytest.mark.parametrize("weighted", [False, True])
def test_linear_ramp_reuses_normalization_across_depths_and_parameters(weighted):
    mapper = linear_ramp(weight_by_initial_state=weighted)
    comm = _CountingComm(MPI.COMM_SELF)
    sigma = np.std([1.0, 3.0, 5.0])

    first = _map([0.9, 0.6], depth=1, mapper=mapper, comm=comm)

    np.testing.assert_allclose(first, [0.9 / sigma, 0.6])
    assert comm.reductions == 2
    for depth, total_params in [(3, 2), (5, 4)]:
        mapped = _map(
            [0.8, 0.4, 0.1],
            depth=depth,
            total_params=total_params,
            mapper=mapper,
            comm=comm,
        ).reshape(depth, total_params)
        fraction = np.linspace(0.0, 1.0, depth)

        np.testing.assert_allclose(mapped[:, 0], 0.8 / sigma * (0.1 + 0.9 * fraction))
        np.testing.assert_allclose(
            mapped[:, 1:],
            np.repeat((0.4 * (1.0 - 0.9 * fraction))[:, None], total_params - 1, axis=1),
        )
        assert comm.reductions == 2


def test_linear_ramp_new_mapper_reflects_changed_observables():
    mapper = linear_ramp()
    comm = _CountingComm(MPI.COMM_SELF)
    observables = np.array([1.0, 3.0, 5.0])
    first = _map([0.9, 0.6], depth=1, observables=observables, mapper=mapper, comm=comm)
    observables *= 2.0

    cached = _map([0.9, 0.6], depth=1, observables=observables, mapper=mapper, comm=comm)

    np.testing.assert_allclose(cached, first)
    assert comm.reductions == 2
    second = _map([0.9, 0.6], depth=1, observables=observables, mapper=linear_ramp(), comm=comm)
    np.testing.assert_allclose(second, [first[0] / 2, first[1]])
    assert comm.reductions == 4


def test_linear_ramp_new_weighted_mapper_reflects_changed_initial_state():
    mapper = linear_ramp(weight_by_initial_state=True)
    comm = _CountingComm(MPI.COMM_SELF)
    observables = np.array([1.0, 3.0, 5.0])
    state = np.ones(3, dtype=np.complex128)
    first = _map(
        [0.9, 0.6], depth=1, observables=observables, state=state, mapper=mapper, comm=comm
    )
    state[:] = [1.0, 0.0, 1.0]

    cached = _map(
        [0.9, 0.6], depth=1, observables=observables, state=state, mapper=mapper, comm=comm
    )

    np.testing.assert_allclose(cached, first)
    assert comm.reductions == 2
    second = _map(
        [0.9, 0.6],
        depth=1,
        observables=observables,
        state=state,
        mapper=linear_ramp(weight_by_initial_state=True),
        comm=comm,
    )
    assert not np.isclose(first[0], second[0])
    np.testing.assert_allclose(second, [0.9 / 2.0, 0.6])
    assert comm.reductions == 4


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("observables", [[2.0, 2.0, 2.0], [1.0, np.nan, 5.0]])
def test_linear_ramp_failed_normalization_can_be_retried(weighted, observables):
    mapper = linear_ramp(weight_by_initial_state=weighted)
    comm = _CountingComm(MPI.COMM_SELF)
    with pytest.raises(ValueError):
        _map([0.9, 0.6], depth=1, observables=observables, mapper=mapper, comm=comm)
    failed_reductions = comm.reductions

    mapped = _map([0.9, 0.6], depth=1, mapper=mapper, comm=comm)

    np.testing.assert_allclose(mapped, [0.9 / np.std([1.0, 3.0, 5.0]), 0.6])
    assert comm.reductions == failed_reductions + 2
    repeated = _map([0.9, 0.6], depth=1, mapper=mapper, comm=comm)
    np.testing.assert_allclose(repeated, mapped)
    assert comm.reductions == failed_reductions + 2


@pytest.mark.parametrize(
    "invalid",
    [
        {"depth": 0},
        {"total_params": 1},
        {"free_vec": [np.nan, 0.6, 0.2]},
    ],
)
def test_linear_ramp_validates_parameters_after_normalization_is_cached(invalid):
    mapper = linear_ramp()
    comm = _CountingComm(MPI.COMM_SELF)
    _map([0.9, 0.6, 0.2], mapper=mapper, comm=comm)

    with pytest.raises(ValueError):
        _map(**({"free_vec": [0.9, 0.6, 0.2]} | invalid), mapper=mapper, comm=comm)

    assert comm.reductions == 2


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_linear_ramp_rejects_invalid_depth(depth):
    with pytest.raises((TypeError, ValueError)):
        _map([0.9, 0.6, 0.2], depth=depth)


@pytest.mark.parametrize("total_params", [0, 1, 2.5, True])
def test_linear_ramp_rejects_invalid_layer_size(total_params):
    with pytest.raises((TypeError, ValueError)):
        _map([0.9, 0.6, 0.2], total_params=total_params)


@pytest.mark.parametrize(
    ("depth", "free_vec"),
    [
        (1, [0.9]),
        (1, [0.9, 0.6, 0.2, 0.1]),
        (3, [0.9, 0.6]),
        (3, [0.9, 0.6, 0.2, 0.1]),
        (3, [[0.9, 0.6, 0.2]]),
        (3, 0.9),
        (3, [np.nan, 0.6, 0.2]),
        (3, [0.9, np.inf, 0.2]),
        (1, [0.9, 0.6, np.nan]),
    ],
)
def test_linear_ramp_rejects_invalid_free_vector(depth, free_vec):
    with pytest.raises(ValueError):
        _map(free_vec, depth=depth)


@pytest.mark.parametrize(
    "observables",
    [[], [1.0], [2.0, 2.0], [1.0, np.nan], [1.0, np.inf]],
)
def test_linear_ramp_rejects_empty_nonfinite_or_constant_observables(observables):
    with pytest.raises(ValueError):
        _map([0.9, 0.6], depth=1, observables=np.asarray(observables))


def test_linear_ramp_rejects_nonfinite_variance():
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        _map([0.9, 0.6], depth=1, observables=np.array([-1e200, 1e200]))


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(3),
        np.ones(2),
        np.array([1.0, np.nan, 1.0]),
        np.array([1.0, np.inf, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    ],
)
def test_linear_ramp_rejects_invalid_or_zero_variance_weights(state):
    with pytest.raises(ValueError):
        _map([0.9, 0.6], depth=1, state=state, weighted=True)
