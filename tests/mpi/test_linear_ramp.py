"""Distributed normalization and ansatz integration for the linear-ramp map."""

import numpy as np
import pytest

from quop_mpi.parameter_maps import linear_ramp

pytestmark = pytest.mark.mpi


class _CountingComm:
    """Count normalization reductions on a real distributed communicator."""

    def __init__(self, comm):
        self.comm = comm
        self.reductions = 0

    def Allreduce(self, *args, **kwargs):  # noqa: N802
        self.reductions += 1
        return self.comm.Allreduce(*args, **kwargs)

    def allreduce(self, *args, **kwargs):
        self.reductions += 1
        return self.comm.allreduce(*args, **kwargs)


@pytest.mark.parametrize("empty_partitions", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_linear_ramp_uses_global_population_variance(mpi_comm, empty_partitions, weighted):
    """Uneven partitions, including empty ranks, match a serial calculation."""
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    count = 2 * size + 1
    observables = np.linspace(-3.0, 4.0, count) ** 3
    state = np.arange(1, count + 1) * np.exp(1j * np.arange(count))
    if empty_partitions:
        local_observables = observables if rank == 0 else np.empty(0)
        local_state = state if rank == 0 else np.empty(0, dtype=np.complex128)
    else:
        local_observables = np.array_split(observables, size)[rank]
        local_state = np.array_split(state, size)[rank]
    if weighted:
        mean = np.average(observables, weights=np.abs(state) ** 2)
        variance = np.average((observables - mean) ** 2, weights=np.abs(state) ** 2)
    else:
        variance = np.var(observables)

    mapper = linear_ramp(weight_by_initial_state=weighted)
    mapped = mapper(
        3,
        2,
        [0.9, 0.6, 0.2],
        local_observables,
        local_state,
        mpi_comm,
    )

    np.testing.assert_allclose(
        mapped,
        [
            0.18 / np.sqrt(variance),
            0.6,
            0.54 / np.sqrt(variance),
            0.36,
            0.9 / np.sqrt(variance),
            0.12,
        ],
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_linear_ramp_caches_global_normalization(mpi_comm, weighted):
    """Changing depth and free parameters performs no further reductions."""
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    observables = np.arange(2 * size + 1, dtype=np.float64) ** 2
    state = np.arange(1, observables.size + 1, dtype=np.complex128)
    local_observables = np.array_split(observables, size)[rank]
    local_state = np.array_split(state, size)[rank]
    if weighted:
        probabilities = np.abs(state) ** 2
        mean = np.average(observables, weights=probabilities)
        sigma = np.sqrt(np.average((observables - mean) ** 2, weights=probabilities))
    else:
        sigma = np.std(observables)
    mapper = linear_ramp(weight_by_initial_state=weighted)
    comm = _CountingComm(mpi_comm)

    first = mapper(1, 2, [0.9, 0.6], local_observables, local_state, comm)

    np.testing.assert_allclose(first, [0.9 / sigma, 0.6])
    assert comm.reductions == 2
    repeated = mapper(3, 2, [0.8, 0.4, 0.1], local_observables, local_state, comm)

    np.testing.assert_allclose(repeated, [0.08 / sigma, 0.4, 0.44 / sigma, 0.22, 0.8 / sigma, 0.04])
    assert comm.reductions == 2


@pytest.mark.parametrize("invalid", ["observables", "weights", "weight_shape"])
def test_linear_ramp_rejects_invalid_partition_collectively(mpi_comm, invalid):
    """One rank's invalid local data raises on every participating rank."""
    observables = np.array([0.0, 2.0])
    state = np.ones(2)
    if mpi_comm.Get_rank() == 0:
        if invalid == "observables":
            observables[0] = np.nan
        elif invalid == "weights":
            state[0] = np.inf
        else:
            state = np.ones(1)

    mapper = linear_ramp(weight_by_initial_state=True)
    with pytest.raises(ValueError):
        mapper(
            3,
            2,
            [0.9, 0.6, 0.2],
            observables,
            state,
            mpi_comm,
        )


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_linear_ramp_qwoa_matches_expanded_parameters(mpi_comm, mpi_sizing, depth, weighted):
    """Interface binding supplies the current observables, state, and subcommunicator."""
    from quop_mpi.algorithm.combinatorial import QWOA

    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    observables = np.linspace(-2.0, 3.0, system_size) ** 2
    amplitudes = np.arange(1, system_size + 1, dtype=np.float64)
    state = amplitudes.astype(np.complex128) / np.linalg.norm(amplitudes)

    def qualities(local_i, local_i_offset):
        return observables[local_i_offset : local_i_offset + local_i]

    def initial_state(local_i, local_i_offset):
        return state[local_i_offset : local_i_offset + local_i]

    free_vec = np.array([0.9, 0.6]) if depth == 1 else np.array([0.9, 0.6, 0.2])
    if weighted:
        mean = np.average(observables, weights=np.abs(state) ** 2)
        sigma = np.sqrt(np.average((observables - mean) ** 2, weights=np.abs(state) ** 2))
    else:
        sigma = np.std(observables)
    if depth == 1:
        expanded = np.array([free_vec[0] / sigma, free_vec[1]])
    else:
        expanded = np.array([0.18 / sigma, 0.6, 0.54 / sigma, 0.36, 0.9 / sigma, 0.12])

    states = []
    for use_map in [True, False]:
        algorithm = QWOA(system_size, mpi_comm)
        try:
            algorithm.set_qualities(qualities)
            algorithm.set_initial_state(initial_state)
            algorithm.set_depth(depth)
            if use_map:
                algorithm.set_parameter_map(
                    linear_ramp(weight_by_initial_state=weighted),
                )
            algorithm.evolve_state(free_vec if use_map else expanded)
            states.append(algorithm.get_final_state())
        finally:
            algorithm.destroy()

    if states[0] is not None:
        np.testing.assert_allclose(states[0], states[1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
def test_linear_ramp_cache_tracks_ansatz_setup(mpi_comm, mpi_sizing, monkeypatch, weighted):
    """Setup changes invalidate normalization while repeated evaluations reuse it."""
    from quop_mpi import parameter_maps
    from quop_mpi.algorithm.combinatorial import QWOA

    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    observables = np.linspace(-2.0, 3.0, system_size) ** 2
    state = np.ones(system_size, dtype=np.complex128) / np.sqrt(system_size)
    normalizations = []
    observable_std = parameter_maps._observable_std

    def count_normalization(*args, **kwargs):
        sigma = observable_std(*args, **kwargs)
        normalizations.append(sigma)
        return sigma

    monkeypatch.setattr(parameter_maps, "_observable_std", count_normalization)

    def qualities(local_i, local_i_offset):
        return observables[local_i_offset : local_i_offset + local_i]

    def initial_state(local_i, local_i_offset):
        return state[local_i_offset : local_i_offset + local_i]

    mapper = linear_ramp(weight_by_initial_state=weighted)
    free_vec = np.array([0.9, 0.6, 0.2])
    algorithm = QWOA(system_size, mpi_comm)

    def evolve_and_check(expected_normalizations):
        algorithm.evolve_state(free_vec)
        if weighted:
            probabilities = np.abs(state) ** 2
            mean = np.average(observables, weights=probabilities)
            sigma = np.sqrt(np.average((observables - mean) ** 2, weights=probabilities))
        else:
            sigma = np.std(observables)
        fraction = np.linspace(0.0, 1.0, algorithm.ansatz_depth)
        expected = np.column_stack(
            [0.9 / sigma * (0.2 + 0.8 * fraction), 0.6 * (1.0 - 0.8 * fraction)]
        ).ravel()
        np.testing.assert_allclose(algorithm.last_evaluated, expected)
        assert len(normalizations) == expected_normalizations
        np.testing.assert_allclose(normalizations[-1], sigma)

    try:
        algorithm.set_qualities(qualities)
        algorithm.set_initial_state(initial_state)
        algorithm.set_depth(3)
        algorithm.set_parameter_map(mapper)
        algorithm.prepare()
        algorithm.prepare()
        assert not normalizations

        evolve_and_check(1)
        algorithm.prepare()
        evolve_and_check(1)
        algorithm.set_depth(5)
        algorithm.prepare()
        evolve_and_check(1)

        observables *= 2.0
        algorithm.set_qualities(qualities)
        algorithm.prepare()
        assert len(normalizations) == 1
        evolve_and_check(2)
        np.testing.assert_allclose(normalizations[1], 2.0 * normalizations[0])

        amplitudes = np.arange(1, system_size + 1, dtype=np.float64)
        state[:] = amplitudes / np.linalg.norm(amplitudes)
        algorithm.set_initial_state(initial_state)
        algorithm.prepare()
        assert len(normalizations) == 2
        evolve_and_check(3)
        if weighted:
            assert not np.isclose(normalizations[2], normalizations[1])
        algorithm.prepare()
        evolve_and_check(3)
    finally:
        algorithm.destroy()


@pytest.mark.parametrize("n_workers", [1, 2])
def test_linear_ramp_execute_records_final_mapped_parameters(mpi_comm, mpi_sizing, n_workers):
    """Final result construction evaluates collective maps on the whole subcomm."""
    from quop_mpi.algorithm.combinatorial import QWOA

    if mpi_comm.Get_size() < n_workers:
        pytest.skip(f"Need at least {n_workers} MPI processes")
    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    observables = np.linspace(-2.0, 3.0, system_size) ** 2
    sigma = np.std(observables)

    def qualities(local_i, local_i_offset):
        return observables[local_i_offset : local_i_offset + local_i]

    algorithm = QWOA(system_size, mpi_comm)
    try:
        algorithm.set_qualities(qualities)
        algorithm.set_depth(3)
        algorithm.set_parameter_map(linear_ramp())
        if n_workers > 1:
            algorithm.set_parallel_jacobian(n_workers=n_workers, method="forward")
        algorithm.set_optimiser("scipy", {"method": "BFGS", "options": {"maxiter": 1}})

        algorithm.execute(np.array([0.9, 0.6, 0.2]))

        if algorithm.subcomms.is_optimiser_leader():
            gamma, time, beta = algorithm.result["x"]
            expected = [
                gamma * beta / sigma,
                time,
                gamma * (1.0 + beta) / (2.0 * sigma),
                time * (1.0 + beta) / 2.0,
                gamma / sigma,
                time * beta,
            ]
            np.testing.assert_allclose(algorithm.quop_result["mapped_parameters"], expected)
    finally:
        algorithm.destroy()
