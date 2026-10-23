"""Infer mapped parameter counts before assigning parallel derivative work."""

import numpy as np
import pytest

from quop_mpi.algorithm.combinatorial import QWOA
from quop_mpi.parameter_maps import linear_ramp
from tests.mpi.test_parallel_jacobian import skip_wavefront_parallel_jacobian

pytestmark = pytest.mark.mpi


def _qualities(system_size):
    values = np.linspace(-1.0, 2.0, system_size) ** 2

    def qualities(local_i, local_i_offset):
        return values[local_i_offset : local_i_offset + local_i]

    return qualities


def _dense_map(ansatz_depth, total_params, free_vec):
    """Exercise every input even when there are more inputs than output pairs."""
    count = ansatz_depth * total_params
    matrix = (np.arange(count * len(free_vec)).reshape(count, -1) % 7 + 1) / 10.0
    return matrix @ free_vec


def _replacement_map(ansatz_depth, total_params, free_vec):
    return np.tile([free_vec[0] ** 2, free_vec[-1] / 2.0], ansatz_depth)


def _reference_gradient(evaluate, x, method, h):
    reference = np.empty(len(x))
    baseline = evaluate(x) if method == "forward" else None
    for index in range(len(x)):
        forward = x.copy()
        forward[index] += h
        backward = x.copy()
        backward[index] -= h
        reference[index] = (
            (evaluate(forward) - baseline) / h
            if method == "forward"
            else (evaluate(forward) - evaluate(backward)) / (2.0 * h)
        )
    return reference


def _assert_assignment(mapping, count, n_workers):
    assert len(mapping) == n_workers
    assert mapping[0] == [], "The optimiser group must not compute partial derivatives"
    assigned = [index for group in mapping[1:] for index in group]
    assert sorted(assigned) == list(range(count)), "Each derivative must be assigned exactly once"
    if n_workers - 1 > count:
        assert any(not group for group in mapping[1:]), "Surplus worker groups should stay idle"


def _workers(mpi_comm):
    if mpi_comm.Get_size() < 2:
        pytest.skip("Need at least two MPI ranks for parallel derivative assignment")
    count = min(4, mpi_comm.Get_size())
    skip_wavefront_parallel_jacobian(count)
    return count


def _probe_optimiser(algorithm):
    """Request one gradient and retain a serial reference without taking a step."""

    def optimise(fun, x, jac, **kwargs):
        input_count = algorithm.n_free_params
        assignment = [group.copy() for group in algorithm.var_map]
        reference = _reference_gradient(fun, x, "central", 1e-5)
        value = fun(x)
        gradient = jac(x)
        return {
            "x": x.copy(),
            "fun": value,
            "jac": gradient,
            "reference_gradient": reference,
            "input_count": input_count,
            "assignment": assignment,
            "success": True,
            "nfev": 2 * len(x) + 1,
        }

    return optimise


def _assert_run(algorithm, mpi_comm, free_count):
    # Excluded ranks and optimiser-subcomm peers also need the inferred count.
    assert mpi_comm.allgather(algorithm.n_free_params) == [free_count] * mpi_comm.Get_size()
    result = mpi_comm.bcast(
        algorithm.result if algorithm.subcomms.is_optimiser_leader() else None,
        root=algorithm.subcomms.optimiser_leader_world_rank(),
    )
    assert result["input_count"] == free_count
    assert len(result["x"]) == free_count
    _assert_assignment(result["assignment"], free_count, algorithm.subcomms.get_n_subcomms())
    assert np.all(np.isfinite(result["jac"]))
    np.testing.assert_allclose(result["jac"], result["reference_gradient"], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("method", ["forward", "central"])
def test_direct_mapped_jacobian_reassigns_for_received_vector(mpi_comm, mpi_sizing, method):
    """A direct Jacobian after prepare uses the leader's vector on every worker."""
    n_workers = _workers(mpi_comm)
    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    with QWOA(system_size, mpi_comm) as algorithm:
        algorithm.set_qualities(_qualities(system_size))
        algorithm.set_depth(3)
        algorithm.set_parameter_map(_dense_map)
        algorithm.set_parallel_jacobian(n_workers=n_workers, method=method, h=1e-5)
        algorithm.prepare()
        with pytest.raises(RuntimeError):
            _ = algorithm.n_free_params

        for count in [1, 3, 2, 5, 1]:
            x = np.linspace(0.2, 0.7, count)
            gradient = algorithm._mpi_jacobian(
                x if algorithm.subcomms.is_optimiser_leader() else None
            )

            def evaluate(point):
                value = algorithm.evaluate(point)
                return 0.0 if value is None else value

            reference = _reference_gradient(evaluate, x, method, algorithm.h)
            result = None
            if algorithm.subcomms.is_optimiser_leader():
                result = (gradient, reference, algorithm.var_map)
            gradient, reference, mapping = mpi_comm.bcast(
                result, root=algorithm.subcomms.optimiser_leader_world_rank()
            )
            _assert_assignment(mapping, count, algorithm.subcomms.get_n_subcomms())
            assert gradient.shape == (count,)
            assert np.all(np.isfinite(gradient))
            np.testing.assert_allclose(gradient, reference, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("reverse_comm", [False, True])
def test_execute_rebuilds_assignments_before_optimiser(mpi_comm, mpi_sizing, reverse_comm):
    """Consecutive leader-only runs survive count, map, depth and worker changes."""
    n_workers = _workers(mpi_comm)
    comm = (
        mpi_comm.Split(0, mpi_comm.Get_size() - 1 - mpi_comm.Get_rank())
        if reverse_comm
        else mpi_comm
    )
    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    try:
        with QWOA(system_size, comm) as algorithm:
            algorithm.set_qualities(_qualities(system_size))
            algorithm.set_parameter_map(_dense_map)
            algorithm.set_depth(3)
            algorithm.set_parallel_jacobian(n_workers=n_workers, method="central", h=1e-5)
            algorithm.set_optimiser(_probe_optimiser(algorithm))

            for count in [1, 3, 2]:
                algorithm.setup()
                initial = np.linspace(0.2, 0.7, count)
                algorithm.execute(initial if algorithm.subcomms.is_optimiser_leader() else None)
                _assert_run(algorithm, comm, count)

            algorithm.set_parameter_map(_replacement_map)
            with pytest.raises(RuntimeError):
                _ = algorithm.n_free_params
            algorithm.set_depth(4)
            algorithm.set_parallel_jacobian(n_workers=2, method="forward", h=1e-6)
            algorithm.prepare()
            with pytest.raises(RuntimeError):
                _ = algorithm.n_free_params
            initial = np.array([0.4]) if algorithm.subcomms.is_optimiser_leader() else None
            algorithm.execute(initial)
            _assert_run(algorithm, comm, 1)
    finally:
        if reverse_comm:
            comm.Free()


def test_depth_one_linear_ramp_accepts_changed_free_count(mpi_comm, mpi_sizing):
    """The optional beta input acquires a zero derivative when supplied at depth one."""
    n_workers = _workers(mpi_comm)
    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    with QWOA(system_size, mpi_comm) as algorithm:
        algorithm.set_qualities(_qualities(system_size))
        algorithm.set_parameter_map(linear_ramp())
        algorithm.set_depth(1)
        algorithm.set_parallel_jacobian(n_workers=n_workers, method="central", h=1e-5)
        algorithm.set_optimiser(_probe_optimiser(algorithm))
        algorithm.setup()
        for initial in [np.array([0.8, 0.3]), np.array([0.8, 0.3, 0.2])]:
            algorithm.execute(initial if algorithm.subcomms.is_optimiser_leader() else None)
            _assert_run(algorithm, mpi_comm, len(initial))
            if algorithm.subcomms.is_optimiser_leader() and len(initial) == 3:
                assert algorithm.result["jac"][2] == 0.0


def test_benchmark_infers_mapped_count_from_leader_only_input(mpi_comm, mpi_sizing):
    """Benchmark forwards the leader's mapped vector before configuring gradients."""
    n_workers = _workers(mpi_comm)
    system_size = mpi_sizing.power_of_two(base=16, min_per_rank=1)
    with QWOA(system_size, mpi_comm) as algorithm:
        algorithm.set_qualities(_qualities(system_size))
        algorithm.set_parameter_map(_dense_map)
        algorithm.set_parallel_jacobian(n_workers=n_workers, method="central", h=1e-5)
        algorithm.set_optimiser(_probe_optimiser(algorithm))
        algorithm.setup()
        initial = np.array([0.4]) if algorithm.subcomms.is_optimiser_leader() else None
        algorithm.benchmark(
            ansatz_depths=[1, 3],
            repeats=1,
            initial_parameters=initial,
            verbose=False,
        )
        _assert_run(algorithm, mpi_comm, 1)


@pytest.mark.requires_nprocs(6)
def test_mapped_count_reaches_excluded_world_ranks(mpi_comm, mpi_sizing):
    """Excluded ranks learn the leader's count before workers evaluate its gradient."""
    from quop_mpi import config

    skip_wavefront_parallel_jacobian(2)
    # Two FFT elements force native MPI worker groups to exclude surplus ranks.
    # Keep a supported partition size on GPU backends, where exclusion differs.
    system_size = 2 if config.backend == "mpi" else mpi_sizing.power_of_two(base=16, min_per_rank=1)
    with QWOA(system_size, mpi_comm) as algorithm:
        algorithm.set_qualities(_qualities(system_size))
        algorithm.set_depth(2)
        algorithm.set_parameter_map(_dense_map)
        algorithm.set_parallel_jacobian(n_workers=2, method="central", h=1e-5)
        algorithm.set_optimiser(_probe_optimiser(algorithm))
        algorithm.setup()
        active_ranks = mpi_comm.allgather(algorithm.subcomms.in_subcomm())
        if config.backend == "mpi":
            assert not all(active_ranks), "The tiny native FFT must exercise excluded ranks"
        initial = np.array([0.2, 0.4, 0.7]) if algorithm.subcomms.is_optimiser_leader() else None
        algorithm.execute(initial)
        _assert_run(algorithm, mpi_comm, 3)
