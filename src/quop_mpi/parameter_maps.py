"""Predefined :term:`Parameter Map Functions<Parameter Map Function>`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mpi4py import MPI
from numpy.typing import ArrayLike, NDArray

__all__ = ["linear_ramp"]


def _observable_std(
    local_observables: ArrayLike,
    ansatz_initial_state: ArrayLike,
    MPI_COMM: MPI.Comm,  # noqa: N803
    weight_by_initial_state: bool,
) -> float:
    """Compute a population standard deviation collectively on MPI_COMM."""
    observables = np.asarray(local_observables, dtype=np.float64).reshape(-1)
    weights = None
    valid = bool(np.all(np.isfinite(observables)))
    if weight_by_initial_state:
        state = np.asarray(ansatz_initial_state, dtype=np.complex128).reshape(-1)
        weights = np.abs(state) ** 2
        valid = valid and weights.shape == observables.shape and bool(np.all(np.isfinite(weights)))

    # Reduce validation along with the moments so a bad local partition raises
    # on every rank before any rank enters the variance reduction.
    local_moments = np.zeros(3, dtype=np.float64)
    if valid:
        if weights is None:
            local_moments[:2] = observables.size, np.sum(observables)
        else:
            local_moments[:2] = np.sum(weights), np.dot(weights, observables)
    else:
        local_moments[2] = 1.0
    global_moments = np.empty_like(local_moments)
    MPI_COMM.Allreduce(local_moments, global_moments, op=MPI.SUM)
    weight, weighted_sum, invalid_partitions = global_moments
    if invalid_partitions:
        if weight_by_initial_state:
            raise ValueError(
                "Observables and initial-state probabilities have incompatible sizes "
                "or contain NaN or infinity"
            )
        raise ValueError("Observables must not contain NaN or infinity")
    if weight <= 0.0 or not np.isfinite(weight):
        if weight_by_initial_state:
            raise ValueError("Initial state has zero, NaN or infinite norm")
        raise ValueError("No observables were provided")
    if not np.isfinite(weighted_sum):
        raise ValueError("Observable sum is NaN or infinite")

    mean = weighted_sum / weight
    squared_deviations = (observables - mean) ** 2
    local_variance = (
        np.sum(squared_deviations) if weights is None else np.dot(weights, squared_deviations)
    )
    variance = MPI_COMM.allreduce(float(local_variance), op=MPI.SUM) / weight
    if variance <= 0.0 or not np.isfinite(variance):
        raise ValueError("Observable standard deviation is zero, NaN or infinite")
    return float(np.sqrt(variance))


def linear_ramp(weight_by_initial_state: bool = False) -> Callable[..., NDArray[np.float64]]:
    """Create a parameter map for the non-variational QWOA linear schedule [1]_.

    Pass the returned :term:`Parameter Map Function` to
    :meth:`~quop_mpi.ansatz.set_parameter_map` as
    ``alg.set_parameter_map(linear_ramp())``. Each ansatz iteration must contain
    a :term:`phase-shift unitary` followed by one or more
    :term:`mixing unitaries <mixing unitary>`, with one parameter per unitary.

    The observable standard deviation is computed once and reused by
    :meth:`~quop_mpi.ansatz.evolve_state` and :meth:`~quop_mpi.ansatz.execute`.
    Calling :meth:`~quop_mpi.ansatz.set_observables` or
    :meth:`~quop_mpi.ansatz.set_initial_state` causes the next call to
    :meth:`~quop_mpi.ansatz.evolve_state` or :meth:`~quop_mpi.ansatz.execute`
    to recompute it. Use a separate map for each ansatz.

    Parameters
    ----------
    weight_by_initial_state : bool, optional
        Whether to weight the observable variance by initial-state
        probabilities, by default ``False`` (equal weight per basis state).

    Returns
    -------
    callable
        Map from ``(gamma, t, beta)`` to a 1-D array of
        ``ansatz_depth * total_params`` parameters, ordered by ansatz iteration
        as ``[gamma_1, t_1, ..., gamma_2, t_2, ...]``. Each mixing unitary in an
        iteration receives the same time.

    Notes
    -----
    Typical choices are ``gamma < 0``, with ``abs(gamma)`` of order one,
    ``t > 0``, and ``0 < beta < 1`` [1]_. The scale of ``t`` depends on
    the mixing unitaries. These ranges are not enforced by the map, so use
    of an appropriately configured optimiser is recommended (see
    :meth:`~quop_mpi.ansatz.set_optimiser`).

    For :term:`ansatz depth` ``p > 1`` and iteration ``l = 1, ..., p``, let
    ``x = (l - 1) / (p - 1)``:

    ``gamma_l = (gamma / sigma) * (beta + (1 - beta) * x)``

    ``t_l = t * (1 + (beta - 1) * x)``.

    Here ``sigma`` is the population standard deviation of the observables.
    The map requires a positive, finite ``sigma``. At depth one the schedule
    is ``[gamma / sigma, t, ...]``; beta is ignored and ``(gamma, t)`` is also
    accepted.

    References
    ----------
    .. [1] T. Bennett, L. Noakes and J. B. Wang, "Analysis of the
       Non-variational Quantum Walk-based Optimisation Algorithm", 2024.
       https://arxiv.org/abs/2408.06368
    """
    return _LinearRampMap(weight_by_initial_state)


class _LinearRampMap:
    """Linear-ramp mapper with normalisation tied to ansatz preparation."""

    def __init__(self, weight_by_initial_state: bool) -> None:
        self._weight_by_initial_state = weight_by_initial_state
        self._sigma: float | None = None

    def _reset_cache(self) -> None:
        self._sigma = None

    def __call__(
        self,
        ansatz_depth: int,
        total_params: int,
        free_vec: ArrayLike,
        local_observables: ArrayLike,
        ansatz_initial_state: ArrayLike,
        MPI_COMM: MPI.Comm,  # noqa: N803
    ) -> NDArray[np.float64]:
        if (
            isinstance(ansatz_depth, (bool, np.bool_))
            or not isinstance(ansatz_depth, (int, np.integer))
            or ansatz_depth < 1
        ):
            raise ValueError(f"ansatz_depth must be a positive integer, got {ansatz_depth!r}")
        if (
            isinstance(total_params, (bool, np.bool_))
            or not isinstance(total_params, (int, np.integer))
            or total_params < 2
        ):
            raise ValueError(f"total_params must be an integer >= 2, got {total_params!r}")
        free_vec = np.asarray(free_vec, dtype=np.float64)
        valid_sizes = (2, 3) if ansatz_depth == 1 else (3,)
        if free_vec.ndim != 1 or free_vec.size not in valid_sizes:
            expected_shape = "(2,) or (3,)" if ansatz_depth == 1 else "(3,)"
            raise ValueError(
                f"Expected free_vec with shape {expected_shape} at depth {ansatz_depth}, "
                f"got {free_vec.shape}"
            )
        if not np.all(np.isfinite(free_vec)):
            raise ValueError("free_vec must not contain NaN or infinity")

        if self._sigma is None:
            self._sigma = _observable_std(
                local_observables, ansatz_initial_state, MPI_COMM, self._weight_by_initial_state
            )
        sigma = self._sigma
        gamma, walk_time = free_vec[:2]
        params = np.empty((ansatz_depth, total_params), dtype=np.float64)
        if ansatz_depth == 1:
            params[0, 0] = gamma / sigma
            params[0, 1:] = walk_time
        else:
            beta = free_vec[2]
            fraction = np.linspace(0.0, 1.0, ansatz_depth)
            params[:, 0] = (gamma / sigma) * (beta + (1.0 - beta) * fraction)
            params[:, 1:] = (walk_time * (1.0 + (beta - 1.0) * fraction))[:, None]
        return params.ravel()
