#!/usr/bin/env python3
"""
grover_params.py

Utilities for Grover search with M marked items in a database of size N.

Returns:
- theta: the Grover angle, sin(theta) = sqrt(M/N)
- k_opt: the (near-)optimal integer number of *standard* Grover iterations
        using π-phase oracle and π-phase diffusion, maximizing success prob.
- last_phase: the phase-matched angle λ for an "exact Grover" final step
        (oracle and diffusion use the same phase λ), after running
        k_floor standard Grover iterations first.

Notes:
- Standard Grover uses oracle phase = diffusion phase = π.
- Exact Grover: do k_floor standard steps, then one final phase-matched step
  with phase λ chosen to land exactly in the marked subspace (up to global phase).
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass


@dataclass(frozen=True)
class GroverResult:
    theta: float          # in radians
    k_opt: int            # rounded optimum for π/π Grover
    k_floor: int          # floor choice used before the exact final step
    last_phase: float     # λ in radians, phase-matched (oracle=diffuser=λ)


def grover_params(M: int, N: int) -> GroverResult:
    """
    Compute Grover parameters for M marked items in a search space of size N.

    Parameters
    ----------
    M : int
        Number of marked items (1 <= M <= N).
    N : int
        Search space size (N >= 1).

    Returns
    -------
    GroverResult
        theta: Grover angle
        k_opt: optimal number of standard Grover iterations (π phases), by rounding
        k_floor: floor number of standard iterations used before a final exact step
        last_phase: phase-matched angle λ for the final exact step (oracle=diffuser=λ)

    Raises
    ------
    ValueError
        If inputs are invalid or the problem is degenerate (M=0 or M=N).
    """
    if not (isinstance(M, int) and isinstance(N, int)):
        raise ValueError("M and N must be integers.")
    if N <= 0:
        raise ValueError("N must be positive.")
    if M <= 0 or M > N:
        raise ValueError("Require 1 <= M <= N.")

    # Degenerate cases:
    if M == N:
        # Already all marked; theta = pi/2, no steps needed, last phase irrelevant.
        return GroverResult(theta=math.pi / 2, k_opt=0, k_floor=0, last_phase=0.0)

    # Grover angle
    theta = math.asin(math.sqrt(M / N))

    # "Near-optimal" number of *standard* Grover iterations with π/π phases:
    # maximize sin^2((2k+1)theta) by choosing (2k+1)theta closest to pi/2.
    k_real = (math.pi / (4.0 * theta)) - 0.5
    k_opt = int(round(k_real))
    if k_opt < 0:
        k_opt = 0

    # For exact Grover, do k_floor standard iterations, then one final matched-phase step.
    k_floor = int(math.floor(k_real))
    if k_floor < 0:
        k_floor = 0

    # State after k_floor standard Grover iterations:
    # |psi> = sin(gamma)|w> + cos(gamma)|r>, where gamma=(2k_floor+1)theta.
    gamma = (2 * k_floor + 1) * theta
    a = math.sin(gamma)  # amplitude on |w>
    b = math.cos(gamma)  # amplitude on |r>

    # Solve for z = e^{iλ} such that applying Q_λ = D_λ O_λ maps b' (|r> coeff) to 0,
    # with phase matching (oracle phase = diffuser phase = λ).
    #
    # In the {|w>,|r>} basis, with |s> = sinθ|w> + cosθ|r>,
    # the condition b' = 0 yields a quadratic in z on the unit circle:
    #
    #   (cosθ * sinθ * a) z^2 - (cosθ * (sinθ*a - cosθ*b)) z + (b * sin^2θ) = 0
    #
    s = math.sin(theta)
    c = math.cos(theta)

    # Coefficients for A z^2 + B z + C = 0
    A = c * s * a
    B = -c * (s * a - c * b)
    C = b * (s * s)

    last_phase = 0.0

    # If A is ~0, the quadratic collapses to linear or trivial; handle safely.
    eps = 1e-14
    if abs(A) < eps:
        # Linear: B z + C = 0  (unless B~0)
        if abs(B) < eps:
            # If also C~0, we're already in |w> (or numerical coincidence).
            last_phase = 0.0
        else:
            z = -C / B
            # Project onto unit circle if numerical drift:
            if abs(z) > eps:
                z /= abs(z)
            last_phase = float(cmath.phase(z) % (2 * math.pi))
    else:
        disc = B * B - 4 * A * C
        sqrt_disc = cmath.sqrt(disc)
        z1 = (-B + sqrt_disc) / (2 * A)
        z2 = (-B - sqrt_disc) / (2 * A)

        # Choose the root closest to the unit circle (|z|=1), then project to unit circle.
        cand = min((z1, z2), key=lambda z: abs(abs(z) - 1.0))
        if abs(cand) > eps:
            cand /= abs(cand)
        last_phase = float(cmath.phase(cand) % (2 * math.pi))

    return GroverResult(theta=theta, k_opt=k_opt, k_floor=k_floor, last_phase=last_phase)


if __name__ == "__main__":
    # Example usage
    examples = [(1, 1024), (4, 1024), (10, 1000), (100, 10000)]
    for M, N in examples:
        r = grover_params(M, N)
        print(f"M={M:>4}, N={N:>5} | theta={r.theta:.6g} rad | "
              f"k_opt={r.k_opt} | k_floor={r.k_floor} | last_phase λ={r.last_phase:.6g} rad")
