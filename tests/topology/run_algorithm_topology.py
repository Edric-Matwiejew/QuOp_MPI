#!/usr/bin/env python3
"""Algorithm-class integration test runners for topology verification.

These scripts exercise the full negotiate path including FFT-dependent
propagators (FFTW for MPI backend, SHAFFT for wavefront backend) to
verify topology consistency under real negotiation constraints.

Usage
-----
  export QUOP_DUMP_COMM_INFO=1
  mpirun -n 8 python run_algorithm_topology.py qaoa 128
  mpirun -n 8 python run_algorithm_topology.py qwoa 128
  mpirun -n 8 python run_algorithm_topology.py qmoa 4 4

For QMOA the arguments are per-dimension exponents (n_qubits per dim),
so ``4 4`` means a 2^4 × 2^4 = 256 state space.
"""

import sys

import numpy as np


def run_qaoa(system_size, n_workers=1):
    """QAOA with diagonal phase + transverse_field mixer."""
    from quop_mpi.algorithm.combinatorial import QAOA, serial

    def zero_qualities():
        return np.zeros(system_size, dtype=np.float64)

    alg = QAOA(system_size)
    alg.set_qualities(serial, {"args": [zero_qualities]})
    alg.set_depth(1)
    if n_workers > 1:
        alg.set_parallel_jacobian(n_workers)
    alg.execute()


def run_qwoa(system_size, n_workers=1):
    """QWOA with diagonal phase + circulant mixer."""
    from quop_mpi.algorithm.combinatorial import QWOA, serial

    def zero_qualities():
        return np.zeros(system_size, dtype=np.float64)

    alg = QWOA(system_size)
    alg.set_qualities(serial, {"args": [zero_qualities]})
    alg.set_depth(1)
    if n_workers > 1:
        alg.set_parallel_jacobian(n_workers)
    alg.execute()


def run_qmoa(Ns, n_workers=1):
    """QMOA with diagonal phase + composite mixer."""
    from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

    n_dims = len(Ns)
    bounds = n_dims * [[-1.0, 1.0]]
    deltas, mins = setup_cartesian(Ns, bounds)

    def zero_function(x):
        return np.zeros(x.shape[0], dtype=np.float64)

    alg = QMOA(Ns)
    alg.set_qualities(cartesian, {"args": [deltas, mins, zero_function]})
    alg.set_depth(1)
    if n_workers > 1:
        alg.set_parallel_jacobian(n_workers)
    alg.execute()


USAGE = """\
Usage:
  {prog} qaoa <system_size> [--workers N]
  {prog} qwoa <system_size> [--workers N]
  {prog} qmoa <N1> <N2> [N3 ...] [--workers N]
"""


def main():
    if len(sys.argv) < 3:
        print(USAGE.format(prog=sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    # Extract --workers before parsing positional args
    args = list(sys.argv[1:])
    n_workers = 1
    if "--workers" in args:
        idx = args.index("--workers")
        n_workers = int(args[idx + 1])
        del args[idx : idx + 2]

    algorithm = args[0].lower()

    if algorithm == "qaoa":
        run_qaoa(int(args[1]), n_workers)
    elif algorithm == "qwoa":
        run_qwoa(int(args[1]), n_workers)
    elif algorithm == "qmoa":
        Ns = [int(x) for x in args[1:]]
        run_qmoa(Ns, n_workers)
    else:
        print(f"Unknown algorithm: {algorithm}", file=sys.stderr)
        print(USAGE.format(prog=sys.argv[0]), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
