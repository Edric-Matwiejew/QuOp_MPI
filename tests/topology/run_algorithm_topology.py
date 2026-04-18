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


def run_qaoa(system_size):
    """QAOA with diagonal phase + transverse_field mixer."""
    from quop_mpi.algorithm.combinatorial import QAOA, serial

    def zero_qualities():
        return np.zeros(system_size, dtype=np.float64)

    alg = QAOA(system_size)
    alg.set_qualities(serial, {"args": [zero_qualities]})
    alg.set_depth(1)
    alg.execute()


def run_qwoa(system_size):
    """QWOA with diagonal phase + circulant mixer."""
    from quop_mpi.algorithm.combinatorial import QWOA, serial

    def zero_qualities():
        return np.zeros(system_size, dtype=np.float64)

    alg = QWOA(system_size)
    alg.set_qualities(serial, {"args": [zero_qualities]})
    alg.set_depth(1)
    alg.execute()


def run_qmoa(Ns):
    """QMOA with diagonal phase + composite mixer."""
    from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

    n_dims = len(Ns)
    bounds = n_dims * [[-1.0, 1.0]]
    deltas, mins = setup_cartesian(Ns, bounds)

    def zero_function(x):
        return np.zeros(x.shape[1], dtype=np.float64)

    alg = QMOA(Ns)
    alg.set_qualities(cartesian, {"args": [deltas, mins, zero_function]})
    alg.set_depth(1)
    alg.execute()


USAGE = """\
Usage:
  {prog} qaoa <system_size>
  {prog} qwoa <system_size>
  {prog} qmoa <N1> <N2> [N3 ...]
"""


def main():
    if len(sys.argv) < 3:
        print(USAGE.format(prog=sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    algorithm = sys.argv[1].lower()

    if algorithm == "qaoa":
        run_qaoa(int(sys.argv[2]))
    elif algorithm == "qwoa":
        run_qwoa(int(sys.argv[2]))
    elif algorithm == "qmoa":
        Ns = [int(x) for x in sys.argv[2:]]
        run_qmoa(Ns)
    else:
        print(f"Unknown algorithm: {algorithm}", file=sys.stderr)
        print(USAGE.format(prog=sys.argv[0]), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
