#!/usr/bin/env python3
"""Minimal diagonal-only QuOp ansatz for topology verification.

Triggers ``execute()`` which, with ``QUOP_DUMP_COMM_INFO=1``,
produces the comm_info dump file used by ``verify_topology.py``.

Usage
-----
  export QUOP_DUMP_COMM_INFO=1
  mpirun -n 8 python run_diagonal_topology.py 128

  # Or via srun on Setonix:
  srun -N1 -n8 python run_diagonal_topology.py 128
"""

import sys

import numpy as np

from quop_mpi import Ansatz
from quop_mpi.observable import serial as obs_serial
from quop_mpi.propagator.diagonal import Unitary as DiagonalUnitary
from quop_mpi.propagator.diagonal.operator import serial as diag_serial


def zero_diagonal(system_size):
    """Return a trivial zero-cost diagonal.

    The actual values are irrelevant — we only need the negotiate /
    communicator setup to run and produce the dump.
    """
    return np.zeros(system_size, dtype=np.float64)


def zero_observable(system_size):
    """Trivial observable (all zeros)."""
    return np.zeros(system_size, dtype=np.float64)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <system_size> [--workers N]", file=sys.stderr)
        sys.exit(1)

    system_size = int(sys.argv[1])
    n_workers = 1
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        n_workers = int(sys.argv[idx + 1])

    UQ = DiagonalUnitary(
        diag_serial,
        operator_dict={"args": [zero_diagonal, system_size]},
    )

    alg = Ansatz(system_size)
    alg.set_unitaries([UQ])
    alg.set_observables(obs_serial, {"args": [zero_observable, system_size]})
    alg.set_depth(1)

    if n_workers > 1:
        alg.set_parallel_jacobian(n_workers)

    alg.execute()


if __name__ == "__main__":
    main()
