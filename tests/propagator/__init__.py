"""
Propagator tests that can be run with both MPI and wavefront backends.

These tests verify low-level propagator functionality independent of
the high-level algorithm interface. They test:
- FFT-based operations (composite propagator)
- Eigenvalue computation
- Phase-shift application
- State normalization preservation

Run with MPI backend:
    mpiexec -n <N> python -m pytest tests/propagator/ -v

Run with wavefront backend:
    QUOP_BACKEND=wavefront mpiexec -n <N> python -m pytest tests/propagator/ -v
"""
