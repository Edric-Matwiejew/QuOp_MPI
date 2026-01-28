"""Predefined :term:`Operator Functions <Operator Function>` for
:class:`quop_mpi.propagator.momentum.unitary`.

The momentum propagator implements kinetic energy evolution for
continuous-variable quantum optimization (QOWE - Quantum Optimization
with Wavepacket Evolution).

**Return Format**

The Operator Function must return:

    momentums : ndarray[complex128, shape=(max(Ns), n_dims)]
        A 2-D complex array where :literal:`momentums[:N_i, i]` contains the
        squared momentum values for the :literal:`i`-th dimension. The first
        dimension is :literal:`max(Ns)` (padded with zeros for smaller grids),
        and the second dimension corresponds to the number of grid dimensions.

The momentum values are computed from the grid parameters
(:literal:`minsk`, :literal:`deltask`) specified when constructing the unitary.

**Propagation Method**

The momentum propagator computes :math:`e^{-itT}|\\psi\\rangle` where
:math:`T` is the kinetic energy operator, using **multi-dimensional FFT**:

1. Forward multi-dimensional FFT transforms the state from position to momentum space
2. Multiply by :math:`e^{-it(k_1^2 + k_2^2 + \cdots)}` (kinetic phase)
3. Inverse multi-dimensional FFT transforms back to position space

The implementation uses FFTW with MPI for parallel multi-dimensional transforms.
Grid parameters (:literal:`Ns`, :literal:`minsq`, :literal:`deltasq`, etc.)
are specified when constructing the unitary.
"""
from .grids import magnitude_squared

__all__ = ["magnitude_squared"]
