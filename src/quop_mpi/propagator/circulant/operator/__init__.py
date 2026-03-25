"""Predefined :term:`Operator Functions <Operator Function>` for
:class:`quop_mpi.propagator.circulant.unitary`.

An Operator Function for :literal:`'circulant'` :literal:`unitary` instances returns a
:literal:`local_i` sized partition of the :term:`operator` eigenvalues with global index offset
:literal:`local_i_offset`.

**Return Format**

The Operator Function must return:

    eigenvalues : ndarray[float64]
        A 1-D real array of size :literal:`local_i` containing the eigenvalues
        of the circulant operator for global indices :literal:`local_i_offset`
        to :literal:`local_i_offset + local_i - 1`.

Circulant matrices are diagonalized by the Discrete Fourier Transform (DFT),
so the eigenvalues are simply the DFT of the first row (or column) of the matrix.

**Propagation Method**

The circulant propagator computes :math:`e^{-itH}|\\psi\\rangle` using FFT:

1. Forward FFT transforms the state to the eigenbasis
2. Multiply by :math:`e^{-it\\lambda_k}` where :math:`\\lambda_k` are eigenvalues
3. Inverse FFT transforms back to the computational basis

This is efficient because circulant matrices are diagonalized by the DFT.
The implementation uses FFTW with MPI for parallel execution.
"""

from .eigenvalues import complete, graph

__all__ = ["complete", "graph"]
