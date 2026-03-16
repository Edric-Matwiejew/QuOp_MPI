"""Predefined :term:`Operator Functions <Operator Function>` for
:class:`quop_mpi.propagator.composite.unitary`.

The composite propagator handles operators that are Cartesian sums of
circulant operators acting on subregisters. This is used for multivariable
optimization problems where the mixing operator decomposes as:

.. math::

    H = H_1 \\otimes I \\otimes \\cdots + I \\otimes H_2 \\otimes \\cdots + \\cdots

where each :math:`H_i` is a circulant matrix acting on a subregister.

**Return Format**

The Operator Function must return:

    eigenvalues : ndarray[float64, shape=(max(Ns), n_dims)]
        A 2-D real array where :literal:`eigenvalues[:N_i, i]` contains the
        eigenvalues of the :literal:`i`-th circulant suboperator. The first
        dimension is :literal:`max(Ns)` (padded with zeros for smaller grids),
        and the second dimension corresponds to the number of grid dimensions.

**Propagation Method**

The composite propagator exploits the Cartesian sum structure using
**multi-dimensional FFT**:

1. Forward multi-dimensional FFT transforms the state to the joint eigenbasis
<<<<<<< HEAD
2. Multiply by :math:`e^{-it(\lambda^{(1)}_{k_1} + \lambda^{(2)}_{k_2} + \cdots)}`
   where :math:`\lambda^{(d)}_{k_d}` are eigenvalues for each dimension
3. Inverse multi-dimensional FFT transforms back to the computational basis

This avoids constructing the full :math:`\prod_i N_i` sized operator by
exploiting the separable structure of Cartesian sums. The implementation
uses FFTW with MPI for parallel multi-dimensional transforms.
"""
=======
2. Multiply by :math:`e^{-it(\\lambda^{(1)}_{k_1} + \\lambda^{(2)}_{k_2} + \\cdots)}`
   where :math:`\\lambda^{(d)}_{k_d}` are eigenvalues for each dimension
3. Inverse multi-dimensional FFT transforms back to the computational basis

This avoids constructing the full :math:`\\prod_i N_i` sized operator by
exploiting the separable structure of Cartesian sums. The implementation
uses FFTW with MPI for parallel multi-dimensional transforms.
"""

>>>>>>> quop_quisa/main
from .eigenvalues import ith

__all__ = ["ith"]
