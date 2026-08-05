"""Optimization utilities for QuOp_MPI.

This module provides:
- Finite difference methods for numerical gradient approximation
- Parallel Jacobian computation using MPI subcommunicators
- NLopt optimizer wrapper
"""

from .finite_differences import central, forward_differences
from .parallel_jacobian import Jacobian

__all__ = ["Jacobian", "forward_differences", "central"]
