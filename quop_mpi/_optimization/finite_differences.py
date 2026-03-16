# cspell:words variational
"""Finite difference methods for numerical approximation of partial derivatives."""

from __future__ import annotations

from copy import copy
from typing import Callable

import numpy as np


def forward_differences(
    variational_parameters: np.ndarray[float], evaluate: Callable, h: float, var: int
) -> float:
    """Computes an approximation of the partial derivative of a QVA at point
    :literal:`variational_parameters` with respect to the parameter of index :literal:`var` using
    the forward differences method.

    Parameters
    ----------
    variational_parameters : ndarray[float]
        1-D real array of ansatz variational parameters
    evaluate : callable
        method or function for computation of the objective function value (see
        :meth:`~quop_mpi.ansatz.evaluate`)
    h : float
        step-size used in forward difference approximation
    var : int
        index of the variational parameter for which the partial derivative is
        to be approximated.

    Returns
    -------
    float
        approximate partial derivative
    """
    expectation = evaluate(variational_parameters)
    x = variational_parameters.copy()
    x[var] += h
    expectation_forward = evaluate(x)
    return (expectation_forward - expectation) / h


def central(
    variational_parameters: np.ndarray[float], evaluate: Callable, h: float, var: int
) -> float:
    """Computes an approximation of the partial derivative of a QVA at point
    :literal:`variational_parameters` with respect to the parameter of index :literal:`var` using
    the central differences method.

    Parameters
    ----------
    variational_parameters : ndarray[float]
        1-D real array of ansatz variational parameters
    evaluate : callable
        method or function for computation of the objective function value (see
        :meth:`~quop_mpi.ansatz.evaluate`)
    h : float
        Step-size used in central difference approximation.
    var : int
        index of the variational parameter for which the partial derivative is
        to be approximated.

    Returns
    -------
    float
        approximate partial derivative.
    """
    x_back = copy(variational_parameters)
    x_forward = copy(variational_parameters)
    x_back[var] -= h
    x_forward[var] += h
    expectation_back = evaluate(x_back)
    expectation_forward = evaluate(x_forward)
    return (expectation_forward - expectation_back) / (2 * h)
