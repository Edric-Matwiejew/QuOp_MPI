.. _api-parameter-maps:

==============
Parameter Maps
==============

Predefined :term:`Parameter Map Functions <Parameter Map Function>` convert
vectors of :term:`free parameters` to :term:`variational parameters`.
Register a map with :meth:`~quop_mpi.ansatz.set_parameter_map`.

.. currentmodule:: quop_mpi.parameter_maps

.. _linear-ramp-parameter-map:

Linear Ramp
===========

The :func:`linear_ramp` function creates a parameter map that generates a
non-variational schedule from three values, ``[gamma, t, beta]``, independently
of :term:`ansatz depth`. Each ansatz iteration must start with a phase-shift
parameter followed by one or more mixer parameters, with one scalar parameter
per unitary. All mixers receive the same value within an iteration.

For depth :math:`p > 1`, iteration :math:`l = 1, \ldots, p` uses

.. math::

    x_l = \frac{l - 1}{p - 1}, \qquad
    \gamma_l = \frac{\gamma}{\sigma}
        \left[\beta + (1 - \beta)x_l\right], \qquad
    t_l = t\left[1 + (\beta - 1)x_l\right].

The phase-shift values ramp from :math:`\gamma\beta/\sigma` to
:math:`\gamma/\sigma`, while mixer values ramp from :math:`t` to
:math:`t\beta`. The map does not impose bounds on ``beta``. Here,
:math:`\sigma` is the population standard deviation of the observables. A
zero or non-finite standard deviation raises ``ValueError``.

For an algorithm ``alg`` with observables, an initial state and compatible
unitaries configured:

.. code-block:: python

    from quop_mpi.parameter_maps import linear_ramp

    alg.set_depth(3)
    alg.set_parameter_map(linear_ramp())
    alg.evolve_state([0.8, 0.4, 0.2])  # gamma, t, beta

Use :meth:`~quop_mpi.ansatz.execute` with the same vector to optimise the three
schedule values. The vector determines the number of free parameters.
:meth:`~quop_mpi.ansatz.benchmark` also requires an explicit
``initial_parameters`` vector when using a parameter map.

By default, the standard deviation gives equal weight to each basis-state
observable. To weight it by the initial-state probabilities
:math:`|\psi_0|^2`, set the option when creating the map:

.. code-block:: python

    alg.set_parameter_map(linear_ramp(weight_by_initial_state=True))

The standard deviation is computed once and reused by
:meth:`~quop_mpi.ansatz.evolve_state` and :meth:`~quop_mpi.ansatz.execute`.
After calling :meth:`~quop_mpi.ansatz.set_observables` or
:meth:`~quop_mpi.ansatz.set_initial_state`,
the next call to :meth:`~quop_mpi.ansatz.evolve_state` or
:meth:`~quop_mpi.ansatz.execute` recomputes it. Create a separate map for each
ansatz.

At depth one, supply ``[gamma, t]`` or retain three values; ``beta`` is
ignored. Other depths require three values.

Module Reference
================

.. automodule:: quop_mpi.parameter_maps
   :members:
