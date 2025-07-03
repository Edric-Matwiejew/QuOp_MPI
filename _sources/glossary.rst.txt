Glossary
========

.. glossary::

    QVA
        A Quantum Variational Algorithm. For a quantum system with :term:`system size` 
        basis states (i.e. a complex vector of length :term:`system size`),
        QuOp_MPI simulates QVAs of the form,

        .. math::

            |\theta \rangle=\left( \prod_{i = 1}^{D}\hat{U}(\theta_i) \right) | \Psi_0 \rangle

        where :math:`| \theta \rangle` is the :term:`final state` of the quantum
        system, :math:`D` is the :term:`ansatz depth`, :math:`\hat{U}` is the
        :term:`ansatz unitary`, :math:`\theta = \theta_i` are real
        :term:`variational parameters` and :math:`|\Psi_0\rangle` is the
        :term:`initial state` of the quantum system. 
        
        See :class:`quop_mpi.Ansatz`.

    system state
        The quantum system prior to or after the action of the :term:`ansatz
        unitary` (the :term:`initial state` or :term:`final state`).

    system size
        The number of basis states (size) of the simulated quantum system.

        See :class:`quop_mpi.Ansatz`.

    initial state
        The starting :term:`system state`, by default an equal superposition
        of all states.

        See :meth:`quop_mpi.Ansatz.set_initial_state`.
        
    final state
        The :term:`system state` after the action of the :term:`ansatz unitary`.

        See :meth:`quop_mpi.Ansatz.get_final_state`.

    ansatz unitary
        The sequence of :term:`unitaries <unitary>` that constitute one
        :term:`ansatz iteration <ansatz depth>`.

        See :meth:`quop_mpi.Ansatz.set_unitaries`.

    unitary 
        A unitary operator parameterised by and arbitrary number of
        :term:`variational parameters`,

        .. math::

            \hat{U}_i(\phi) = \exp(-\text{i} (\phi_0 \hat{M}(\phi_1,...,\phi_m))

        where :math:`\phi = (\phi_0,...,\phi_m)` are a sequential subsection of
        the :term:`variational parameters` and :math:`\hat{M}` is a matrix
        :term:`operator`. The :math:`\phi_0` is a :term:`unitary parameter` and
        :math:`(\phi_1,...,\phi_m)` are :term:`operator parameters <operator parameter>`.

        See :class:`quop_mpi.Unitary` and :mod:`quop_mpi.propagator`.

    operator
        The matrix exponent of a :term:`unitary`, parameterised by arbitrary
        number (none or more) of :term:`variational parameters`.

    unitary parameter
        A :term:`variational parameter <variational parameters>` that scales the elements of an :term:`operator`
        via multiplication.

        See :class:`quop_mpi.Unitary` and :mod:`quop_mpi.propagator`.

    operator parameter
        :term:`Variational parameter <variational parameters>` that
        parametertise the structure of an :term:`operator`.

        See :class:`quop_mpi.Unitary` and :mod:`quop_mpi.propagator`.
        
    Phase-Shift Unitary
        A :term:`unitary` with a diagonal :term:`operator`. Typically used to
        phase-encode the solution :term:`quality values <observables>` of a
        particular optimisation problem.

        See :mod:`quop_mpi.propagator.diagonal`.
    
    Mixing Unitary
        A :term:`unitary` whose :term:`operator` has off-diagonal elements.
        Drives the transfer of probability amplitude between quantum basis
        states.

        See,
        
        * :mod:`quop_mpi.propagator.sparse`
        * :mod:`quop_mpi.propagator.circulant`
        * :mod:`quop_mpi.propagator.composite`
        * :mod:`quop_mpi.propagator.momentum`

    ansatz depth
        The number of repeats (or s) of the :term:`ansatz unitary`.

        See :meth:`quop_mpi.Ansatz.set_depth`.

    variational parameters 
        Classically tunable parameters of an :term:`ansatz unitary`. Each
        :term:`ansatz iteration <ansatz depth>` is associated with its own subset of parameters,
        such that the total number of varitional parameters grows linearly with
        the :term:`ansatz depth`.

    observables
        A real vector of scalar quality values associated with each simulated basis
        state (lower is better). Defines the diagonal observables operator :math:`\hat{O}`. 

        See :meth:`quop_mpi.Ansatz.set_observables`.

    objective function
        The expectation value of the :term:`observables` operator, minimised by
        the classical :term:`optimiser`. 

        .. math::
            
            \langle \theta | \hat{O} | \theta \rangle

        :meth:`quop_mpi.Ansatz.objective_function`.

    free parameters
        The subset of :term:`variational parameters` to be tuned via classical
        optimisation of the :term:`objective function`. See
        :meth:`quop_mpi.Ansatz.set_free_params`.

        :meth:`quop_mpi.Ansatz.set_free_params`.

    optimiser
        The classical optimiser responsible for minimisation of the
        :term:`objective function` via variation of the 
        :term:`free variational parameters <free parameters>`.

        :meth:`quop_mpi.Ansatz.set_optimiser`.