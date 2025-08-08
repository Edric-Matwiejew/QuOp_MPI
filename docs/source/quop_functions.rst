QuOp Functions
==============
.. glossary::

    QuOp Function
        QuOp Functions define the various aspects of a :term:`QVA` or modify the
        simulation methods used by the :class:`quop_mpi.Ansatz` class.

    FunctionDict
        Prior to :term:`QVA` simulation, positional arguments of a QuOp Function
        are bound to the attributes of the receiving class if a match is found.
        Additional positional and keyword are specified via a FunctionDict:

        .. code-block:: python

            function_dict : {"args":List[Any], "kwargs":Dict}

        The ``"args"`` and ``"kwargs"`` elements of a FunctionDict are both
        optional. If present, these are passed to a bound QuOp Function as:

        .. code-block:: python

            bound_quop_function(*function_dict["args"], **function_dict["kwargs"])

    Observables Function
        Returns a 1-D  real array containing ``local_i`` elements of the
        :term:`observables` with global offset ``local_i_offset``. Passed to the
        :meth:`quop_mpi.Ansatz.set_observables` method and bound to the
        attributes of the :class:`quop_mpi.Ansatz` class.

        Predefined Observables Functions are included in the
        :mod:`quop_mpi.state` module. See :class:`quop_mpi.Ansatz` for a
        selected list of available attributes.

        **Typical structure:**

        .. code-block:: python

            def observables_function(
                system_size : int
                local_i : int,
                local_i_offset : int,
                *args,
                **kwargs) -> np.ndarray[np.complex128]:

                ...

                return observables

    Initial State Function
        Returns a 1-D complex array containing ``local_i`` elements of the
        :term:`initial state` with global offset ``local_i_offset``. Passed to
        the :meth:`quop_mpi.Ansatz.set_initial_state` method and bound to the
        attributes of the :class:`quop_mpi.Ansatz` class.

        **Typical structure:**

        Predefined Initial State Functions are included in the
        :mod:`quop_mpi.state` module. See :class:`quop_mpi.Ansatz` for a
        selected list of available attributes.

        See :class:`quop_mpi.Ansatz` for a selected list of available attributes.

        .. code-block:: python

            def initial_state_function(
                system_size : int
                local_i : int,
                local_i_offset : int,
                *args,
                **kwargs) -> np.ndarray[np.complex128]:

                ...

                return initial_state

    Parameter Map Function
        Defines a mapping from a reduced “free” parameter vector to the full
        variational-parameter vector used by a :term:`QVA`.  This allows you to
        optimise only a subset of parameters while reconstructing the complete
        vector internally.
    
        A Parameter Map Function is passed to
        :meth:`quop_mpi.Ansatz.set_parameter_map` together with an optional
        :term:`FunctionDict` of extra arguments.  At runtime the free vector
        is bound via the same interface machinery used by other QuOp Functions.
    
        **FunctionDict usage:**
    
        .. code-block:: python
        
            mapping_dict : {"args": List[Any], "kwargs": Dict[str,Any]}
    
        Any entries in `"args"` and `"kwargs"` will be forwarded to your map
        function after the free-vector argument.
    
        **Typical structure:**
    
        .. code-block:: python
        
            def parameter_map_function(
                ansatz_depth: int, # number of iterations
                total_params: int, # parameters per ansatz iteration
                free_vec: np.ndarray,
                *args,
                **kwargs
            ) -> np.ndarray:
                """
                Return full_vec of shape (ansatz_depth * total_params,)
                by embedding or expanding the entries in free_vec according
                to your chosen scheme.
                """
                # e.g. start with a copy of the previous full vector or zeros
                full_vec = np.zeros(ansatz_depth * total_params, dtype=np.float64)
    
                # insert free parameters into selected indices
                for idx, var_idx in enumerate(free_indices):
                    full_vec[var_idx] = free_vec[idx]
    
                # optionally fill remaining entries via some rule
                # full_vec[other_indices] = ...
    
                return full_vec
    
        When you call
    
        .. code-block:: python
        
            ansatz.set_parameter_map(parameter_map_function, mapping_dict)
    
        the `parameter_map_function` will be bound to the `Ansatz` instance and
        invoked automatically whenever variational parameters must be expanded
        from the free vector.
   
    Sampling Function
        Returns an :term:`objective function` value computed from batches of
        :term:`observables` values that are sampled based on the probability
        distribution of the wavefunction state vector during simulation together
        with a boolean that specifies wether the :term:`objective function`
        value should be passed to the :term:`optimiser` or more sample batches
        taken. Passed to :meth:`quop_mpi.Ansatz.set_sampling`.

        See :class:`quop_mpi.Ansatz` for a selected list of available attributes,

        .. note::

            The :class:`quop_mpi.Ansatz` class computes the expectation value
            exactly by default.

        **Typical Structure**

        .. code-block:: python

            def sampling_function(
                samples : List[ndarray[float64]],
                *args,
                **kwargs
            ) -> (float, bool)

                ...

                return objective_function_value, value_accepted

        The ``samples`` argument is a list of 1-D real arrays containing
        ``sample_block_size`` :term:`observables` values. If
        ``value_accepted`` is not ``True``, an additional sample block is
        appended to ``samples``.

    Jacobian Function
        Enables distributed parallel computation of the :term:`objective
        function` gradient. Returns the partial derivative of the
        :term:`objective function` with respect to the variational parameter
        with index ``var``. Used to compute the :term:`objective function`
        gradient is parallel if using a gradient-informed :term:`optimiser`.
        Passed to :meth:`quop_mpi.Ansatz.set_parallel_jacobian`.

        The :class:`quop_mpi.Ansatz` supports numerical approximation of the
        gradient using the forward and central finite difference methods
        (specified via :meth:`quop_mpi.Ansatz.set_parallel_jacobian`).  See
        :class:`quop_mpi.Ansatz` for a list of available attributes.

        .. note::

            * The :class:`quop_mpi.Ansatz` class computes the :term:`objective
              function` gradient sequentially by default. 
            
            * The default optimisation method of the :class:`quop_mpi.Ansatz`
              class, the BFGS algorithm, is gradient informed.

        **Typical Structure**

        .. code-block:: python

            def jacobian_function(
                variational_parameters: np.ndarray[np.float],
                evaluate: Callable,
                var: int,
                *args,
                **kwargs
            ) -> float:

            ...

                return partial_derivative_value

        The ``evaluate`` argument is bound to the
        :meth:`quop_mpi.Ansatz.evaluate` method which implements lazy
        computation of the :term:`objective function`. This is the recommended
        method for use in numerical approximation of the gradient by
        finite-difference methods.

    Operator Function
        Returns an :term:`operator` object that is compatible with the propagation method of
        specific :class:`unitary` class. See :meth:`quop_mpi.Unitary`.

        Predefined Operator Functions are included with each ``unitary`` class
        in the :mod:`quop_mpi.propagator` module under
        ``quop_mpi.propagator.<unitary>.operator``. See
        :class:`quop_mpi.Unitary` and the predefined ``unitary`` classes in the
        :mod:`quop_mpi.propagator` module for lists of available attributes.

        **Typical Structure**

        .. code-block:: python
            
            def operator_function(
                system_size : int,
                local_i : int,
                local_i_offset : int,
                variational_parameters : ndarray[float],
                *args,
                **kwargs
            ) -> Any:

                ...

                return operator

        Operator Functions with one or more :term:`variational parameters`
        require the ``variational_parameters`` positional argument. Operator
        Functions with no :term:`variational parameters` do not.

    Parameter Function
        Returns initial values for the :term:`variational parameters` associated
        with an instance of the :meth:`quop_mpi.Unitary` class.

        Predefined Parameter Functions are included in the :mod:`quop_mpi.param`
        module. See :class:`quop_mpi.Unitary` for a list of available
        attributes.

        **Typical Structure**

        .. code-block:: python

            def param_function(
               n_params : int,
               *args,
               **kwargs
            ) -> List[float]:

                ...

                return variational_parameters
        
    Objective Function
        Called after state-evolution during parameter optimisation. Returns a
        scalar value for minimisation.
        Passed to `quop_mpi.Ansatz.set_objective`.

        **Typical Structure**

        .. code-block:: python

            def objective_function(
               local_probabilities: nd.array[np.float64],
               observables: nd.array[np.float64],
               MPI_COMM: MPI.Intracomm,
               *args,
               **kwargs
            ) -> float:

                ...

                return objective_function_value