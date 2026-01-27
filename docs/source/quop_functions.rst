QuOp Functions
==============
.. glossary::

    QuOp Function
        QuOp Functions define the various aspects of a :term:`QVA` or modify the
        simulation methods used by the :class:`quop_mpi.Ansatz` class.

        **Implementation patterns**

        QuOp Functions can be implemented in three ways, depending on whether
        you need to maintain state between calls:

        1. **Plain function** — simplest, for stateless computations:

           .. code-block:: python

               def my_observables(local_i, local_i_offset, scale, *args, **kwargs):
                   """Compute observables from scratch each time."""
                   result = np.zeros(local_i, dtype=np.float64)
                   for j in range(local_i):
                       result[j] = compute_cost(local_i_offset + j) * scale
                   return result

               # Usage:
               ansatz.set_observables(my_observables, {"args": [2.0]})

        2. **Factory function (closure)** — for caching or stateful behaviour:

           .. code-block:: python

               def create_my_function(config_param: float):
                   """Factory returning a stateful QuOp Function."""
                   _cache = {}  # state persists across calls

                   def my_function(local_i, local_i_offset, *args, **kwargs):
                       # config_param captured from enclosing scope
                       # _cache persists between calls (e.g., for expensive one-time setup)
                       if "data" not in _cache:
                           _cache["data"] = expensive_computation()
                       # ... use _cache["data"] ...
                       return result

                   return my_function

               # Usage:
               ansatz.set_observables(create_my_function(scale=2.0), obs_dict)

        3. **Callable class** — for complex state or easier debugging:

           .. code-block:: python

               class MyFunction:
                   def __init__(self, config_param: float):
                       self.config_param = config_param
                       self._cache = None  # state as instance attributes

                   def __call__(self, local_i, local_i_offset, *args, **kwargs):
                       if self._cache is None:
                           self._cache = expensive_computation()
                       # ... implementation ...
                       return result

               # Usage:
               ansatz.set_observables(MyFunction(scale=2.0), obs_dict)

        Use plain functions when no state is needed. Use the factory/closure
        pattern when you need to cache expensive computations (e.g., computing
        global statistics via MPI) or carry configuration. Use callable classes
        when you need subclassing, multiple methods, or easier state inspection
        for debugging.

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
        :mod:`quop_mpi.observable` module. See :class:`quop_mpi.Ansatz` for a
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

        Predefined Initial State Functions are included in the
        :mod:`quop_mpi.state` module. See :class:`quop_mpi.Ansatz` for a
        selected list of available attributes.

        **Typical structure:**

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
        Defines a mapping from a reduced "free" parameter vector to the full
        variational-parameter vector used by a :term:`QVA`.  This allows you to
        optimise over a smaller parameter space while the mapping function
        reconstructs the complete vector internally.

        Passed to :meth:`quop_mpi.Ansatz.set_parameter_map` together with the
        number of free parameters and an optional :term:`FunctionDict`.

        **Method signature:**

        .. code-block:: python

            ansatz.set_parameter_map(
                n_free_params,   # int: dimensionality of the optimisation problem
                mapping_fn,      # callable: your mapping function
                mapping_dict     # optional FunctionDict for extra arguments
            )

        **Typical structure:**

        The mapping function receives the free parameter vector as its first
        argument. Additional positional parameters (e.g., ``ansatz_depth``,
        ``observables``, ``MPI_COMM``) are automatically bound from the
        :class:`quop_mpi.Ansatz` instance.

        .. code-block:: python

            def mapping_fn(
                free_vec: np.ndarray,
                ansatz_depth: int,
                total_params: int,
                *args,
                **kwargs
            ) -> np.ndarray:
                """
                Map free_vec to full parameter vector of shape
                (ansatz_depth * total_params,).
                """
                full_vec = np.zeros(ansatz_depth * total_params, dtype=np.float64)
                # ... your mapping logic ...
                return full_vec

        **Factory pattern example:**

        For Parameter Map Functions, the factory pattern conveniently returns
        both ``n_free_params`` and the mapping function together:

        .. code-block:: python

            def create_linear_schedule(scale: float):
                """Factory returning (n_free_params, mapping_fn)."""
                n_free_params = 3
                _cache = {}

                def mapping_fn(free_vec, ansatz_depth, observables, MPI_COMM):
                    if "sigma" not in _cache:
                        _cache["sigma"] = compute_global_std(observables, MPI_COMM)
                    # ... build full_vec from free_vec ...
                    return full_vec

                return n_free_params, mapping_fn

            # Usage:
            n_free, param_map = create_linear_schedule(scale=1.0)
            ansatz.set_parameter_map(n_free, param_map)

        See :term:`QuOp Function` for the general implementation patterns
        (factory/closure vs callable class).

    Sampling Function
        Returns an :term:`objective function` value computed from batches of
        :term:`observables` values that are sampled based on the probability
        distribution of the wavefunction state vector during simulation together
        with a boolean that specifies whether the :term:`objective function`
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
        specific :class:`unitary` class. See :class:`quop_mpi.Unitary`.

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
        with an instance of the :class:`quop_mpi.Unitary` class.

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
        Passed to :meth:`quop_mpi.Ansatz.set_objective`.

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