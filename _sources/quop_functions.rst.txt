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
        Additional positional and keyword arguments are specified via a FunctionDict:

        .. code-block:: python

            function_dict : {"args":List[Any], "kwargs":Dict}

        The ``"args"`` and ``"kwargs"`` elements of a FunctionDict are both
        optional. If present, these are passed to a bound QuOp Function as:

        .. code-block:: python

            bound_quop_function(*function_dict["args"], **function_dict["kwargs"])

Bindable Attributes
-------------------

When defining a QuOp Function, positional parameters are automatically bound
to class attributes **by matching the parameter name to the attribute name**.
This is the key mechanism: if your function has a parameter named ``local_i``,
it will automatically receive the value of that attribute.

**Binding sources differ by function type**:

* **Ansatz-level functions** (Observables, Initial State, Parameter Map,
  Sampling, Objective): bind to :class:`quop_mpi.Ansatz` attributes
* **Unitary-level functions** (Operator, Parameter): bind to
  :class:`quop_mpi.Unitary` attributes

Many attributes (like ``local_i``, ``system_size``, ``MPI_COMM``) are shared
between Ansatz and Unitary, so they work in both contexts. However, some
attributes are specific to each class.

Ansatz Bindable Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

These attributes are available for Observables, Initial State, Parameter Map,
Sampling, and Objective Functions:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter Name
     - Type
     - Description
   * - ``system_size``
     - int
     - Total number of quantum basis states
   * - ``local_i``
     - int
     - Number of elements in this MPI rank's partition
   * - ``local_i_offset``
     - int
     - Global index offset for this rank's partition
   * - ``partition_table``
     - ndarray[int]
     - Array where ``partition_table[rank+1] - partition_table[rank] = local_i``
   * - ``observables``
     - ndarray[float64]
     - Local partition of observable values (available after setup)
   * - ``ansatz_initial_state``
     - ndarray[complex128]
     - Local partition of the initial state vector
   * - ``final_state``
     - ndarray[complex128]
     - Local partition of current/final state vector
   * - ``variational_parameters``
     - ndarray[float64]
     - Current variational parameter values
   * - ``ansatz_depth``
     - int
     - Number of ansatz iterations (layers)
   * - ``total_params``
     - int
     - Number of variational parameters per ansatz iteration
   * - ``MPI_COMM``
     - MPI.Intracomm
     - MPI subcommunicator for this Ansatz instance
   * - ``expectation``
     - float
     - Last computed objective function value
   * - ``seed``
     - int
     - Random seed for parameter generation

**Important**: Parameter names in your function signature must exactly match
the attribute names above to be bound. Parameters that don't match will need
to be provided via ``FunctionDict["args"]``.

**Naming convention for custom parameters**: To avoid unintended binding, we
recommend prefixing your custom parameter names with an underscore:

.. code-block:: python

    def my_observables(
        local_i,           # bound from Ansatz
        local_i_offset,    # bound from Ansatz  
        _n_customers,      # custom - passed via FunctionDict["args"]
        _penalty_weight,   # custom - passed via FunctionDict["args"]
    ):
        ...

    # Usage:
    ansatz.set_observables(my_observables, {"args": [10, 0.5]})

This prevents accidental collisions with current or future Ansatz attributes
(e.g., ``seed``, ``expectation``).

**Runtime discovery**: Use these methods to discover available bindings:

* ``ansatz.print_bindable_attributes()`` — show Ansatz attributes only
* ``ansatz.print_all_bindable_attributes()`` — show Ansatz AND all Unitary attributes
* ``unitary.print_bindable_attributes()`` — show attributes for a specific Unitary

For programmatic access, use ``get_bindable_attributes()`` which returns a dictionary.

**Extensibility**: Subclasses (algorithms, propagators) can extend the available
bindable attributes by defining their own ``BINDABLE_ATTRIBUTES`` class variable.
The discovery methods automatically collect attributes from the entire class
hierarchy.

Unitary Bindable Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These attributes are available for Operator and Parameter Functions. They are
bound from the :class:`quop_mpi.Unitary` instance:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter Name
     - Type
     - Description
   * - ``system_size``
     - int
     - Total number of quantum basis states (shared with Ansatz)
   * - ``local_i``
     - int
     - Number of elements in this MPI rank's partition (shared)
   * - ``local_i_offset``
     - int
     - Global index offset for this rank's partition (shared)
   * - ``partition_table``
     - ndarray[int]
     - Array describing global partitioning (shared)
   * - ``MPI_COMM``
     - MPI.Intracomm
     - MPI subcommunicator (shared)
   * - ``seed``
     - int
     - Random seed for parameter generation (shared)
   * - ``alloc_local``
     - int
     - Size of the operator array (equals ``local_i`` for non-array operators)
   * - ``lb``
     - int
     - Lower global index of the local partition
   * - ``ub``
     - int
     - Upper global index of the local partition
   * - ``n_params``
     - int
     - Total parameters for this Unitary (operator + unitary params)
   * - ``operator_n_params``
     - int
     - Number of operator variational parameters
   * - ``unitary_n_params``
     - int
     - Number of unitary variational parameters
   * - ``variational_parameters``
     - ndarray[float64]
     - Operator variational parameters (only for parameterised operators)
   * - ``initial_state``
     - ndarray[complex128]
     - Local partition of input state to this unitary
   * - ``final_state``
     - ndarray[complex128]
     - Local partition of output state from this unitary

.. note::

   Attributes marked "(shared)" have the same values in both Ansatz and
   Unitary contexts. Unitary-specific attributes like ``variational_parameters``
   are only meaningful for Operator Functions that define parameterised operators.

.. glossary::

    Observables Function
        Returns a 1-D  real array containing ``local_i`` elements of the
        :term:`observables` with global offset ``local_i_offset``. Passed to the
        :meth:`quop_mpi.Ansatz.set_observables` method and bound to the
        attributes of the :class:`quop_mpi.Ansatz` class.

        Predefined Observables Functions are included in the
        :mod:`quop_mpi.observable` module.

        **Commonly used parameters**:

        * ``local_i`` — number of observables this rank must compute
        * ``local_i_offset`` — starting global index for this rank
        * ``system_size`` — total number of basis states
        * ``partition_table`` — for advanced partitioning schemes

        **Typical structure:**

        .. code-block:: python

            def observables_function(
                local_i : int,
                local_i_offset : int,
                *args,
                **kwargs) -> np.ndarray[np.float64]:

                ...

                return observables

    Initial State Function
        Returns a 1-D complex array containing ``local_i`` elements of the
        :term:`initial state` with global offset ``local_i_offset``. Passed to
        the :meth:`quop_mpi.Ansatz.set_initial_state` method and bound to the
        attributes of the :class:`quop_mpi.Ansatz` class.

        Predefined Initial State Functions are included in the
        :mod:`quop_mpi.state` module.

        **Commonly used parameters**:

        * ``local_i`` — number of state elements this rank must compute
        * ``local_i_offset`` — starting global index for this rank
        * ``system_size`` — total number of basis states

        **Typical structure:**

        .. code-block:: python

            def initial_state_function(
                local_i : int,
                local_i_offset : int,
                system_size : int,
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

        **Parameters:**

        The first positional parameter always receives the free parameter vector
        from the optimiser. Additional parameters depend on your mapping logic:

        * ``ansatz_depth`` — number of ansatz iterations (for computing output size)
        * ``total_params`` — parameters per iteration (for computing output size)
        * ``observables`` — for normalising parameters by observable statistics
        * ``MPI_COMM`` — for computing global statistics across ranks

        **Typical structure:**

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
        a specific :class:`unitary` class. See :class:`quop_mpi.Unitary`.

        .. note::

           Operator Functions bind to **Unitary** attributes, not Ansatz
           attributes. See the "Unitary Bindable Attributes" table above.

        Predefined Operator Functions are included with each ``unitary`` class
        in the :mod:`quop_mpi.propagator` module under
        ``quop_mpi.propagator.<unitary>.operator``.

        **Commonly used parameters** (bound from Unitary):

        * ``local_i`` — partition size for this rank
        * ``local_i_offset`` — global index offset
        * ``system_size`` — total number of basis states
        * ``variational_parameters`` — only if the operator is parameterised

        **Typical Structure**

        .. code-block:: python
            
            def operator_function(
                local_i : int,
                local_i_offset : int,
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

        .. note::

           Parameter Functions bind to **Unitary** attributes, not Ansatz
           attributes. See the "Unitary Bindable Attributes" table above.

        Predefined Parameter Functions are included in the :mod:`quop_mpi.param`
        module.

        **Commonly used parameters** (bound from Unitary):

        * ``n_params`` — number of parameters to generate

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

        **Commonly used parameters**:

        * ``local_probabilities`` — probability amplitudes for this rank's partition
        * ``observables`` — observable values for this rank's partition
        * ``MPI_COMM`` — MPI subcommunicator (for global reductions, e.g., CVaR)

        **Typical Structure**

        .. code-block:: python

            def objective_function(
               local_probabilities: np.ndarray[np.float64],
               observables: np.ndarray[np.float64],
               MPI_COMM: MPI.Intracomm,
               *args,
               **kwargs
            ) -> float:

                ...

                return objective_function_value