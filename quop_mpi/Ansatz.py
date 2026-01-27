# cspell:words jacobian scipy nlopt BFGS gtol maxcomm wavefunction nfev subcomm dtype
from __future__ import annotations
from importlib import import_module
import os
import csv
import textwrap
import atexit
from copy import copy, deepcopy
from time import time
import numpy as np
from mpi4py import MPI
from . import config
from ._utils._interface import interface
from ._utils._mpi import gather_array
from ._utils._filenames import ensure_path_and_extension
from ._utils._tracker import job_tracker
import inspect

from ._utils._comm_size import vector_partitioning, max_compatible_size
from ._lib.context import context
from ._sampling import Sampling
from ._logging import Logging
from ._communicator import Communicator
from ._optimization import Jacobian, forward_differences, central
from ._benchmark import Benchmark
from ._utils._bindable import Bindable

##########################################
# Collect profiling data if QUOP_PROFILE=1
##########################################

from ._profile import profiler

####################################
# imports and classes for type hints
####################################

from quop_mpi import Unitary
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm
iterable = Iterable


###################
# QuOp Ansatz Class
###################


# @MPI_trace
class Ansatz(Sampling, Logging, Communicator, Jacobian, Benchmark, Bindable):
    """Define and simulate a :term:`QVA`.

    Associated QuOp Functions:

    * :term:`Initial State Function` (:meth:`~quop_mpi.Ansatz.set_initial_state`)
    * :term:`Observables Function` (:meth:`~quop_mpi.Ansatz.set_observables`)
    * :term:`Parameter Map Function` (:meth:`~quop_mpi.Ansatz.set_parameter_map`)
    * :term:`Jacobian Function` (:meth:`~quop_mpi.Ansatz.set_parallel_jacobian`)
    * :term:`Sampling Function` (:meth:`~quop_mpi.Ansatz.set_sampling`)

    Examples
    --------
    Minimal definition of an arbitrary :term:`QVA`, of size :term:`system size`.
    Where :literal:`[UQ, UW]` defines the :term:`ansatz unitary` and
    :literal:`observable_function` is an :term:`Observables Function`.

    .. code-block :: python

        alg = Ansatz(system_size)
        alg.set_unitaries([UQ, UW])
        alg.set_observables(observable_function)

    Attributes
    ----------
    system_size : int
        The size of the :term:`simulated quantum system <QVA>`.
    local_i : int
        parallel partition size of :term:`system state` and :term:`observables`
    local_i_offset : int
        global index offset of the local parallel partition
    partition_table : ndarray[int]
        1-D integer array describing the global partitioning scheme such that
        for a given MPI rank :literal:`partition_table[rank + 1] - partition_table[rank] = local_i`
    observables : ndarray[float64]
        1-D real array of :literal:`local_i` :term:`observables`
    variational_parameters : ndarray[float64]
        1-D real array of :term:`variational parameters`, updated during
        :term:`optimisation <optimiser>`
    ansatz_depth : int
        number of :term:`ansatz iterations <ansatz depth>`, by default :literal:`1`
    total_params : int
        number of :term:`variational parameters` associated with each
        :term:`ansatz iteration <ansatz depth>`
    expectation : float
        last computed :term:`objective function` value, updated during
        :term:`optimisation <optimiser>`
    ansatz_initial_state : ndarray[complex128]
        1-D complex array of :literal:`local_i` values, the :term:`initial system state <initial state>`
    final_state : ndarray[complex128]
        1-D array of :literal:`local_i` elements, the :term:`system state` after
        computation of the state evolution under the action of an
        :term:`ansatz unitary`.
    last_evaluated : ndarray[float]
        1-D real array, the last :term:`variational parameters` passed to
        :meth:`~quop_mpi.Ansatz.evolve_state`
    objective_cnt : int
        number of :term:`objective function` evaluations during :term:`QVA` simulation
    result : dict
        last result returned by the  :meth:`~quop_mpi.Ansatz.execute` method
    seed : int
        seeds random number generation, incremented before each repeat in the
        :meth:`~quop_mpi.Ansatz.benchmark` method
    sample_indexes : list[ndarray[int32]]
        if simulating sampling, contains the global indexes for each block of
        sampled :term:`observables`, resets to :literal:`[]` when the
        :term:`objective function` value is accepted
    samples : list[ndarray[float64]]
        if simulating sampling, contains the observable value for each block of
        sampled :term:`observables`, resets to :literal:`[]` when the
        :term:`objective function` value is accepted
    sample_minimum_indexes : list[int]
        if simulating sampling, contains the index of the minimum
        :term:`observable <observables>` sampled for each computation
        of the :term:`objective function`

    Parameters
    ----------
    system_size : int
        number of quantum basis states in the simulated system
    MPI_communicator : Intracomm, optional
        MPI Intracomm, by default MPI.COMM_WORLD
    """

    def __init__(self, system_size: int, MPI_communicator: Intracomm = MPI.COMM_WORLD):

        self.system_size = system_size
        self.MPI_COMM_WORLD = MPI_communicator.Dup()

        # variables that must be set by the 'pre' method of the child class
        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None
        self.partition_table = False
        self.observables = None
        self.observable_dict = None
        self.observable_function = None
        self.variational_parameters = None
        self.initial_state_dict = None
        self.objective_dict = None

        self.objective_function = None

        # can be set using methods in the system class
        # but default values are used if not set
        self.ansatz_depth = 1  # ansatz circuit depth
        self.total_params = None
        self.initial_state_type = None
        self.optimiser = (
            None  # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize
        )

        # variables managed by the 'system' class
        self.stop = False  # synchronise ranks during optimisation

        self.expectation = None  # expectation value of the system
        self.initial_state_input = None
        self.ansatz_initial_state = None  # initial state before algorithm evolution
        self.final_state = None  # quantum state during and after simulation
        self.last_evaluated = np.empty(
            0
        )  # last set of variational parameters passed to 'evolve_state'.

        self.setup_called = False
        self.destroy_called = False

        self.verbose_objective = False
        self.objective_cnt = 0
        self.record_objective = False
        self.objective_history = []

        self.n_evolutions = 0
        self.total_n_evolutions = []

        self.setup_depth = True
        self.setup_unitaries = True
        self.setup_observables = True
        self.setup_initial_state = True
        self.setup_optimiser = True
        self.setup_objective = False

        self.time_limit = None
        self.suspend_path = None
        self.available_time = None

        self.result = None

        self.seed = 0

        # Initialize sampling subsystem
        self._init_sampling()
        # Initialize logging subsystem
        self._init_logging()
        # Initialize communicator subsystem (MPI subcommunicators)
        self._init_communicator()
        # Initialize jacobian subsystem (optional parallel gradient)
        self._init_jacobian()
        # Initialize benchmark subsystem
        self._init_benchmark()
        self.pre_execution_methods = []
        self.post_execution_methods = []
        self.quop_result = {}
        self.setup_var_map = True
        self.setup_called = False
        self.reset = False

        self._has_param_map = False  # flag
        self._param_map_raw = lambda x, *a, **k: x  # identity fallback
        self._param_map_parsed = None  # interface-wrapped fn
        self.param_map_dict = {"args": [], "kwargs": {}}
        self._need_bind_param_map = False  # postpone binding until SUBCOMM exists
        self._n_free_params = None  # set when param map is configured

        atexit.register(self.__exit)

    def set_parameter_map(
        self,
        n_free_params: int,
        mapping_fn: Callable[[np.ndarray], np.ndarray],
        mapping_dict: dict | None = None,
    ):
        """Register a mapping from a subset of optimisable parameters to the full
        set of variational parameters.

        Parameters
        ----------
        n_free_params : int
            The number of free parameters in the reduced parameter vector.
            This is the dimensionality of the optimization problem.
        mapping_fn : callable
            ``mapping_fn(free_vec, *args, **kwargs) -> full_vec``.
            *free_vec* is the vector presented to the optimiser;
            *full_vec* must have length ``ansatz_depth * total_params``.
        mapping_dict : FunctionDict, optional
            FunctionDict supplying extra positional and keyword arguments
            to the mapping function.
        """

        self._has_param_map = True
        self._param_map_raw = mapping_fn
        self._n_free_params = n_free_params
        self.__parse_function_dict__(mapping_dict, "param_map_dict")
        self._need_bind_param_map = True

    def __to_full(self, vec: np.ndarray) -> np.ndarray:
        """Ensure vec is the full-length parameter vector.
        Applies the user mapping if necessary.
        """
        full_len = self.ansatz_depth * self.total_params
        vec = np.asarray(vec, dtype=np.float64)

        if not self._has_param_map:
            if vec.size != full_len:
                raise ValueError(
                    f"Expected {full_len} variational parameters (ansatz_depth={self.ansatz_depth}, total_params={self.total_params}), got {vec.size}"
                )
            return vec

        # otherwise, map the parameters
        self._param_map_parsed.update_parameters()
        full_vec = self._param_map_parsed.call(
            vec,
            *self.param_map_dict["args"],
            **self.param_map_dict["kwargs"],
        )
        full_vec = np.asarray(full_vec, dtype=np.float64)
        if full_vec.size != full_len:
            raise ValueError(
                f"Parameter mapping returned {full_vec.size} parameters. Expected {full_len} variational parameters (ansatz_depth={self.ansatz_depth}, total_params={self.total_params}), got {vec.size}"
            )
        return full_vec

    def __exit(self):
        """Called on program exit or on destruction of an :class:`~quop_mpi.Ansatz` instance.
        Frees :class:`~quop_mpi.Ansatz` -created MPI :literal:`Intracomm` instances and memory allocations
        managed by extension modules.
        """
        if self.setup_called:
            self.destroy()
        self.MPI_COMM_WORLD.barrier()

    def __parse_function_dict__(self, function_dict: dict, attribute_name: str):
        """Takes a user specified :literal:`FunctionDict` and sets :literal:`attribute_name`
        to a :literal:`ParsedFunctionDict` containing the values associated with the
        "args" and "kwargs" keys of the input :literal:`FunctionDict`. If either of these
        keys are not present, or if `function_dict` is :literal:`None`, the resulting
        `ParsedFunctionDict` will contain the key-values pairs :literal:`'args':[]` and
        :literal:`'kwargs':{}` respectively.


        Parameters
        ----------
        function_dict : dict or None
            a QuOp :term:`FunctionDict`
        attribute_name : str
            :class:`~quop_mpi.Ansatz` attribute to be set to a :literal:`ParsedFunctionDict` instance
        """

        function_dict = {} if function_dict is None else function_dict
        parsed_dict = {"args": [], "kwargs": {}}

        for key in function_dict:
            if function_dict[key] is not None:
                parsed_dict[key] = function_dict[key]

        setattr(self, attribute_name, parsed_dict)

    def __pre(self):
        """Preparation for simulation of a QVA under the parallelisation
        scheme generated by :meth:`~quop_mpi.Ansatz.setup`.
        """
        if self.setup_depth:
            self.__gen_depth()
            self.setup_depth = False

        self._update_var_map()

        if self.setup_observables:
            self.__gen_observables()
            self.setup_observables = False

        if self.setup_unitaries:
            self.__gen_unitaries()
            self.setup_unitaries = False

        if self.setup_initial_state:
            self.__gen_initial_state()
            self.setup_initial_state = False

        if self.setup_objective:
            self.__gen_objective()
            self.setup_objective = False

        if self.setup_optimiser:
            self.__gen_optimiser()
            self.setup_optimiser = False

        if self.setup_sampling:
            self._gen_sampling()
            self.setup_sampling = False

        if self.setup_log:
            self._gen_log()
            self.setup_log = False

        if self._need_bind_param_map:
            self._param_map_parsed = interface(
                [self],
                self._param_map_raw,
                "parameter map",
                self.subcomms.SUBCOMM,
            )
            self._need_bind_param_map = False

        for method in self.pre_execution_methods:
            method()

    def __populate_quop_result(self):
        """Populate fields of the :attr:`~quop_mpi.Ansatz.quop_result` dictionary.

        Called by rank 0 in :attr:`~quop_mpi.Ansatz.MPI_COMM_WORLD` only.
        """
        self.quop_result["fun"] = copy(self.result["fun"])
        self.quop_result["qubits"] = copy(np.log2(self.system_size))
        self.quop_result["system size"] = copy(self.system_size)
        self.quop_result["ansatz_depth"] = copy(self.ansatz_depth)
        self.quop_result["variational_parameters"] = deepcopy(self.result["x"])
        self.quop_result["mapped_parameters"] = deepcopy(
            self.__to_full(self.result["x"])
        )
        self.quop_result["final state norm"] = copy(self.state_norm)
        self.quop_result["execution time"] = copy(self.time)

        for key in self.result.keys():
            if key not in ["fun"]:
                self.quop_result[key] = copy(self.result[key])

    def __post(self):
        """Calls post-simulation methods."""

        if self.subcomms.get_subcomm_index() == 0:
            self.state_norm = self.__get_state_norm()

        if (self.MPI_COMM_WORLD.Get_rank() == 0) and (self.result is not None):
            self.__populate_quop_result()

        # Only ranks in the subcomm have a valid SUBCOMM to call barrier on
        if self.subcomms.in_subcomm():
            self.subcomms.SUBCOMM.barrier()
        self.variational_parameters = None

        for method in self.post_execution_methods:
            method()

    def set_unitaries(self, unitaries: list[Unitary]):
        """Define the :term:`ansatz unitary`.

         :term:`Unitaries<unitary>` are passed as a python list in order of
         application from left to right.

        Parameters
        ----------
        unitaries: list[unitary]
            list of :term:`unitaries<unitary>` specifying the action of one
            :term:`ansatz iteration <ansatz depth>`
        """

        self.unitaries = unitaries

        self.param_map = np.zeros(len(self.unitaries) + 1, int)

        for i, unitary in enumerate(self.unitaries):
            self.param_map[i + 1] = unitary.n_params

        self.total_params = np.sum(self.param_map)
        self.param_map = np.cumsum(self.param_map)

        self.reset = True

    def set_observables(
        self,
        function: Union[Callable, int],
        observable_dict: dict = None,
    ):
        """Specify the :term:`observables`.

        Parameters
        ----------
        function : callable or int
            an :term:`Observables Function` or an integer specifying the index
            of a phase-shift unitary in the list passed to
            :meth:`~quop_mpi.Ansatz.set_unitaries` whose exponent contains the
            observable vector.

        observable_dict : FunctionDict, optional
            :term:`FunctionDict` for the Observables Function
        """

        self.__parse_function_dict__(observable_dict, "observable_dict")

        self.observable_function = function

        self.setup_observables = True

    def set_optimiser(
        self,
        optimiser: str,
        optimiser_args: dict = None,
        optimiser_log: list[str] = None,
    ):
        """Define the classical :term:`optimiser` for :term:`QVA` simulation.

        Optionally allows for specification of arguments passed to the optimiser
        and fields in the optimiser dictionary to write to the log file (see
        :meth:`~quop_mpi.Ansatz.set_log`). QuOp_MPI supports optimisers provided by SciPy
        through its minimize method `minimize
        <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
        and optimisers provided by the `NLopt
        <http://nlopt.readthedocs.io/en/latest/>`_ package with respect to
        minimisation with scalar constraints through a SciPy-like interface.

        Parameters
        ----------

        optimiser: {'scipy', 'nlopt'}
            'scipy' to use the SciPy, 'nlopt' to use NLopt, or a callable
            QuOp_MPI-compatible optimisation function.
        optimiser_args: dict
            arguments to pass to the optimiser
        optimiser_log: list[str]
            results of the optimisation process are stored in a dictionary.
            These values may be logged by passing a list of the corresponding
            keys

        Examples
        --------

        The default optimiser is the BFGS algorithm, which is set internally as
        follows:

        .. code-block:: python

            Ansatz.set_optimiser( 'scipy',
                    {'method':'BFGS','options':{'gtol':1e-3}},
                                ['fun','nfev','success'])

        """
        if optimiser_args is None:
            optimiser_args = {}

        if optimiser == "scipy":
            from scipy.optimize import minimize as sp_minimize

            self.optimiser = sp_minimize
        elif optimiser == "nlopt":
            from quop_mpi._optimization.nlopt_wrap import minimize as nlopt_minimize

            self.optimiser = nlopt_minimize
        elif callable(optimiser):
            self.optimiser = optimiser

        self.optimiser_args = optimiser_args
        self.optimiser_log = optimiser_log

        self.setup_optimiser = True

    # __parse_jacobian is inherited from Jacobian mixin

    def set_depth(self, depth: int):
        """Set the simulated :term:`ansatz depth`.

        Parameters
        ----------
        depth : int
            number of ansatz iterations
        """
        if depth != self.ansatz_depth:
            self.ansatz_depth = depth
            self.setup_depth = True

    def set_initial_state(self, function: Callable, initial_state_dict: dict = None):
        """Define the :term:`initial state`.

        Parameters
        ----------
        function : callable
            :term:`Initial State Function`
        initial_state_dict : FunctionDict, optional
            :term:`FunctionDict` for the Initial State Function
        """

        self.__parse_function_dict__(initial_state_dict, "initial_state_dict")
        self.initial_state_function = function

        self.setup_initial_state = True

    def __parse_initial_state_function(self):
        """Map the arguments of a QuOp Initial State Function to the attributes
        of an :class:`~quop_mpi.Ansatz` instance.
        """

        self.initial_state_function = interface(
            [self, self.unitaries],
            self.initial_state_function,
            "initial state",
            self.subcomms.SUBCOMM,
        )

    # Sampling methods (set_sampling, unset_sampling, etc.) are inherited from Sampling mixin
    # Logging methods (set_log, save, etc.) are inherited from Logging mixin

    # Bindable attributes for QuOp Functions - used for documentation and validation.
    # Subclasses can extend this by defining their own BINDABLE_ATTRIBUTES dict.
    BINDABLE_ATTRIBUTES = {
        # Core partitioning
        "system_size": "Total number of quantum basis states",
        "local_i": "Number of elements in this rank's partition",
        "local_i_offset": "Global index offset for this rank's partition",
        "partition_table": "Array describing global partitioning scheme",
        # Observables and state
        "observables": "Local partition of observable values (after setup)",
        "ansatz_initial_state": "Local partition of initial state vector",
        "final_state": "Local partition of current/final state vector",
        # Variational parameters
        "variational_parameters": "Current variational parameter values",
        "ansatz_depth": "Number of ansatz iterations (layers)",
        "total_params": "Number of variational parameters per iteration",
        # MPI
        "MPI_COMM": "MPI subcommunicator for this Ansatz instance",
        # Execution state
        "expectation": "Last computed objective function value",
        "seed": "Random seed for parameter generation",
    }

    def print_all_bindable_attributes(self):
        """Print bindable attributes for this Ansatz AND all its Unitaries.

        This shows the complete picture of what parameters can be bound in
        QuOp Functions:

        - Ansatz-level functions (Observables, Initial State, Parameter Map,
          Sampling, Objective) bind to Ansatz attributes
        - Unitary-level functions (Operator, Parameter) bind to Unitary attributes

        Call this after :meth:`set_unitaries` to see Unitary attributes.

        See Also
        --------
        print_bindable_attributes : Ansatz attributes only
        """
        # Print Ansatz attributes
        self.print_bindable_attributes()

        # Print Unitary attributes if unitaries have been set
        if hasattr(self, "unitaries") and self.unitaries:
            for i, unitary in enumerate(self.unitaries):
                unitary.print_bindable_attributes()
        else:
            print(
                "(No unitaries set yet - call set_unitaries() first to see Unitary attributes)\n"
            )

    def set_seed(self, seed: int):
        """Integer for seeding of random number generation.

        Parameters
        ----------
        seed : int
            seeds the generation of random parameters
        """
        self.seed = seed

    def get_expectation_value(self) -> float:
        """Compute the :term:`objective function` at the current
        value of :attr:`~quop_mpi.Ansatz.variational_parameters`.

        Returns
        -------
        float
            objective function value
        """

        if self.subcomms.get_subcomm_index() == 0:
            return self.__get_expectation_value()

    def set_objective(self, function: Callable, objective_dict: dict = None):
        """Set a custom objective function (i.e. an objective function other
        than the expectation value of the prepared state).

        The function is called after state evolution - returning a scalar
        value that is passed to the minimizer.

        Parameters
        ----------
        function: callable
            an :term:`Objective Function`

        objective_dict: FunctionDict, optional
            :term:`FunctionDict` for the `Objective Function`
        """
        self.__parse_function_dict__(objective_dict, "objective_dict")
        self.objective_function = function
        self.setup_objective = True

    def __parse_objective(self):
        self.objective_function = interface(
            [self, self.unitaries],
            self.objective_function,
            "objective",
            self.subcomms.SUBCOMM,
        )

    def __gen_objective(self):
        self.__parse_objective()

    def objective(
        self, variational_parameters: Union[list[float], np.ndarray[float]]
    ) -> float:
        """Compute the :term:`objective function` at :term:`variational parameters`
        :literal:`variational_parameters`.

        Parameters
        ----------
        variational_parameters : list or ndarray[float]

        Returns
        -------
        float
            objective function value
        """
        if self.subcomms.get_subcomm_index() == 0:
            return self.__objective(variational_parameters)

    # set_parallel_jacobian is inherited from Jacobian mixin

    def __check_comm_size(self):
        """Ensure that all MPI ranks have been assigned at least :literal:`local_i = 1`
        elements of the distributed state vector. All MPI ranks with
        :literal:`local_i = 0` are dropped from Ansatz subcommunicators.
        """

        busy_comm = False

        if self.subcomms.in_subcomm():

            if self.system_size // self.subcomms.SUBCOMM.Get_size() == 0:
                newsize = self.system_size // 2
            else:
                newsize = 0
        else:
            newsize = 0

        newsize = self.MPI_COMM_WORLD.allreduce(newsize, op=MPI.MAX)

        if newsize > 0:

            self.subcomms.shrink_subcomms(self.subcomms.SUBCOMM.Get_size() - newsize)

        while not busy_comm:

            if self.subcomms.in_subcomm():

                max_comm_size = max_compatible_size(
                    self.unitaries,
                    self.system_size,
                    self.subcomms.SUBCOMM.size,
                    self.subcomms.SUBCOMM.py2f(),
                )
                dropcount = self.subcomms.SUBCOMM.size - max_comm_size

            else:
                break

            if dropcount > 0:
                self.subcomms.shrink_subcomms(dropcount)
            else:
                busy_comm = True

        if self.subcomms.in_subcomm():
            # create the default vector partitioning, may be altered during the unitary planning phase.
            (
                self.local_i,
                self.local_i_offset,
                self.alloc_local,
                self.partition_table,
            ) = vector_partitioning(self.system_size, self.subcomms.SUBCOMM)
            # Update MPI_COMM to the (possibly shrunken) subcomm
            self.MPI_COMM = self.subcomms.SUBCOMM

    @property
    def n_free_params(self):
        """Number of free parameters presented to the optimizer.

        Without a parameter map, this equals n_variational_parameters.
        With a parameter map, this is the size of the reduced parameter vector.
        """
        if self._has_param_map and self._n_free_params is not None:
            return self._n_free_params
        return self.n_variational_parameters

    # __update_var_map is inherited from Jacobian mixin

    # __gen_parallel is inherited from Jacobian mixin

    def __gen_unitaries(self):
        """Calls methods associated with :literal:`Unitary` instances to determine the
        parallelisation scheme required for computation of the system dynamics.
        Generates operators associated with the :literal:`Unitary` instances.
        """
        if self.subcomms.in_subcomm():
            for i, unitary in enumerate(self.unitaries):
                unitary._Unitary__plan(self.system_size, self.subcomms.SUBCOMM)
                unitary.parse_plan([self.local_i, self.alloc_local])

                if unitary.operator_n_params == 0:
                    unitary.gen_operator()

                unitary.seed = self.seed + i

    def __gen_depth(self):
        """Computes the total number of variational parameters at the current
        ansatz depth."""
        self.n_variational_parameters = self.total_params * self.ansatz_depth

    def __gen_initial_state(self):
        """Generates the initial system state, defaults to a uniform
        superposition if not otherwise specified by the
        :meth:`~quop_mpi.Ansatz.set_initial_state` method.
        """

        if self.subcomms.in_subcomm():

            if self.initial_state_dict is None:
                from .state import equal

                self.set_initial_state(equal)

            self.__parse_initial_state_function()

            self.ansatz_initial_state = self.initial_state_function.call(
                *self.initial_state_dict["args"], **self.initial_state_dict["kwargs"]
            )

    def __gen_observables(self):
        """Generates the observables for computation of the QVA objective
        function."""

        if not self.subcomms.in_subcomm():
            return

        if callable(self.observable_function):

            self.parsed_observable_function = interface(
                [self], self.observable_function, "observable", self.subcomms.SUBCOMM
            )

            self.observables = self.parsed_observable_function.call(
                *self.observable_dict["args"], **self.observable_dict["kwargs"]
            )

            if self.observables.shape[0] != self.local_i:
                self.observables = np.reshape(self.observables, (self.local_i,))

        else:

            unitary = self.unitaries[self.observable_function]

            if unitary.unitary_type == "diagonal":
                self.observables = unitary.operator
            else:
                raise RuntimeError(
                    f"Rank {self.subcomms.SUBCOMM.Get_rank()}: Cannot identify observables, no diagonal unitary defined"
                )

        self.context.observables = self.observables.astype(np.float64)

    def __gen_optimiser(self):
        """Prepares the optimisation method using default or user-specified
        options with or without parallel computation of the objective
        function Jacobian.
        """
        if self.subcomms.in_subcomm():

            if self.optimiser is None:
                self.set_optimiser(
                    "scipy",
                    {"method": "BFGS", "options": {"gtol": 1e-3}},
                    ["fun", "nfev", "success"],
                )

            # Configure parallel jacobian if requested (from Jacobian mixin)
            self._configure_parallel_jacobian()

    def __assign_backend(self):

        self.backend = import_module(f"quop_mpi._lib.{config.backend}")

        for unitary in self.unitaries:
            unitary.assign_backend(self.backend)

    def __initialise_context(self):

        if self.subcomms.in_subcomm():

            self.context = context(
                self.backend,
                self.system_size,
                self.alloc_local,
                self.local_i,
                self.local_i_offset,
                self.subcomms.SUBCOMM,
            )

            self.subcomms.SUBCOMM.barrier()

            for unitary in self.unitaries:
                unitary.context = self.context

    def setup(self):
        """Determine the parallelisation scheme and performs setup tasks
        required by extension modules.
        """
        if self.reset and not self.setup_called:
            self.seed += 1

            # TODO trigger setup on changes to config.backend
            self.__assign_backend()

            self._gen_parallel()
            self.setup_parallel = False  # Indicate parallel resources need cleanup

            self.__check_comm_size()

            self.__initialise_context()

            self.setup_depth = True
            self.setup_observables = True
            self.setup_initial_state = True
            self.setup_optimiser = True

            self.reset = False
            self.setup_called = True

    def prepare(self):
        """Fully initialize the Ansatz for inspection without running optimization.

        This method runs both :meth:`setup` and internal preparation steps,
        bringing the Ansatz to its runtime state. After calling this method:

        - All Unitary instances have their attributes populated
        - Observables, initial state, and operators are generated
        - :meth:`print_all_bindable_attributes` shows actual runtime values
        - :meth:`get_expectation_value` can be called

        This is useful for:

        - Debugging QuOp Functions before optimization
        - Inspecting the parallel partitioning scheme
        - Querying bindable attributes with their runtime values
        - Testing observables and initial state functions

        Examples
        --------
        >>> alg = qwoa(1024)
        >>> alg.set_qualities(my_observables)
        >>> alg.prepare()  # Fully initialize
        >>> alg.print_all_bindable_attributes()  # Now shows actual values
        >>> print(f"Observables range: {alg.observables.min():.2f} to {alg.observables.max():.2f}")

        See Also
        --------
        setup : Lower-level setup (parallel resources only)
        execute : Run optimization
        """
        self.setup()
        self._Ansatz__pre()

    def __post_unitaries(self):
        """Free memory managed by extension modules on simulation completion."""
        if self.subcomms.in_subcomm():
            for unitary in self.unitaries:
                if unitary.planned:
                    unitary.destroy()

    # __post_parallel is inherited from Jacobian mixin

    def __del__(self):
        """Destructor to ensure proper cleanup when the object is deleted.

        Called automatically when `del` is used on the object or when the
        object goes out of scope. Ensures all MPI resources and extension
        module memory are properly freed.
        """
        try:
            # Force cleanup regardless of lifecycle state
            if hasattr(self, "setup_called") and self.setup_called:
                # Close log file if open
                if hasattr(self, "log") and self.log and not self.benchmarking:
                    if hasattr(self, "logfile") and self.logfile is not None:
                        try:
                            self.logfile.close()
                        except:
                            pass

                # Free unitary resources
                if hasattr(self, "unitaries") and hasattr(self, "subcomms"):
                    if self.subcomms.in_subcomm():
                        for unitary in self.unitaries:
                            if hasattr(unitary, "planned") and unitary.planned:
                                try:
                                    unitary.destroy()
                                except:
                                    pass

                # Free subcommunicators
                if hasattr(self, "subcomms"):
                    try:
                        self.subcomms.free()
                    except:
                        pass

            # Free the duplicated communicator
            if hasattr(self, "MPI_COMM_WORLD") and self.MPI_COMM_WORLD is not None:
                try:
                    self.MPI_COMM_WORLD.Free()
                except:
                    pass
        except:
            # Suppress any exceptions during destruction
            pass

    def destroy(self):
        """Call methods to close the results log file, free memory managed by
        extension modules and free MPI subcommunicators created by the
        :class:`~quop_mpi.Ansatz` instance.
        """

        # Skip cleanup if:
        # - reset=False (no config change) - resources are still valid
        # - setup_called=False (never set up) - nothing to clean up
        if not self.reset or not self.setup_called:
            return

        if not self.benchmarking and self.log:
            self._post_log()

        if not self.setup_unitaries:
            self.__post_unitaries()
            self.setup_unitaries = True

        if not self.setup_parallel:
            self._post_parallel()
            self.setup_parallel = True

    def evolve_state(
        self, variational_parameters: Union[list[float], np.ndarray[float]]
    ):
        """Compute the :term:`system state` under the action of the
        :term:`ansatz unitary`.

        See Also
        --------
        :meth:`~quop_mpi.Ansatz.set_unitaries`

        Parameters
        ----------
        variational_parameters : list[float] or ndarray[float]
            1-D :literal:`(ansatz_depth * total_params,)` real array of
            :term:`variational parameters`.
        """

        self.destroy()
        self.setup()
        self.__pre()

        self.__evolve_state(variational_parameters)

        self.__post()

    def __evolve_state(self, x: Union[list[float], np.ndarray[float]]):
        """Compute the system state given input variational parameters `x`.

        Parameters
        ----------
        x : {list[float], ndarray[float]}
            1-D :literal:`(ansatz_depth * total_params,)` real array of variational
            parameters
        """

        if isinstance(x, list):
            x = np.array(x, dtype=np.float64)

        if self.subcomms.in_subcomm():

            x = self.__to_full(x)  # apply parameter mapping if present

            self.context.state = self.ansatz_initial_state.astype(np.complex128)
            params_split = np.split(x, self.ansatz_depth)

            for params in params_split:

                for i, unitary in enumerate(self.unitaries):

                    param_slice = params[self.param_map[i] : self.param_map[i + 1]]

                    if unitary.operator_n_params > 0:

                        evolution_parameter = param_slice[: -unitary.operator_n_params]

                        unitary.variational_parameters = param_slice[
                            unitary.unitary_n_params : :
                        ]

                        unitary.gen_operator()

                        if (
                            isinstance(self.observable_function, int)
                            and i == self.observable_function
                        ):
                            self.observables = unitary.operator

                    else:
                        evolution_parameter = param_slice

                    unitary.propagate(evolution_parameter)

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                self.n_evolutions += 1
            self.last_evaluated = copy(x)

    def evaluate(
        self, variational_parameters: Union[list[float], np.ndarray[float]]
    ) -> float:
        """Lazily computes the :term:`objective function` value.

        The :class:`~quop_mpi.Ansatz` instance stores the last :term:`variational
        parameters` passed to :literal:`evaluate` and the corresponding objective
        function value. If the input variational parameters match,
        re-computation of the :term:`final state` is skipped and the previously
        computed objective function value is returned.

        Parameters
        ----------
        variational_parameters : list[float] or ndarray[float]
            1-D :literal:`(ansatz_depth * total_params,)` real array of variational
            parameters

        Returns
        -------
        float
            objective function value
        """

        if not np.array_equal(self.last_evaluated, variational_parameters):
            self.__evolve_state(variational_parameters)
        return self.__get_expectation_value()

    def execute(
        self, variational_parameters: Union[list[float], np.ndarray[float]] = None
    ):
        """Simulate a :term:`QVA`.

        If :literal:`variational_parameters` is :literal:`None`, initial parameter values are
        generated using the :term:`Parameter Function` of the corresponding
        :literal:`unitary` instances.

        Parameters
        ----------
        variational_parameters : list[float] or ndarray[float]
            1-D :literal:`(ansatz_depth * total_params,)` real array of
            :term:`variational parameters`
        """

        if not self.benchmarking:

            self.destroy()
            self.setup()

            self.__pre()

            self.variational_parameters = self.MPI_COMM_WORLD.bcast(
                variational_parameters, root=0
            )

            if self.variational_parameters is None:
                if self._has_param_map:
                    raise ValueError(
                        "Parameter map function is set, initial parameters must be supplied to execute."
                    )
                else:
                    self.variational_parameters = self.gen_initial_params(
                        self.ansatz_depth
                    )

        if self.subcomms.in_subcomm():

            self.stop = False
            self.n_evolutions = 0

            if self.subcomms.get_subcomm_index() == 0:

                self.objective_cnt = 0

                if self.subcomms.SUBCOMM.Get_rank() == 0:

                    self.__execute_subcomm_group_zero()
                else:

                    while not self.stop:
                        self.__objective(None)

                self.__post()

                if self.log:
                    self._log_update()

            else:
                while not self.stop:
                    self._mpi_jacobian(None)

                self.__post()

    def __execute_subcomm_group_zero(self):
        """Tasks carried out at :attr:`~quop_mpi.Ansatz.subcomms` group zero during simulation
        of a QVA via a call to :meth:`~quop_mpi.Ansatz.execute`."""
        if self.record_objective:
            self.total_n_evolutions = []

        self.neval_mpi_jac = 0

        self.time = time()

        self.result = self.optimiser(
            self.__objective, self.variational_parameters, **self.optimiser_args
        )

        self.stop = True

        self.__objective(None)

        if self.subcomms.get_n_subcomms() > 1:
            self._mpi_jacobian(None)

        self.time = time() - self.time

    def print_result(self):
        """Print a summary of the results of the last :term:`QVA` simulation."""

        if self.MPI_COMM_WORLD.Get_rank() != 0:
            return

        print("\nQuOp_MPI Simulation Summary", flush=True)
        print("===========================\n", flush=True)
        for i, key in enumerate(self.quop_result.keys()):
            printkey = f"{key}:"
            if i == 8:
                print("\nOptimiser Output")
                print("----------------", flush=True)
            print(
                *textwrap.wrap(
                    f"{printkey:24}{self.quop_result[key]}",
                    subsequent_indent=f"\n{' ':24}",
                    width=80,
                )
            )
        print("")

    def print_optimiser_result(self):
        """Print the result returned from the :term:`optimiser` for the last
        :term:`QVA` simulation."""
        if self.MPI_COMM_WORLD.Get_rank() == 0:
            print("\nOptimisation Result", flush=True)
            print("===================\n", flush=True)
            print(self.result, flush=True)

    # benchmark method is inherited from Benchmark mixin

    def get_final_state(self) -> Union[np.ndarray[np.complex128], None]:
        """Gather the :term:`final state` to rank 0 of the :literal:`Ansatz` MPI subcommunicator.

        Requires a previous call to :meth:`~quop_mpi.Ansatz.execute`, :meth:`~quop_mpi.Ansatz.evolve_state`
        or :meth:`~quop_mpi.Ansatz.benchmark`. If called after :meth:`~quop_mpi.Ansatz.benchmark` the
        gathered state will correspond to the last performed simulation.

        Returns
        -------
        ndarray[complex128] or None
            the final state at rank 0 of the :literal:`Ansatz` subcommunicator, :literal:`None` otherwise
        """

        if self.subcomms.in_subcomm() and self.subcomms.get_subcomm_index() == 0:
            return gather_array(
                self.context.state,
                self.unitaries[0].partition_table,
                self.subcomms.SUBCOMM,
            )

    def get_probabilities(self) -> Union[np.ndarray[np.float64], None]:
        """Gather probabilities computed from the :term:`final state` at rank 0
        of the :literal:`Ansatz` MPI subcommunicator.

        Requires a previous call to :meth:`~quop_mpi.Ansatz.execute`,
        :meth:`~quop_mpi.Ansatz.evolve_state` or :meth:`~quop_mpi.Ansatz.benchmark`. If called after
        :meth:`~quop_mpi.Ansatz.benchmark` the gathered state will correspond to the last
        performed simulation.

        Returns
        -------
        ndarray[float64] or None
            1-D real array of state probabilities at rank 0 of the :literal:`Ansatz`
            subcommunicator, :literal:`None` otherwise
        """

        if self.subcomms.in_subcomm() and self.subcomms.get_subcomm_index() == 0:
            return gather_array(
                np.abs(self.context.state) ** 2,
                self.unitaries[0].partition_table,
                self.subcomms.SUBCOMM,
            )

    # save method is inherited from Logging mixin

    def gen_initial_params(self, ansatz_depth: int = None) -> np.ndarray[np.float64]:
        """Generate initial :term:`variational parameters`.

        Values are generated using the :term:`Parameter Function` associated
        with each :literal:`unitary` passed to the :meth:`~quop_mpi.Ansatz.set_unitaries`
        method.

        .. note::
            If :literal:`ansatz_depth` is :literal:`None` the :term:`ansatz depth` defaults
            to `1` or the depth specified by the :meth:`~quop_mpi.Ansatz.set_depth` method.

        Parameters
        ----------
        ansatz_depth : int, optional
            number of :term:`ansatz iterations<ansatz depth>`

        Returns
        -------
        ndarray[float64]
            1-D :literal:`(ansatz_depth * total_params,)` real array of variational
            parameters
        """

        if ansatz_depth is None:
            params = self.__gen_initial_params()
        else:
            params = self.__gen_initial_params(ansatz_depth)

        n_params = len(params) if self.MPI_COMM_WORLD.Get_rank() == 0 else None
        n_params = self.MPI_COMM_WORLD.bcast(n_params, 0)

        if self.subcomms.colour != 0:
            params = np.empty(n_params, dtype=np.float64)

        self.MPI_COMM_WORLD.Bcast([np.array(params, dtype=np.float64), MPI.DOUBLE], 0)

        return params

    def __gen_initial_params(self, ansatz_depth: int = None) -> np.ndarray[np.float64]:
        """Generates and returns initial ansatz variational parameters.

        Parameters
        ----------
        ansatz_depth : int or None
            number of ansatz iterations
        Returns
        -------
        ndarray[float64]
            1-D :literal:`(ansatz_depth * total_params,)` real array of variational
            parameters
        """

        if self.subcomms.get_subcomm_index() != 0:
            return

        if ansatz_depth is None:
            ansatz_depth = self.ansatz_depth

        params = np.zeros(ansatz_depth * self.total_params, dtype=np.float64)

        param_iterations = np.split(params, ansatz_depth)

        for param_iters in param_iterations:
            for i, unitary in enumerate(self.unitaries):
                unitary.seed += i + 1
                param_iters[self.param_map[i] : self.param_map[i + 1]] = (
                    unitary.gen_initial_params()
                )

        self.subcomms.SUBCOMM.Bcast([params, MPI.DOUBLE], 0)

        return params

    def _get_local_probabilities(self) -> np.ndarray[np.float64]:
        """Compute the probabilities of states local to each MPI process.

        Returns
        -------
        ndarray[float64]
            1-D array containing :meth:`~quop_mpi.Ansatz.local_i` state probabilities with
            global index offset :meth:`~quop_mpi.Ansatz.local_i_offset`
        """
        self.local_probabilities = (
            np.abs(self.context.state[: self.local_i]) ** 2
        ).astype(np.float64)
        return self.local_probabilities

    def __get_state_norm(self) -> float:
        """Compute norm of the wavefunction state vector.

        Returns
        -------
        float
            norm of the wavefunction state vector
        """
        if self.subcomms.get_subcomm_index() == 0:
            self.state_norm = self.context.get_state_norm()
            return self.state_norm

    def __get_expectation_value(self) -> float:
        """Compute the expectation value at :meth:`~quop_mpi.Ansatz.variational_parameters`.

        Returns
        -------
        float
            expectation value at :meth:`~quop_mpi.Ansatz.variational_parameters`
        """

        if self.sampling:
            return self._sample_expectation_value()

        self._get_local_probabilities()

        local_expectation = np.dot(self.local_probabilities, self.observables)

        return np.real(self.subcomms.SUBCOMM.allreduce(local_expectation, op=MPI.SUM))

    def __objective(
        self, variational_parameters: Union[list[float], np.ndarray[float]]
    ) -> Union[float, None]:
        """Compute the objective function at `variational_parameters`.

        Parameters
        ----------
        variational_parameters : {list[float], ndarray[float]}
            1-D real array of variational parameters

        Returns
        -------
        float or None
            returns the objective function value at rank 0 in
            :attr:`~quop_mpi.Ansatz.MPI_COMM_WORLD`, None otherwise
        """
        self.stop = self.subcomms.SUBCOMM.bcast(self.stop, root=0)

        if not self.stop:

            self.variational_parameters = self.subcomms.SUBCOMM.bcast(
                variational_parameters, root=0
            )

            self.__evolve_state(self.variational_parameters)

            if self.objective_function is not None:
                self._get_local_probabilities()
                self.objective_function.update_parameters()
                self.expectation = self.objective_function.call(
                    *self.objective_dict["args"], **self.objective_dict["kwargs"]
                )
            else:
                self.expectation = self.get_expectation_value()

            if self.subcomms.SUBCOMM.Get_rank() == 0:

                if self.verbose_objective:

                    self.objective_cnt += 1

                    print(
                        f"Call # {self.objective_cnt}, f(x) = {self.expectation}",
                        flush=True,
                    )

                if self.record_objective:
                    expectation = deepcopy(self.expectation)
                    self.objective_history.append(expectation)

                if self.record_objective:
                    self.total_n_evolutions.append(self.n_evolutions)
                return self.expectation
