"""Core Ansatz class for defining and simulating quantum variational algorithms."""

# cspell:words jacobian scipy nlopt BFGS gtol maxcomm wavefunction nfev subcomm dtype
from __future__ import annotations

import textwrap
from collections.abc import Callable
from copy import copy, deepcopy
from enum import IntFlag, auto
from importlib import import_module
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

from . import config
from ._benchmark import Benchmark
from ._communicator import Communicator
from ._lib.comm_info_wrapper import comm_info_wrapper as _ciw
from ._lib.context import Context
from ._logging import Logging
from ._optimization import Jacobian
from ._sampling import Sampling
from ._scope import scope
from ._utils._bindable import Bindable
from ._utils._comm_size import QuopMpiLayout
from ._utils._dump import dump_comm_info
from ._utils._interface import Interface
from ._utils._mpi import gather_array

if TYPE_CHECKING:
    from types import ModuleType, TracebackType

    from quop_mpi import UnitaryBase

    Intracomm = MPI.Intracomm


ParsedFunctionDict = dict[str, list[Any] | dict[str, Any]]

# -- Dirty-flag invalidation model ----------------------------------
# Every mutable subsystem has a corresponding _Dirty bit.  Setters OR
# the relevant flags (with cascades); setup() and __pre() process them
# in dependency order and clear each flag after completion.


class _Dirty(IntFlag):
    NONE = 0
    NEGOTIATION = auto()  # unitaries / system_size changed -> re-negotiate
    WORKER_SPLIT = auto()  # n_jacobian_workers changed -> re-split + negotiate
    CONTEXT = auto()  # layout changed -> rebuild context (GPU mem, state)
    PLANS = auto()  # context changed -> rebuild propagator plans
    OBSERVABLES = auto()  # observable function changed
    INITIAL_STATE = auto()  # initial state function changed
    DEPTH = auto()  # ansatz_depth changed (param arrays only)
    OPTIMISER = auto()  # optimiser configuration changed
    OBJECTIVE = auto()  # objective function changed
    SAMPLING = auto()  # sampling configuration changed
    LOG = auto()  # log configuration changed
    PARAM_MAP_BIND = auto()  # param map needs Interface binding after SUBCOMM creation


###################
# QuOp Ansatz Class
###################


# @MPI_trace
class Ansatz(Sampling, Logging, Communicator, Jacobian, Benchmark, Bindable):
    """Define and simulate a :term:`QVA`.

    Associated QuOp Functions:

    * :term:`Initial State Function` (:meth:`~quop_mpi.ansatz.set_initial_state`)
    * :term:`Observables Function` (:meth:`~quop_mpi.ansatz.set_observables`)
    * :term:`Parameter Map Function` (:meth:`~quop_mpi.ansatz.set_parameter_map`)
    * :term:`Jacobian Function` (:meth:`~quop_mpi.ansatz.set_parallel_jacobian`)
    * :term:`Sampling Function` (:meth:`~quop_mpi.ansatz.set_sampling`)

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
        1-D complex array of :literal:`local_i` values, the
        :term:`initial system state <initial state>`
    final_state : ndarray[complex128]
        1-D array of :literal:`local_i` elements, the :term:`system state` after
        computation of the state evolution under the action of an
        :term:`ansatz unitary`.
    last_evaluated : ndarray[float]
        1-D real array, the last :term:`variational parameters` passed to
        :meth:`~quop_mpi.ansatz.evolve_state`
    objective_cnt : int
        number of :term:`objective function` evaluations during :term:`QVA` simulation
    result : dict
        last result returned by the  :meth:`~quop_mpi.ansatz.execute` method
    seed : int
        seeds random number generation, incremented before each repeat in the
        :meth:`~quop_mpi.ansatz.benchmark` method
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

    def __init__(self, system_size: int, MPI_communicator: Intracomm = MPI.COMM_WORLD) -> None:  # noqa: N803
        """Initialise an Ansatz instance.

        Parameters
        ----------
        system_size : int
            Number of quantum basis states in the simulated system.
        MPI_communicator : Intracomm, optional
            MPI communicator, by default ``MPI.COMM_WORLD``.
        """

        self.system_size: int = system_size
        self.MPI_COMM_WORLD: Intracomm = MPI_communicator

        # -- Dirty-flag lifecycle state --------------------------
        # _dirty tracks what has changed since the last setup().
        # _setup_done is True after the first successful setup().
        self._dirty: _Dirty = (
            _Dirty.NEGOTIATION
            | _Dirty.CONTEXT
            | _Dirty.PLANS
            | _Dirty.OBSERVABLES
            | _Dirty.INITIAL_STATE
            | _Dirty.DEPTH
            | _Dirty.OPTIMISER
        )
        self._setup_done: bool = False
        self.setup_called: bool = False

        # variables that must be set by the 'pre' method of the child class
        self.local_observables: np.ndarray = np.empty(0, dtype=np.float64)
        self.local_probabilities: np.ndarray = np.empty(0, dtype=np.float64)
        self.observable_dict: ParsedFunctionDict | None = None
        self.observable_function: Callable[..., Any] | int | None = None
        self.variational_parameters: np.ndarray | None = None
        self.initial_state_dict: ParsedFunctionDict | None = None
        self.objective_dict: ParsedFunctionDict | None = None

        self.objective_function: Callable[..., Any] | Interface | None = None
        self._objective_function_raw: Callable[..., Any] | None = None
        self._initial_state_function_raw: Callable[..., Any] | None = None

        # can be set using methods in the system class
        # but default values are used if not set
        self.ansatz_depth: int = 1  # ansatz circuit depth
        self.total_params: int | None = None
        self.initial_state_type: object | None = None
        self.optimiser: Callable[..., Any] | None = None
        # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize

        # variables managed by the 'system' class
        self.stop: bool = False  # synchronise ranks during optimisation

        self.expectation: float | None = None  # expectation value of the system
        self.initial_state_input: object | None = None
        self.ansatz_initial_state: np.ndarray | None = None
        self.final_state: np.ndarray | None = None
        self.last_evaluated: np.ndarray = np.empty(
            0, dtype=np.float64
        )  # last set of variational parameters passed to 'evolve_state'.
        self.state_norm: float | None = None

        self.verbose_objective: bool = False
        self.objective_cnt: int = 0
        self.record_objective: bool = False
        self.objective_history: list[float] = []

        self.n_evolutions: int = 0
        self.total_n_evolutions: list[int] = []

        self.time_limit: float | None = None
        self.suspend_path: str | None = None
        self.available_time: float | None = None

        self.result: dict[str, Any] | None = None

        self.seed: int = 0

        # -- Attributes set later by setup / configuration methods --
        self.unitaries: list[UnitaryBase] | None = None
        self.param_map: np.ndarray | None = None
        self.backend: ModuleType | None = None
        self.context: Context | None = None
        self.initial_state_function: Callable[..., Any] | Interface | None = None
        self.n_variational_parameters: int | None = None
        self.optimiser_args: dict[str, Any] | None = None
        self.optimiser_log: list[str] | None = None
        self.parsed_observable_function: Interface | None = None
        self.time: float | None = None

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
        self.pre_execution_methods: list[Callable[[], None]] = []
        self.post_execution_methods: list[Callable[[], None]] = []
        self.quop_result: dict[str, Any] = {}

        self._has_param_map: bool = False  # flag
        self._param_map_raw: Callable[..., Any] = lambda x, *a, **k: x
        self._param_map_parsed: Interface | None = None
        self.param_map_dict: ParsedFunctionDict = {"args": [], "kwargs": {}}
        self._n_free_params: int | None = None  # set when param map is configured
        self.free_vec: np.ndarray | None = None  # bound by Interface for param map

        # -- Scope-nesting validation stack ----------------------
        self._scope_stack: list[tuple[int, str]] = []

    # -- Layout property (canonical partitioning source of truth) ----

    @property
    def layout(self) -> QuopMpiLayout | None:
        """The Fortran-backed ``QuopMpiLayout``, or ``None`` before setup."""
        return self._layout

    # -- Dirty-flag proxy properties --------------------------------
    # These expose individual _Dirty bits as boolean attributes so that
    # mixin code and tests can read/write them without importing _Dirty.

    @property
    def setup_objective(self) -> bool:
        return bool(self._dirty & _Dirty.OBJECTIVE)

    @setup_objective.setter
    def setup_objective(self, value: bool) -> None:
        if value:
            self._dirty |= _Dirty.OBJECTIVE
        else:
            self._dirty &= ~_Dirty.OBJECTIVE

    @property
    def setup_sampling(self) -> bool:
        return bool(self._dirty & _Dirty.SAMPLING)

    @setup_sampling.setter
    def setup_sampling(self, value: bool) -> None:
        if value:
            self._dirty |= _Dirty.SAMPLING
        else:
            self._dirty &= ~_Dirty.SAMPLING

    @property
    def setup_log(self) -> bool:
        return bool(self._dirty & _Dirty.LOG)

    @setup_log.setter
    def setup_log(self, value: bool) -> None:
        if value:
            self._dirty |= _Dirty.LOG
        else:
            self._dirty &= ~_Dirty.LOG

    @property
    def _need_bind_param_map(self) -> bool:
        return bool(self._dirty & _Dirty.PARAM_MAP_BIND)

    @_need_bind_param_map.setter
    def _need_bind_param_map(self, value: bool) -> None:
        if value:
            self._dirty |= _Dirty.PARAM_MAP_BIND
        else:
            self._dirty &= ~_Dirty.PARAM_MAP_BIND

    @scope("world")
    def set_parameter_map(
        self,
        n_free_params: int,
        mapping_fn: Callable[[np.ndarray], np.ndarray],
        mapping_dict: dict | None = None,
    ) -> None:
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
        self._dirty |= _Dirty.PARAM_MAP_BIND

    @scope("subcomm")
    def __to_full(self, vec: np.ndarray) -> np.ndarray:
        """Ensure vec is the full-length parameter vector.
        Applies the user mapping if necessary.
        """
        full_len = self.ansatz_depth * self.total_params
        vec = np.asarray(vec, dtype=np.float64)

        if not self._has_param_map:
            if vec.size != full_len:
                raise ValueError(
                    f"Expected {full_len} variational parameters"
                    f" (ansatz_depth={self.ansatz_depth},"
                    f" total_params={self.total_params}),"
                    f" got {vec.size}"
                )
            return vec

        # otherwise, map the parameters
        self.free_vec = vec
        self._param_map_parsed.update_parameters()
        full_vec = self._param_map_parsed.call(
            *self.param_map_dict["args"],
            **self.param_map_dict["kwargs"],
        )
        full_vec = np.asarray(full_vec, dtype=np.float64)
        if full_vec.size != full_len:
            raise ValueError(
                f"Parameter mapping returned {full_vec.size} parameters."
                f" Expected {full_len} variational parameters"
                f" (ansatz_depth={self.ansatz_depth},"
                f" total_params={self.total_params}),"
                f" got {vec.size}"
            )
        return full_vec

    def __parse_function_dict__(
        self, function_dict: dict[str, Any] | None, attribute_name: str
    ) -> None:
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
            :class:`~quop_mpi.ansatz` attribute to be set to a
            :literal:`ParsedFunctionDict` instance
        """

        function_dict = {} if function_dict is None else function_dict
        parsed_dict: ParsedFunctionDict = {"args": [], "kwargs": {}}

        for key in function_dict:
            if function_dict[key] is not None:
                parsed_dict[key] = function_dict[key]

        setattr(self, attribute_name, parsed_dict)

    @scope("world")
    def __pre(self) -> None:
        """Preparation for simulation of a QVA under the parallelisation
        scheme generated by :meth:`~quop_mpi.ansatz.setup`.

        Dirty-flag driven: only regenerates subsystems whose ``_Dirty``
        bits are set.
        """
        if self._dirty & _Dirty.DEPTH:
            self.__gen_depth()
            self._dirty &= ~_Dirty.DEPTH

        self._update_var_map()

        if self._dirty & _Dirty.OBSERVABLES:
            self.__gen_observables()
            self._dirty &= ~_Dirty.OBSERVABLES

        if self._dirty & _Dirty.PLANS:
            self.__gen_unitaries()
            self._dirty &= ~_Dirty.PLANS

        if self._dirty & _Dirty.INITIAL_STATE:
            self.__gen_initial_state()
            self._dirty &= ~_Dirty.INITIAL_STATE

        if self._dirty & _Dirty.OBJECTIVE:
            self.__gen_objective()
            self._dirty &= ~_Dirty.OBJECTIVE

        if self._dirty & _Dirty.OPTIMISER:
            self.__gen_optimiser()
            self._dirty &= ~_Dirty.OPTIMISER

        if self._dirty & _Dirty.SAMPLING:
            self._gen_sampling()
            self._dirty &= ~_Dirty.SAMPLING

        if self._dirty & _Dirty.LOG:
            self._gen_log()
            self._dirty &= ~_Dirty.LOG

        if self._dirty & _Dirty.PARAM_MAP_BIND:

            if self.subcomms.in_subcomm():

                self._param_map_parsed = Interface(
                    [self],
                    self._param_map_raw,
                    "parameter map",
                    self.subcomms.SUBCOMM,
                    call_args=self.param_map_dict["args"],
                    call_kwargs=self.param_map_dict["kwargs"],
                )

            self._dirty &= ~_Dirty.PARAM_MAP_BIND

        for method in self.pre_execution_methods:
            method()

    @scope("subcomm")
    def __populate_quop_result(self) -> None:
        """Populate fields of the :attr:`~quop_mpi.ansatz.quop_result` dictionary.

        Called by rank 0 in :attr:`~quop_mpi.ansatz.MPI_COMM_WORLD` only.
        """
        self.quop_result["fun"] = copy(self.result["fun"])
        self.quop_result["qubits"] = copy(np.log2(self.system_size))
        self.quop_result["system size"] = copy(self.system_size)
        self.quop_result["ansatz_depth"] = copy(self.ansatz_depth)
        self.quop_result["variational_parameters"] = deepcopy(self.result["x"])
        self.quop_result["mapped_parameters"] = deepcopy(self.__to_full(self.result["x"]))
        self.quop_result["final state norm"] = copy(self.state_norm)
        self.quop_result["execution time"] = copy(self.time)

        for key in self.result.keys():
            if key not in ["fun"]:
                self.quop_result[key] = copy(self.result[key])

    @scope("world")
    def __post(self) -> None:
        """Calls post-simulation methods."""

        if self.subcomms.get_subcomm_index() == 0:
            self.state_norm = self.__get_state_norm()

        if (
            self.subcomms.in_subcomm()
            and self.subcomms.get_subcomm_index() == 0
            and self.subcomms.SUBCOMM.Get_rank() == 0
            and self.result is not None
        ):
            self.__populate_quop_result()

        # Only ranks in the subcomm have a valid SUBCOMM to call barrier on
        if self.subcomms.in_subcomm():
            self.subcomms.SUBCOMM.barrier()
        self.variational_parameters = None

        for method in self.post_execution_methods:
            method()

    @scope("world")
    def set_unitaries(self, unitaries: list[UnitaryBase]) -> None:
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

        self.total_params = int(np.sum(self.param_map))
        self.param_map = np.cumsum(self.param_map)

        self._dirty |= _Dirty.NEGOTIATION | _Dirty.CONTEXT | _Dirty.PLANS

    @scope("world")
    def set_observables(
        self,
        function: Callable | int,
        observable_dict: dict | None = None,
    ) -> None:
        """Specify the :term:`observables`.

        Parameters
        ----------
        function : callable or int
            an :term:`Observables Function` or an integer specifying the index
            of a phase-shift unitary in the list passed to
            :meth:`~quop_mpi.ansatz.set_unitaries` whose exponent contains the
            observable vector.

        observable_dict : FunctionDict, optional
            :term:`FunctionDict` for the Observables Function
        """

        self.__parse_function_dict__(observable_dict, "observable_dict")

        self.observable_function = function

        self._dirty |= _Dirty.OBSERVABLES

    @scope("world")
    def set_optimiser(
        self,
        optimiser: str,
        optimiser_args: dict | None = None,
        optimiser_log: list[str] | None = None,
    ) -> None:
        """Define the classical :term:`optimiser` for :term:`QVA` simulation.

        Optionally allows for specification of arguments passed to the optimiser
        and fields in the optimiser dictionary to write to the log file (see
        :meth:`~quop_mpi.ansatz.set_log`). QuOp_MPI supports optimisers provided by SciPy
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

        self._dirty |= _Dirty.OPTIMISER

    # __parse_jacobian is inherited from Jacobian mixin

    @scope("world")
    def set_depth(self, depth: int) -> None:
        """Set the simulated :term:`ansatz depth`.

        Parameters
        ----------
        depth : int
            number of ansatz iterations
        """
        if depth != self.ansatz_depth:
            self.ansatz_depth = depth
            self._dirty |= _Dirty.DEPTH

    @scope("world")
    def set_initial_state(self, function: Callable, initial_state_dict: dict | None = None) -> None:
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
        self._initial_state_function_raw = function  # keep raw for re-wrap

        self._dirty |= _Dirty.INITIAL_STATE

    @scope("subcomm")
    def __parse_initial_state_function(self) -> None:
        """Map the arguments of a QuOp Initial State Function to the attributes
        of an :class:`~quop_mpi.ansatz` instance.
        """

        self.initial_state_function = Interface(
            [self, self.unitaries],
            self._initial_state_function_raw,
            "initial state",
            self.subcomms.SUBCOMM,
            call_args=self.initial_state_dict["args"],
            call_kwargs=self.initial_state_dict["kwargs"],
        )

    # Sampling methods (set_sampling, unset_sampling, etc.) are inherited from Sampling mixin
    # Logging methods (set_log, save, etc.) are inherited from Logging mixin

    # Bindable attributes for QuOp Functions - used for documentation and validation.
    # Subclasses can extend this by defining their own BINDABLE_ATTRIBUTES dict.
    BINDABLE_ATTRIBUTES: dict[str, str] = {
        # Core partitioning
        "system_size": "Total number of quantum basis states",
        "local_i": "Number of elements in this rank's partition",
        "local_i_offset": "Global index offset for this rank's partition",
        "partition_table": "Array describing global partitioning scheme",
        # Observables and state
        "local_observables": "Local partition of observable values (after setup)",
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

    @scope("world")
    def print_all_bindable_attributes(self) -> None:
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
        if self.unitaries is not None:
            for unitary in self.unitaries:
                unitary.print_bindable_attributes()
        else:
            print("(No unitaries set yet - call set_unitaries() first to see Unitary attributes)\n")

    @scope("world")
    def set_seed(self, seed: int) -> None:
        """Integer for seeding of random number generation.

        Parameters
        ----------
        seed : int
            seeds the generation of random parameters
        """
        self.seed = seed
    
    @scope("subcomm", returns="all")
    def get_state_norm(self) -> float:
        """Compute the norm of the wavefunction, should be 1 for a properly normalised state.

        Returns
        -------
        float
            objective function value, or None on excluded ranks
        """

        if self.subcomms.get_subcomm_index() == 0:
            state_norm = self.__get_state_norm()
        else:
            state_norm = None

        return self.subcomms.SUBCOMM.bcast(state_norm, root=0)


    @scope("subcomm", returns="all")
    def get_expectation_value(self) -> float:
        """Compute the :term:`objective function` at the current
        value of ``variational_parameters``.

        Returns
        -------
        float
            objective function value, or None on excluded ranks
        """

        if self.subcomms.get_subcomm_index() == 0:
            expectation_value = self.__get_expectation_value()
        else:
            expectation_value = None

        return self.subcomms.SUBCOMM.bcast(expectation_value, root=0)

    @scope("world")
    def set_objective(self, function: Callable, objective_dict: dict | None = None) -> None:
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
        self._objective_function_raw = function
        self.objective_function = function
        self._dirty |= _Dirty.OBJECTIVE

    @scope("subcomm")
    def __parse_objective(self) -> None:

        self.objective_function = Interface(
            [self, self.unitaries],
            self._objective_function_raw,
            "objective",
            self.subcomms.SUBCOMM,
            call_args=self.objective_dict["args"],
            call_kwargs=self.objective_dict["kwargs"],
        )

    @scope("subcomm")
    def __gen_objective(self) -> None:
        self.__parse_objective()

    @scope("subcomm", returns="root")
    def objective(self, variational_parameters: list[float] | np.ndarray[float]) -> float:
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

    @scope("world")
    def __check_comm_size(self) -> None:
        """Negotiate communicator size and partitioning via Fortran.

        Calls the Fortran negotiate() entry point which:
        - Phases 1-5: CREATE, NEGOTIATE loop, FINALISE, VALIDATE, LOCK
        - Propagators are queried via bind(C) callbacks during negotiation
        - Handles communicator shrinking if needed

        After negotiate, the QuopMpiLayout (self._layout) has its layout_ptr set
        and provides partition_table, local_i, local_i_offset, etc.

        Requires _gen_parallel() to have been called first (creates self._layout).
        """
        backend_flag = 1 if config.backend == "wavefront" else 0

        # Use split_ptr and topo_ptr from _layout (created by _gen_parallel)
        split_ptr = self._layout.split_ptr
        topo_ptr = self._layout.topo_ptr
        n_workers = self._layout.get_n_subcomms()

        # Store constraints on each propagator before negotiate
        for unitary in self.unitaries:
            for prop in unitary.propagators:
                prop.store_constraints(unitary.comm_size_constraints)

        # Collect propagator pointers and callback pointers
        prop_ptrs = []
        cb_ptrs = []
        for unitary in self.unitaries:
            for prop in unitary.propagators:
                prop_ptrs.append(prop.ptr)
                cb_ptrs.append(prop.negotiate_callback)

        prop_ptrs = np.array(prop_ptrs, dtype=np.int64)
        cb_ptrs = np.array(cb_ptrs, dtype=np.int64)

        # Call Fortran negotiate
        layout_ptr, status = _ciw.wrapper_negotiate(
            split_ptr,
            topo_ptr,
            np.int64(self.system_size),
            np.int32(backend_flag),
            prop_ptrs,
            cb_ptrs,
        )
        layout_ptr = int(layout_ptr)

        def _cleanup_failed_negotiate() -> None:
            if self._layout is None:
                return
            if layout_ptr and getattr(self._layout, "handle", None) in (None, 0):
                self._layout.set_layout_ptr(layout_ptr)
            self._layout.destroy()
            self._layout = None

        if status == -1:
            # Rank excluded during negotiate (communicator shrunk/filtered)
            # Still keep the Fortran layout handle so collectives like
            # dump_comm_info can run without deadlock and so we can free it.
            self._layout.set_layout_ptr(layout_ptr)
            # Mark this rank as excluded so in_subcomm() returns False
            self._layout.mark_excluded()
        elif status == 0:
            # Set the layout pointer (now the layout is fully initialized)
            self._layout.set_layout_ptr(layout_ptr)
        else:
            _cleanup_failed_negotiate()
            if status == 3:
                raise RuntimeError("Fortran negotiate failed to converge")
            if status == 4:
                raise RuntimeError("Fortran negotiate failed while finalizing communicator shrink")
            if status == 6:
                raise RuntimeError("Fortran negotiate failed while rebuilding device communicators")
            if status == 7:
                raise RuntimeError("Fortran negotiate failed while filtering zero-host-data ranks")
            if status == 5:
                raise RuntimeError(
                    "Fortran negotiate failed while computing initial block distribution"
                )
            if status >= 1000:
                raise RuntimeError(
                    "Fortran negotiate propagator callback failed with " f"status {status - 1000}"
                )
            if 100 <= status < 200:
                val_err = status - 100
                failed = [
                    name for flag, name in QuopMpiLayout._VALIDATE_FLAGS.items() if val_err & flag
                ]
                if not failed:
                    raise ValueError(
                        "Layout validation failed during negotiate: "
                        f"unknown validation code {val_err}"
                    )
                raise ValueError("Layout validation failed during negotiate: " + ", ".join(failed))
            if 200 <= status < 300:
                raise RuntimeError(
                    "Fortran negotiate failed while finalizing partition_table "
                    f"(status {status})"
                )
            if 300 <= status < 400:
                raise RuntimeError(
                    f"Fortran negotiate failed while locking layout (status {status})"
                )
            raise RuntimeError(f"Fortran negotiate returned error status {status}")

        # Recreate JACCOMM after negotiate (for parallel Jacobian)
        if n_workers > 1:
            _ciw.wrapper_create_jaccomm(self.MPI_COMM_WORLD.py2f(), split_ptr, layout_ptr)

            # Eagerly cache subcomm roots while all ranks are still
            # participating collectively.  get_subcomm_roots() uses
            # MPI_COMM_WORLD.Allgather, so it must be called before
            # the execute() loop diverges (where only JACCOMM members
            # would be executing _mpi_jacobian).
            self._layout.get_subcomm_roots()

        # Recreate ROOTCOMM after negotiate (connects subcomm leaders)
        _ciw.wrapper_create_rootcomm(self.MPI_COMM_WORLD.py2f(), split_ptr, layout_ptr)

    @property
    def n_free_params(self) -> int:
        """Number of free parameters presented to the optimizer.

        Without a parameter map, this equals n_variational_parameters.
        With a parameter map, this is the size of the reduced parameter vector.
        """
        if self._has_param_map and self._n_free_params is not None:
            return self._n_free_params
        if self.n_variational_parameters is None:
            raise RuntimeError("n_variational_parameters is not available before setup.")
        return self.n_variational_parameters

    # __update_var_map is inherited from Jacobian mixin

    # __gen_parallel is inherited from Jacobian mixin

    @scope("subcomm")
    def __gen_unitaries(self) -> None:
        """Calls methods associated with :literal:`Unitary` instances to determine the
        parallelisation scheme required for computation of the system dynamics.
        Generates operators associated with the :literal:`Unitary` instances.
        """
        for i, unitary in enumerate(self.unitaries):
            unitary._UnitaryBase__plan(self.system_size, self._layout)

            if unitary.operator_n_params == 0:
                unitary.gen_operator()

            unitary.seed = self.seed + i

    def __gen_depth(self) -> None:
        """Computes the total number of variational parameters at the current
        ansatz depth."""
        self.n_variational_parameters = self.total_params * self.ansatz_depth

    @scope("subcomm")
    def __gen_initial_state(self) -> None:
        """Generates the initial system state, defaults to a uniform
        superposition if not otherwise specified by the
        :meth:`~quop_mpi.ansatz.set_initial_state` method.
        """

        if self.initial_state_dict is None:
            from .state import equal

            # Inline the essential attribute assignments from
            # set_initial_state() to avoid calling a world-scoped method
            # from this subcomm-scoped context.
            self.__parse_function_dict__(None, "initial_state_dict")
            self.initial_state_function = equal
            self._initial_state_function_raw = equal

        self.__parse_initial_state_function()
        self.initial_state_function.update_parameters()

        self.ansatz_initial_state = np.asarray(
            self.initial_state_function.call(
                *self.initial_state_dict["args"], **self.initial_state_dict["kwargs"]
            ),
            dtype=np.complex128,
        )

    @scope("subcomm")
    def __gen_observables(self) -> None:
        """Generates the observables for computation of the QVA objective
        function."""

        if callable(self.observable_function):

            self.parsed_observable_function = Interface(
                [self],
                self.observable_function,
                "observable",
                self.subcomms.SUBCOMM,
                call_args=self.observable_dict["args"],
                call_kwargs=self.observable_dict["kwargs"],
            )
            self.parsed_observable_function.update_parameters()

            self.local_observables = np.asarray(
                self.parsed_observable_function.call(
                    *self.observable_dict["args"], **self.observable_dict["kwargs"]
                )
            )

            if self.local_observables.shape[0] != self.local_i:
                self.local_observables = np.reshape(self.local_observables, (self.local_i,))

        else:

            unitary = self.unitaries[self.observable_function]

            if unitary.unitary_type == "diagonal":
                self.local_observables = np.asarray(unitary.operator)
            else:
                raise RuntimeError(
                    f"Rank {self.subcomms.SUBCOMM.Get_rank()}:"
                    " Cannot identify observables,"
                    " no diagonal unitary defined"
                )

        self.context.observables = self.local_observables.astype(np.float64)

    @scope("subcomm")
    def __gen_optimiser(self) -> None:
        """Prepares the optimisation method using default or user-specified
        options with or without parallel computation of the objective
        function Jacobian.
        """

        if self.optimiser is None:
            # Inline the essential attribute assignments from
            # set_optimiser() to avoid calling a world-scoped method
            # from this subcomm-scoped context.
            from scipy.optimize import minimize as sp_minimize

            self.optimiser = sp_minimize
            self.optimiser_args = {"method": "BFGS", "options": {"gtol": 1e-3}}
            self.optimiser_log = ["fun", "nfev", "success"]

        # Configure parallel jacobian if requested (from Jacobian mixin)
        self._configure_parallel_jacobian()

    @scope("world")
    def __assign_backend(self) -> None:

        self.backend = import_module(f"quop_mpi._lib.{config.backend}")

        for unitary in self.unitaries:
            unitary.assign_backend(self.backend)

    @scope("subcomm")
    def __initialise_context(self) -> None:

        self.context = Context(
            self.backend,
            comm_info=self._layout,
        )

        self.MPI_COMM.barrier()

        for unitary in self.unitaries:
            unitary.context = self.context

    # -- Teardown helper (used by setup() on re-negotiate) ---------

    @scope("world")
    def _teardown_layout(self) -> None:
        """Free resources from a previous ``setup()`` to prepare for
        re-negotiation.

        Called internally when :pyattr:`_Dirty.NEGOTIATION` or
        :pyattr:`_Dirty.WORKER_SPLIT` fires on an already-set-up instance.
        """
        # Free unitary plans
        if self._layout is not None and self._layout.in_subcomm():
            for unitary in self.unitaries:
                if unitary.planned:
                    unitary.destroy()
                    unitary.planned = False

        # Free context BEFORE layout -- context_destroy uses borrowed SUBCOMM
        if self.context is not None:
            self.context.destroy()
            self.context = None

        # Destroy the layout (frees layout_ptr, split_ptr, topo_ptr)
        if self._layout is not None:
            self._layout.destroy()
            self._layout = None

    # -- Idempotent setup ----------------------------------------

    # Flags that setup() is responsible for -- others are handled by __pre().
    _SETUP_FLAGS = _Dirty.NEGOTIATION | _Dirty.WORKER_SPLIT | _Dirty.CONTEXT

    @scope("world")
    def setup(self) -> None:
        """Determine the parallelisation scheme and perform setup tasks
        required by extension modules.

        Idempotent: returns immediately when nothing that ``setup()``
        handles is dirty and a prior ``setup()`` has completed.  Dirty
        flags for items handled by ``__pre`` (plans, observables,
        initial state, depth, optimiser) do **not** trigger a re-setup.
        """
        if self._setup_done and not (self._dirty & self._SETUP_FLAGS):
            return

        # -- Full re-negotiate path ------------------------------
        if self._dirty & (_Dirty.NEGOTIATION | _Dirty.WORKER_SPLIT):

            # Tear down previous resources on re-setup
            if self._setup_done:
                self._teardown_layout()

            self.seed += 1
            self.__assign_backend()

            # Rebuild the worker split only when explicitly dirty
            if self.subcomms is None or (self._dirty & _Dirty.WORKER_SPLIT):
                self._gen_parallel()

            self.__check_comm_size()

            # Diagnostic dump: layout created, communicators set, pre-context
            if self._layout is not None:
                dump_comm_info(self._layout, "init")

            self.__initialise_context()

            # Diagnostic dump: fully locked with context
            if self._layout is not None:
                dump_comm_info(self._layout, "locked")

            # Clear processed flags and cascade to children
            self._dirty &= ~(_Dirty.WORKER_SPLIT | _Dirty.NEGOTIATION | _Dirty.CONTEXT)
            self._dirty |= (
                _Dirty.PLANS
                | _Dirty.OBSERVABLES
                | _Dirty.INITIAL_STATE
                | _Dirty.DEPTH
                | _Dirty.OPTIMISER
            )

            self.setup_called = True

        # -- Context-only re-init path ---------------------------
        elif self._dirty & _Dirty.CONTEXT:
            self.__initialise_context()
            self._dirty &= ~_Dirty.CONTEXT
            # Cascade: plans depend on context
            self._dirty |= _Dirty.PLANS

        self._setup_done = True

    @scope("world")
    def prepare(self) -> None:
        """Fully initialize the Ansatz for inspection without running optimization.

        This method runs both ``setup()`` and internal preparation steps,
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
        >>> print(f"Observables range: {alg.local_observables.min():.2f} to {alg.local_observables.max():.2f}")

        Related methods: ``setup()`` for low-level setup and ``execute()`` to
        run optimization.
        """
        self.setup()
        self._Ansatz__pre()

    @scope("subcomm")
    def __post_unitaries(self) -> None:
        """Free memory managed by extension modules on simulation completion."""
        for unitary in self.unitaries:
            if unitary.planned:
                unitary.destroy()
                unitary.planned = False

    # __post_parallel is inherited from Jacobian mixin

    def __enter__(self) -> Ansatz:
        """Return ``self`` so :class:`Ansatz` can be used as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Deterministically release MPI/native resources at scope exit."""
        self.destroy()
        return False

    def __del__(self) -> None:
        """Destructor intentionally avoids MPI/native teardown.

        ``Ansatz`` owns collective MPI resources and backend-native state.
        Releasing those from Python finalizers is not reliable, so callers
        must use :meth:`destroy` explicitly or rely on the context-manager
        path via ``with``.
        """
        return

    @scope("world")
    def destroy(self) -> None:
        """Free all resources owned by this :class:`~quop_mpi.ansatz`.

        Releases negotiated layout state, backend plans, and context buffers
        while leaving the caller-owned world communicator untouched. The
        instance may be set up again after destruction.

        All ranks must call this collectively.
        """
        if self._setup_done and not self.benchmarking and self.log:
            self._post_log()

        if self.unitaries is not None:
            for unitary in self.unitaries:
                if getattr(unitary, "planned", False):
                    unitary.destroy()
                    unitary.planned = False

        # Free context BEFORE layout -- context_destroy uses borrowed SUBCOMM
        if self.context is not None:
            self.context.destroy()
            self.context = None

        # Free subcommunicators and layout
        if getattr(self, "_layout", None) is not None:
            self._post_parallel()

        # Mark everything dirty so a subsequent setup()+execute() fully re-initializes.
        self._setup_done = False
        self.setup_called = False
        self._dirty = (
            _Dirty.NEGOTIATION
            | _Dirty.CONTEXT
            | _Dirty.PLANS
            | _Dirty.OBSERVABLES
            | _Dirty.INITIAL_STATE
            | _Dirty.DEPTH
            | _Dirty.OPTIMISER
        )
        if self._objective_function_raw is not None:
            self._dirty |= _Dirty.OBJECTIVE

    @scope("world")
    def evolve_state(self, variational_parameters: list[float] | np.ndarray[float]) -> None:
        """Compute the :term:`system state` under the action of the
        :term:`ansatz unitary`.

        See Also
        --------
        :meth:`~quop_mpi.ansatz.set_unitaries`

        Parameters
        ----------
        variational_parameters : list[float] or ndarray[float]
            1-D :literal:`(ansatz_depth * total_params,)` real array of
            :term:`variational parameters`.
        """

        self.setup()
        self.__pre()

        self.__evolve_state(variational_parameters)

        self.__post()

    @scope("subcomm")
    def __evolve_state(self, x: list[float] | np.ndarray[float]) -> None:
        """Compute the system state given input variational parameters `x`.

        Parameters
        ----------
        x : {list[float], ndarray[float]}
            1-D :literal:`(ansatz_depth * total_params,)` real array of variational
            parameters
        """

        if isinstance(x, list):
            x = np.array(x, dtype=np.float64)

        x = self.__to_full(x)  # apply parameter mapping if present
        self.context.state = self.ansatz_initial_state.astype(np.complex128)
        params_split = np.split(x, self.ansatz_depth)

        for params in params_split:

            for i, unitary in enumerate(self.unitaries):
                param_slice = params[self.param_map[i] : self.param_map[i + 1]]

                if unitary.operator_n_params > 0:
                    evolution_parameter = param_slice[: -unitary.operator_n_params]

                    unitary.variational_parameters = param_slice[unitary.unitary_n_params : :]

                    unitary.gen_operator()

                    if isinstance(self.observable_function, int) and i == self.observable_function:
                        self.local_observables = np.asarray(unitary.operator)

                else:
                    evolution_parameter = param_slice

                unitary.propagate(evolution_parameter)

        if self.subcomms.SUBCOMM.Get_rank() == 0:
            self.n_evolutions += 1
        self.final_state = self.context.state
        self.last_evaluated = copy(x)

    @scope("subcomm", returns="all")
    def evaluate(self, variational_parameters: list[float] | np.ndarray[float]) -> float:
        """Lazily computes the :term:`objective function` value.

        The :class:`~quop_mpi.ansatz` instance stores the last :term:`variational
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
            objective function value, or None on excluded ranks
        """

        if not np.array_equal(self.last_evaluated, variational_parameters):
            self.__evolve_state(variational_parameters)
        expectation_value = self.__get_expectation_value()

        return self.subcomms.SUBCOMM.bcast(expectation_value, root=0)

    @scope("world")
    def execute(
        self,
        variational_parameters: list[float] | np.ndarray[float] | None = None,
    ) -> None:
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

            self.setup()

            self.__pre()

            # Early check: parameter map requires explicit initial parameters
            if variational_parameters is None and self._has_param_map:
                raise ValueError(
                    "Parameter map function is set, initial parameters must be supplied to execute."
                )

            # Broadcast parameters over SUBCOMM only (excluded ranks skip)
            if self.subcomms.in_subcomm():
                broadcast_parameters = self.subcomms.SUBCOMM.bcast(
                    variational_parameters, root=0
                )
                self.variational_parameters = (
                    None
                    if broadcast_parameters is None
                    else np.asarray(broadcast_parameters, dtype=np.float64)
                )
            else:
                self.variational_parameters = None

            if self.subcomms.in_subcomm() and self.variational_parameters is None:
                self.variational_parameters = self.gen_initial_params(self.ansatz_depth)

        if self.subcomms.in_subcomm():

            self._parallel_jacobian_control_active = (
                self.parallel_jacobian_enabled and self.subcomms.get_n_subcomms() > 1
            )
            self.stop = False
            self.n_evolutions = 0

            try:
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
                    if self._parallel_jacobian_control_active:
                        while not self.stop:
                            command = self._await_parallel_jacobian_command()
                            if command is None:
                                raise RuntimeError(
                                    "Worker subcommunicator received no parallel jacobian command."
                                )
                            if command not in (
                                self.PARALLEL_JAC_COMMAND_EVALUATE,
                                self.PARALLEL_JAC_COMMAND_STOP,
                            ):
                                raise RuntimeError(
                                    f"Unknown parallel jacobian command {command}."
                                )
                            self._mpi_jacobian(None)
                    else:
                        while not self.stop:
                            self._mpi_jacobian(None)

                    self.__post()

                # Broadcast results to all active ranks via SUBCOMM
                self.result = self.subcomms.SUBCOMM.bcast(self.result, root=0)
                self.quop_result = self.subcomms.SUBCOMM.bcast(self.quop_result, root=0)
                self.state_norm = self.subcomms.SUBCOMM.bcast(self.state_norm, root=0)
            finally:
                self._parallel_jacobian_control_active = False
        else:
            # Excluded ranks: nothing to do, mark as stopped
            self.stop = True

    @scope("subcomm")
    def __execute_subcomm_group_zero(self) -> None:
        """Tasks carried out at :attr:`~quop_mpi.ansatz.subcomms` group zero during simulation
        of a QVA via a call to :meth:`~quop_mpi.ansatz.execute`."""
        if self.record_objective:
            self.total_n_evolutions = []

        self.neval_mpi_jac = 0

        self.time = time()

        try:
            self.result = self.optimiser(
                self.__objective, self.variational_parameters, **self.optimiser_args
            )
        except Exception:
            self.stop = True
            try:
                self.__objective(None)
                if self._parallel_jacobian_control_active:
                    self._mpi_jacobian(None)
            finally:
                self.time = time() - self.time
            raise

        self.stop = True

        self.__objective(None)

        if self._parallel_jacobian_control_active:
            self._mpi_jacobian(None)

        self.time = time() - self.time

    @scope("world")
    def print_result(self) -> None:
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

    @scope("world")
    def print_optimiser_result(self) -> None:
        """Print the result returned from the :term:`optimiser` for the last
        :term:`QVA` simulation."""
        if self.MPI_COMM_WORLD.Get_rank() == 0:
            print("\nOptimisation Result", flush=True)
            print("===================\n", flush=True)
            print(self.result, flush=True)

    # benchmark method is inherited from Benchmark mixin

    @scope("subcomm", returns="root")
    def get_final_state(self) -> np.ndarray[np.complex128] | None:
        """Gather the :term:`final state` to rank 0 of the :literal:`Ansatz` MPI subcommunicator.

        Requires a previous call to :meth:`~quop_mpi.ansatz.execute`,
        :meth:`~quop_mpi.ansatz.evolve_state` or
        :meth:`~quop_mpi.ansatz.benchmark`. If called after
        :meth:`~quop_mpi.ansatz.benchmark` the gathered state will
        correspond to the last performed simulation.

        Returns
        -------
        ndarray[complex128] or None
            the final state at rank 0 of the :literal:`Ansatz`
            subcommunicator, :literal:`None` otherwise
        """

        if self.subcomms.get_subcomm_index() == 0:
            return gather_array(
                self.context.state,
                self.unitaries[0].partition_table,
                self.subcomms.SUBCOMM,
            )

    @scope("subcomm", returns="root")
    def get_probabilities(self) -> np.ndarray[np.float64] | None:
        """Gather probabilities computed from the :term:`final state` at rank 0
        of the :literal:`Ansatz` MPI subcommunicator.

        Requires a previous call to :meth:`~quop_mpi.ansatz.execute`,
        :meth:`~quop_mpi.ansatz.evolve_state` or :meth:`~quop_mpi.ansatz.benchmark`. If called after
        :meth:`~quop_mpi.ansatz.benchmark` the gathered state will correspond to the last
        performed simulation.

        Returns
        -------
        ndarray[float64] or None
            1-D real array of state probabilities at rank 0 of the :literal:`Ansatz`
            subcommunicator, :literal:`None` otherwise
        """

        if self.subcomms.get_subcomm_index() == 0:
            return gather_array(
                np.abs(self.context.state) ** 2,
                self.unitaries[0].partition_table,
                self.subcomms.SUBCOMM,
            )

    # save method is inherited from Logging mixin

    @scope("subcomm", returns="all")
    def gen_initial_params(self, ansatz_depth: int = None) -> np.ndarray[np.float64]:
        """Generate initial :term:`variational parameters`.

        Values are generated using the :term:`Parameter Function` associated
        with each :literal:`unitary` passed to the :meth:`~quop_mpi.ansatz.set_unitaries`
        method.

        .. note::
            If :literal:`ansatz_depth` is :literal:`None` the :term:`ansatz depth` defaults
            to `1` or the depth specified by the :meth:`~quop_mpi.ansatz.set_depth` method.

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

        # __gen_initial_params already bcasts over SUBCOMM.
        # For multi-subcomm (parallel jacobian), params are generated
        # on subcomm 0 and distributed via JACCOMM during optimisation.
        return params

    @scope("subcomm")
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

    def _root_world(self) -> int:
        """Return world-rank of SUBCOMM root (0 if subcomms unset)."""
        if self.subcomms is None:
            return 0
        return self.subcomms.get_root_world()

    @scope("subcomm")
    def _get_local_probabilities(self) -> np.ndarray[np.float64]:
        """Compute the probabilities of states local to each MPI process.

        Returns
        -------
        ndarray[float64]
            1-D array containing :meth:`~quop_mpi.ansatz.local_i` state probabilities with
            global index offset :meth:`~quop_mpi.ansatz.local_i_offset`
        """
        self.local_probabilities = (np.abs(self.context.state[: self.local_i]) ** 2).astype(
            np.float64
        )
        return self.local_probabilities

    @scope("subcomm")
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

    @scope("subcomm")
    def __get_expectation_value(self) -> float:
        """Compute the expectation value at the current ``variational_parameters``.

        Returns
        -------
        float
            expectation value at the current ``variational_parameters``
        """

        if self.sampling:
            return self._sample_expectation_value()

        self._get_local_probabilities()

        local_expectation = np.dot(self.local_probabilities, self.local_observables)

        return np.real(self.subcomms.SUBCOMM.allreduce(local_expectation, op=MPI.SUM))

    @scope("subcomm")
    def __objective(self, variational_parameters: list[float] | np.ndarray[float]) -> float | None:
        """Compute the objective function at `variational_parameters`.

        Parameters
        ----------
        variational_parameters : {list[float], ndarray[float]}
            1-D real array of variational parameters

        Returns
        -------
        float or None
            returns the objective function value at rank 0 in
            :attr:`~quop_mpi.ansatz.MPI_COMM_WORLD`, None otherwise
        """
        self.stop = self.subcomms.SUBCOMM.bcast(self.stop, root=0)

        if not self.stop:

            broadcast_parameters = self.subcomms.SUBCOMM.bcast(
                variational_parameters, root=0
            )
            self.variational_parameters = np.asarray(broadcast_parameters, dtype=np.float64)

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
