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
from .__utils.__interface import interface
from .__utils.__mpi import gather_array, subcomms
from .__utils.__filenames import ensure_path_and_extension
from .__utils.__tracker import job_tracker
import inspect

from .__utils.comm_size import vector_partitioning, max_compatible_size
from .__lib.context import context

####################################
# imports and classes for type hints
####################################

from quop_mpi import Unitary
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm
iterable = Iterable

########################################
# Functions and decorating for debugging
########################################


def print_method(func):
    def decorator(*args, **kwargs):
        with open(f"trace_{MPI.COMM_WORLD.Get_rank()}.txt", "a") as f:
            f.write(f"{MPI.COMM_WORLD.Get_rank()}, {func.__name__}\n")
            f.flush()
        return func(*args, **kwargs)

    return decorator


def MPI_trace(cls):
    for name, method in inspect.getmembers(cls):
        if (
            not inspect.ismethod(method) and not inspect.isfunction(method)
        ) or inspect.isbuiltin(method):
            continue
        setattr(cls, name, print_method(method))
    return cls


################################################################
# Numerical Approximations of the objective function derivatives
################################################################


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
        :meth:`~quop_mpi.Ansatz.evaluate`)
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
        :meth:`~quop_mpi.Ansatz.evaluate`)
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
    expectation = evaluate(variational_parameters)
    x_back = copy(variational_parameters)
    x_forward = copy(variational_parameters)
    x_back[var] -= h
    x_forward[var] += h
    expectation_back = evaluate(x_back)
    expectation_forward = evaluate(x_forward)
    return (expectation_forward - expectation_back) / (2 * h)


###################
# QuOp Ansatz Class
###################

#@MPI_trace
class Ansatz:
    """Define and simulate a :term:`QVA`.

    Associated QuOp Functions:

    * :term:`Initial State Function` (:meth:`~quop_mpi.Ansatz.set_initial_state`)
    * :term:`Observables Function` (:meth:`~quop_mpi.Ansatz.set_observables`)
    * :term:`Free Parameters Function` (:meth:`~quop_mpi.Ansatz.set_free_params`)
    * :term:`Jacobian Function` (:meth:`~quop_mpi.Ansatz.set_parallel_jacobian`)
    * :term:`Sampling Function` (:meth:`~quop_mpi.Ansatz.set_sampling`)

    Examples
    --------
    Minimal definition of an arbitrary :term:`QVA`, of size :term:`system size`.
    Where :literal:`[UQ, UW]` defines the :term:`ansatz unitary` and
    :literal:`observable_function` is an :term:`Observables Function`.

    .. code-block :: python

        alg = Ansatz(system_size) alg.set_unitaries([UQ, UW])
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
        self.MPI_COMM_WORLD = MPI.Comm.Dup(MPI_communicator)

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

        # can be set using methods in the system class
        # but default values are used if not set
        self.ansatz_depth = 1  # ansatz circuit depth
        self.total_params = None
        self.initial_state_type = None
        self.optimiser = (
            None  # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize
        )
        self.optimiser = (
            None  # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize
        )

        self.setup_log = False  # whether results will be recorded in a *.log file.

        # variables managed by the 'system' class self.stop = False  # synchronise ranks during optimisation

        self.expectation = None  # expectation value of the system
        self.initial_state_input = None
        self.ansatz_initial_state = None  # initial state before algorithm evolution
        self.final_state = None  # quantum state during and after simulation
        self.jacobian_input = None  # for parallel jacobian evaluation
        self.var = None  # for parallel jacobian evaluation
        self.benchmarking = False  # indicates whether the benchmark method is running
        self.last_evaluated = np.empty(
            0
        )  # last set of variational parameters passed to 'evolve_state'.

        self.setup_called = False
        self.destroy_called = False

        self.jac_ranks = None

        self.verbose_objective = False
        self.objective_cnt = 0
        self.record_objective = False

        self.n_evolutions = 0
        self.total_n_evolutions = []

        self.log = False

        # arguments for subcomms class initialisation
        self.nodes_per_subcomm = None
        self.processes_per_node = None
        self.maxcomm = None

        self.setup_depth = True
        self.setup_parallel = True
        self.setup_unitaries = True
        self.setup_observables = True
        self.setup_initial_state = True
        self.setup_log = False
        self.setup_optimiser = True

        self.time_limit = None
        self.suspend_path = None
        self.available_time = None

        self.result = None

        self.seed = 0

        self.sampling_dict = {}
        self.sample_indexes = []
        self.samples = []
        self.sample_minimum_indexes = []
        self.variational_parameter_history = []

        # sampling variables
        self.samples = None
        self.sample_block_size = None
        self.max_sample_iterations = None
        self.sampling_function = None
        self.sampling_function_input = None
        self.sampling = False
        self.global_minimum = None
        self.minimum_sampled = np.inf
        self.shots_to_global_minimum = "not found"
        self.global_minimum_found = False
        self.total_shots = 0
        self.setup_sampling = False
        self.filename = None
        self.pre_execution_methods = []
        self.post_execution_methods = []
        self.quop_result = {}
        self.setup_var_map = True
        self.setup_called = False
        self.var_map = None
        self.reset = False

        self.free_params = None
        self.free_params_function = None
        self.free_params_dict = None



        atexit.register(self.__exit)

    def __exit(self):
        """Called on program exit or on destruction of an :class:`~quop_mpi.Ansatz` instance.
        Frees :class:`~quop_mpi.Ansatz` -created MPI :literal:`Intracomm` instances and memory allocations
        managed by extension modules.
        """
        if self.setup_called:
            self.destroy()
        self.MPI_COMM_WORLD.barrier()

    def __parse_function_dict__(
        self, function_dict: dict, attribute_name: str
    ):
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

        parsed_dict = getattr(self, attribute_name)

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

        self.__update_free_params()
        self.__update_var_map()

        if self.setup_observables:
            self.__gen_observables()
            self.setup_observables = False

        if self.setup_unitaries:
            self.__gen_unitaries()
            self.setup_unitaries = False

        if self.setup_initial_state:
            self.__gen_initial_state()
            self.setup_initial_state = False

        if self.setup_optimiser:
            self.__gen_optimiser()
            self.setup_optimiser = False

        if self.setup_sampling:
            self.__gen_sampling()
            self.setup_sampling = False

        if self.setup_log:
            self.__gen_log()
            self.setup_log = False

        for method in self.pre_execution_methods:
            method()

    def __populate_quop_result(self):
        """Populate fields of the :meth:`~quop_mpi.Ansatz.quop_result` dictionary.

        Called by rank 0 in :meth:`~quop_mpi.Ansatz.MPI_COMM_WORLD` only.
        """
        self.quop_result["fun"] = copy(self.result["fun"])
        self.quop_result["qubits"] = copy(np.log2(self.system_size))
        self.quop_result["system size"] = copy(self.system_size)
        self.quop_result["ansatz_depth"] = copy(self.ansatz_depth)
        self.quop_result["free params"] = copy(self.free_params)
        self.quop_result["variational parameters"] = deepcopy(
            self.variational_parameters
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
            of a phase-shift unitary in the list passed to the
            :meth:`~quop_mpi.Ansatz.set_observables` whose exponent contains the
            observable vector.

        observables_dict: FunctionDict, optional
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
            from quop_mpi.__utils.__nlopt_wrap import minimize as nlopt_minimize

            self.optimiser = nlopt_minimize
        elif callable(optimiser):
            self.optimiser = optimiser

        self.optimiser_args = optimiser_args
        self.optimiser_log = optimiser_log

        self.setup_optimiser = True

    def __parse_jacobian(self):
        """Bind a QuOp Jacobian Function to the attributes of an instantiated
        :class:`~quop_mpi.Ansatz` instance.
        """
        self.jacobian = interface(
            [self], self.jacobian_input[0], "jacobian", self.subcomms.SUBCOMM
        )

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

    def set_initial_state(
        self, function: Callable, initial_state_dict: dict = None
    ):
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

    def set_sampling(
        self,
        sample_block_size: int,
        function: Callable = None,
        max_sample_iterations: int = 100,
        sampling_dict: dict = None,
    ):
        """Compute the :term:`objective function` using simulated sampling.

        Samples are taken in blocks of `sample_block_size`. These are passed as
        a list of lists to :literal:`function` (a :term:`Sampling Function`), which returns a value for expectation
        value/objective function and a boolean that indicates wether the sampled
        result should be passed to the classical optimiser.

        If :literal:`function` is :literal:`None`, the :term:`objective function` is
        computed as the mean of :literal:`sample_block_size` shots.

        Parameters
        ----------
        sample_block_size : int
            number of shots taken between successive computation of the
            expectation value/objective function
        function : callable, optional
            :term:`Sampling Function`
        max_sample_iterations : int, optional
            maximum number of sample blocks per computation of the expectation
            value/objective function,  overrides the boolean returned by
            :literal:`function`, by default 100
        sampling_dict : FunctionDict, optional
            :term:`FunctionDict` for the Sampling Function
        """

        self.__parse_function_dict__(sampling_dict, "sampling_dict")

        if function is None:
            function = lambda samples: (np.mean(samples), True)

        self.sample_block_size = sample_block_size
        self.max_sample_iterations = max_sample_iterations
        self.sampling_function_input = function

        if not self.setup_sampling:
            self.pre_execution_methods.append(self.__pre_sampling)
            self.post_execution_methods.append(self.__post_sampling)

        self.setup_sampling = True

    def unset_sampling(self):
        """Revert to simulation using exact computation of the
        :term:`objective function`.
        """
        self.setup_sampling = False
        self.sampling = False
        self.pre_execution_methods.remove(self.__pre_sampling)

    def __pre_sampling(self):
        """Preparation for simulated sampling."""

        self.minimum_sampled = np.inf
        self.total_shots = 0
        self.global_minimum_found = False
        self.shots_to_global_minimum = "not found"

        if self.MPI_COMM_WORLD.Get_rank() == 0:
            print("Executing with simulated sampling.")

    def __post_sampling(self):
        """Post simulation steps for simulated sampling."""

        if self.MPI_COMM_WORLD.Get_rank() == 0:

            self.quop_result["sampling total shots"] = self.total_shots
            self.quop_result["sampling minimum measured"] = self.minimum_sampled
            self.quop_result[
                "sampling shots to minimum measured"
            ] = self.shots_to_global_minimum
            self.quop_result["observables global minimum"] = self.global_minimum

    def __gen_sampling(self):
        """Setup for simulated sampling."""

        if self.subcomms.in_subcomm():

            self.sampling = True

            self.__parse_sampling_function()

            self.global_minimum = self.subcomms.SUBCOMM.reduce(
                np.min(np.real(self.observables)), op=MPI.MIN
            )

    def __parse_sampling_function(self):
        """Bind the arguments of a QuOp Sampling Function to the attributes of
        and :class:`~quop_mpi.Ansatz` instance.
        """

        self.sampling_function = interface(
            [self, self.unitaries],
            self.sampling_function_input,
            "sampling test function",
            self.subcomms.SUBCOMM,
        )

    def __sample_expectation_value(self) -> float:
        """Returns the expectation value of QVA with solution quality values
        sampled according to the probability distribution of the system
        state vector.

        Returns
        -------
        float
            expectation value of the sampled solution qualities
        """
        if not self.subcomms.in_subcomm():
            return

        if self.subcomms.SUBCOMM.Get_rank() == 0:
            self.samples = []
            self.sample_indexes = []
        else:
            self.samples = [None]
            self.sample_indexes = [None]

        for _ in range(self.max_sample_iterations):

            # Get the probability from each node in MPI_COMM
            self.__get_local_probabilities()
            total_local_probability = np.array(
                [self.local_probabilities.sum()], dtype=np.float32
            )

            comm_opt_size = self.subcomms.SUBCOMM.Get_size()

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                process_probabilities = np.empty(comm_opt_size, dtype=np.float32)
            else:
                process_probabilities = None

            self.subcomms.SUBCOMM.Gather(
                total_local_probability, process_probabilities, root=0
            )

            if self.subcomms.SUBCOMM.Get_rank() == 0:

                rank_samples = np.random.choice(
                    list(range(comm_opt_size)),
                    size=self.sample_block_size,
                    replace=True,
                    p=process_probabilities,
                )

                ranks, counts = np.unique(rank_samples, return_counts=True)

                samples_per_rank = np.zeros(comm_opt_size, dtype=int)

                for rank, count in zip(ranks, counts):
                    samples_per_rank[rank] = count

            else:
                samples_per_rank = np.empty(comm_opt_size, dtype=int)

            self.subcomms.SUBCOMM.Bcast(samples_per_rank, root=0)

            local_normed_probabilities = (
                self.local_probabilities / self.local_probabilities.sum()
            )

            local_samples_inds = np.random.choice(
                list(range(self.local_i)),
                size=samples_per_rank[self.subcomms.SUBCOMM.Get_rank()],
                replace=True,
                p=local_normed_probabilities,
            ).astype(np.int32)

            local_samples = np.real(self.observables[local_samples_inds]).astype(
                np.float64
            )

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                self.samples.append(np.empty(self.sample_block_size, dtype=np.float64))
                self.sample_indexes.append(
                    np.empty(self.sample_block_size, dtype=np.int32)
                )

            self.subcomms.SUBCOMM.Gatherv(
                local_samples, [self.samples[-1], samples_per_rank], 0
            )

            self.subcomms.SUBCOMM.Gatherv(
                local_samples_inds + self.local_i_offset,
                [self.sample_indexes[-1], samples_per_rank],
                0,
            )

            if self.subcomms.SUBCOMM.Get_rank() == 0:

                self.sampling_function.update_parameters()

                sampling_function_result = self.sampling_function.call(
                    *self.sampling_dict["args"], **self.sampling_dict["kwargs"]
                )

                self.total_shots += len(self.samples[-1])

            else:
                sampling_function_result = None

            expectation, sample_test = self.subcomms.SUBCOMM.bcast(
                sampling_function_result, root=0
            )

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                argmin = np.argmin(self.samples[-1])
                self.sample_minimum_indexes.append(self.sample_indexes[-1][argmin])
                self.variational_parameter_history.append(self.variational_parameters)
                sample_min = self.samples[-1][argmin]
                if self.minimum_sampled > sample_min:
                    self.minimum_sampled = sample_min
                    self.shots_to_global_minimum = copy(self.total_shots)
            if sample_test:
                break

        return expectation

    def set_log(self, filename: str, label: str, action: str = "a"):
        """Creates a CSV in which to save simulation results after a call to
        :meth:`~quop_mpi.Ansatz.execute`.

        Parameters
        ----------
        filename : str
            path to the log file
        label : str
           simulation identifier
        action : {'a', 'w'}, optional
            'a' to append or 'w' overwrite, by default 'a'
        """

        self.filename = ensure_path_and_extension(filename, "csv")
        self.label = label
        self.log_action = action

        self.repeat = 1  # needed if logging results from the execute method

        self.setup_log = True

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
        value of :meth:`~quop_mpi.Ansatz.variational_parameters`.

        Returns
        -------
        float
            objective function value
        """

        if self.subcomms.get_subcomm_index() == 0:
            return self.__get_expectation_value()

    def objective(self, variational_parameters: Union[list[float], np.ndarray[float]]) -> float:
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

    def set_parallel_jacobian(
        self,
        nodes_per_subcomm: int,
        processes_per_node: int,
        maxcomm: int,
        method: Union[str, Callable] = "forward",
        h: float = None,
    ):
        """Specify :term:`optimisation<optimiser>` of the :term:`variational
        parameters` using parallel computation of the jacobian.

        This creates MPI subcommunicators containing duplicates of the
        :class:`~quop_mpi.Ansatz` instance which return partial derivative information to
        the root MPI process during optimisation.

        Parameters
        ----------
        nodes_per_subcomm : int
            MPI nodes per subcommunicator
        processes_per_node : int
            MPI processors associated with each node
        maxcomm : int
            maximum number of created MPI subcommunicators (and :class:`~quop_mpi.Ansatz`
            instance duplicates) if `nodes_per_subcomm > 1`, or the maximum
            number of MPI subcommunicators per node if `nodes_per_subcomm = 1`
        method :{'forward', 'central'} or callable, optional
            'forward' or 'central' to used the forward difference or central
            difference method for numerical approximation of the partial
            derivatives, or a QuOp Jacobian Function, by default 'forward'
        h : float, optional
            step-size used by the forward or central difference methods, by
            default :literal:`np.sqrt(np.finfo(float).eps)`
        """

        self.nodes_per_subcomm = nodes_per_subcomm
        self.processes_per_node = processes_per_node
        self.maxcomm = maxcomm
        self.jacobian_input = [method]
        self.h = h if h is not None else np.sqrt(np.finfo(float).eps)
        self.reset = True

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
            self.subcomms.SUBCOMM = self.subcomms.SUBCOMM

        while not busy_comm:

            if self.subcomms.in_subcomm():

                max_comm_size = max_compatible_size(self.unitaries, self.system_size, self.subcomms.SUBCOMM.size, self.subcomms.SUBCOMM.py2f())
                dropcount = self.subcomms.SUBCOMM.size  - max_comm_size 

            else:
                break

            if dropcount > 0:
                self.subcomms.shrink_subcomms(dropcount)
                self.subcomms.SUBCOMM = self.subcomms.SUBCOMM
            else:

                busy_comm = True
            

        if self.subcomms.in_subcomm():
            # create the default vector partitioning, may be altered durring the unitary planning phase.
            self.local_i, self.local_i_offset, self.alloc_local, self.partition_table = vector_partitioning(self.system_size, self.subcomms.SUBCOMM)
            
    def __update_var_map(self):
        """Queries :literal:`Unitary` instances passed to the :class:`~quop_mpi.Ansatz` instance via the
        :meth:`~quop_mpi.Ansatz.set_unitaries` methods to determine the number and ordering of
        QVA variational parameters.
        """
        if self.subcomms.get_n_subcomms() > 1:
            self.var_map = [[] for _ in range(self.subcomms.get_n_subcomms())]
            if self.subcomms.in_subcomm():
                for var in range(self.n_free_params):
                    self.var_map[1:][var % (self.subcomms.get_n_subcomms() - 1)].append(
                        var
                    )
        else:
            self.var_map = None

    def __gen_parallel(self):
        """Creates MPI subcommunicators for QVA simulation with or without
        parallel computation of the objective function Jacobian.
        """

        self.subcomms = subcomms(
            self.nodes_per_subcomm,
            self.processes_per_node,
            self.maxcomm,
            self.MPI_COMM_WORLD,
        )

        if self.subcomms.in_subcomm():
            self.MPI_COMM = self.subcomms.SUBCOMM

        if self.subcomms.get_n_subcomms() > 1 and self.subcomms.in_subcomm():

            self.subcomms.create_jaccomm()

            if self.subcomms.in_jaccomm():
                self.subcomms.JACCOMM = self.subcomms.JACCOMM

    def __gen_unitaries(self):
        """Calls methods associated with :literal:`Unitary` instances to determine the
        parallelisation scheme required for computation of the system dynamics.
        Generates operators associated with the :literal:`Unitary` instances.
        """
        if self.subcomms.in_subcomm():
            #for unitary in self.unitaries:
            #    if unitary is not self.planner:
            #        unitary._Unitary__copy_plan(self.planner)

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

            if self.initial_state_input is None:
                from .state import equal
                self.set_initial_state(equal)

            self.__parse_initial_state_function()

            self.ansatz_initial_state = self.initial_state_function.call(
                *self.initial_state_dict["args"], **self.initial_state_dict["kwargs"]
            )

            # do in evolve state
            #self.context.state = self.ansatz_initial_state

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
                RuntimeError(
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

            if self.jacobian_input is not None:

                if self.jacobian_input[0] == "forward":
                    self.jacobian_input = [forward_differences]
                elif self.jacobian_input[0] == "central":
                    self.jacobian_input = [central]

                self.__parse_jacobian()

                self.optimiser_args["jac"] = self.__mpi_jacobian


    def __assign_backend(self):

        self.backend = import_module(f"quop_mpi.__lib.{config.backend}")

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
                    self.subcomms.SUBCOMM)

            self.subcomms.SUBCOMM.barrier()

            for unitary in self.unitaries:
                unitary.context = self.context

    def setup(self):
        """Determine the parallelisation scheme and performs setup tasks
        required by extension modules.
        """
        if self.reset and not self.setup_called:
            self.seed += 1

            #TODO trigger setup on changes to config.backend
            self.__assign_backend()

            self.__gen_parallel()

            self.__check_comm_size()

            self.__initialise_context()

            self.__gen_free_params()
            self.setup_depth = True
            self.setup_observables = True
            self.setup_initial_state = True
            self.setup_optimiser = True

            self.reset = False
            self.setup_called = True

    def __post_log(self):
        """Close the results log file on simulation completion."""

        if self.MPI_COMM_WORLD.Get_rank() == 0 and self.log:
            self.logfile.close()

    def __post_unitaries(self):
        """Free memory managed by extension modules on simulation completion."""
        if self.subcomms.in_subcomm():
            for unitary in self.unitaries:
                if unitary.planned:
                    unitary.destroy()

    def __post_parallel(self):
        """Free subcommunicators associated with the :class:`~quop_mpi.Ansatz` instance on
        simulation completion."""
        self.subcomms.free()

    def destroy(self):
        """Call methods to close the results log file, free memory managed by
        extension modules and free MPI subcommunicators created by the
        :class:`~quop_mpi.Ansatz` instance.
        """

        if not self.reset or not self.setup_called:
            return

        if not self.benchmarking and self.log:
            self.__post_log()

        if not self.setup_unitaries:
            self.__post_unitaries()
            self.setup_unitaries = True

        if not self.setup_parallel:
            self.__post_parallel()
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

                    #unitary.initial_state[: self.local_i] = self.final_state[
                    #    : self.local_i
                    #]

                    #print(unitary.context.initial_state)
                    unitary.propagate(evolution_parameter)
                    #self.context.state = self.context.state

                    # propgators handle this now using pointer swapping if needed.
                    #self.final_state[: self.local_i] = unitary.final_state[
                    #    : self.local_i
                    #]

            if self.subcomms.SUBCOMM.Get_rank() == 0:
                self.n_evolutions += 1
            self.last_evaluated = copy(x)

    def evaluate(self, variational_parameters: Union[list[float], np.ndarray[float]]) -> float:
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

            if variational_parameters is not None:
                self.variational_parameters = np.array(
                    variational_parameters, dtype=np.float64
                )

                self.set_depth(len(variational_parameters) // self.total_params)

            self.__pre()

            if self.variational_parameters is None:
                self.variational_parameters = self.gen_initial_params(self.ansatz_depth)

        if self.subcomms.in_subcomm():

            self.stop = False
            self.n_evolutions = 0

            if self.subcomms.get_subcomm_index() == 0:
                
                self.objective_cnt = 0

                if self.subcomms.SUBCOMM.Get_rank() == 0:

                    self.__execute_subcomm_group_zero()
                else:

                    while not self.stop:
                        self.__free_params_objective(None)

                self.__post()

                if self.log:
                    self.__log_update()

            else:
                while not self.stop:
                    self.__mpi_jacobian(None)

                self.__post()

    def __execute_subcomm_group_zero(self):
        """Tasks carried out at :meth:`~quop_mpi.Ansatz.subcomms` group zero during simulation
        of a QVA via a called to :meth:`~quop_mpi.Ansatz.execute`"""
        if self.record_objective:
            self.total_n_evolutions = []

        self.neval_mpi_jac = 0

        self.time = time()
        x = self.__get_free_params()

        if len(x) > 0:

            self.result = self.optimiser(
                self.__free_params_objective, x, **self.optimiser_args
            )

        else:
            # if there are no free parameters, just compute the value
            # of the objective function and then stop.
            self.result = {"fun": None}
            self.result["fun"] = self.objective(self.variational_parameters)
            self.result["nfev"] = 1
            self.result["success"] = "N/A"
            self.result["x"] = copy(self.variational_parameters)

        self.stop = True

        self.__free_params_objective(None)

        if self.subcomms.get_n_subcomms() > 1:
            self.__mpi_jacobian(None)

        self.time = time() - self.time

    def print_result(self):
        """Print a summary of the results of the last :term:`QVA` simulation."""

        if self.MPI_COMM_WORLD.Get_rank() != 0:
            return

        print("\nQuOp_MPI Simulatuion Summary", flush=True)
        print("============================\n", flush=True)
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

    def benchmark(
        self,
        ansatz_depths: iterable[int],
        repeats: int,
        param_persist: bool = False,
        verbose: bool = True,
        filename: str = None,
        label: str = "test",
        save_action: str = "a",
        time_limit: int = None,
        suspend_path: str = None,
    ):
        """A method by which to study how a :term:`QVA` performs as the number
        of :term:`ansatz iterations<ansatz depth>` increases.

        Parameters
        ----------
        ansatz_depths : iterable[int]
            integers specifying a sequence of :term:`ansatz depths<ansatz depth>`
        repeats : int
            number of repeats at each :term:`ansatz depth`
        param_persist : bool, optional
            if :literal:`True` the :term:`optimised<optimiser>` 
            :term:`variational parameter <variational parameters>` values which achieved
            the lowest :term:`objective function` value  for all repeats at
            :literal:`ansatz_depth` will be used as starting parameters for the first
            :literal:`ansatz_depth * total_params` at :literal:`ansatz_depth += 1`
        verbose : bool, optional
            if :literal:`True`, print current the :term:`ansatz depth`, repeat number and 
            :term:`optimisation<optimiser>` results by default :literal:`True`
        filename : str or None, optional
            name of :literal:`*.h5` file in which to :meth:`~quop_mpi.Ansatz.system.save` the
            optimised :term:`system state` and :term:`observables`
        label : str, optional
            if :literal:`filename` is not :literal:`None`, :literal:`*.h5` data will be saved as
            "filename/label_depth_repeat", by default :literal:`"test"`
        save_action : {'a', 'w'}, optional
            action taken during first file write: 'a' to append, 'w' to
            overwrite, by default 'a'
        time_limit : int or None, optional
            total allocated in-program time in seconds, if the time of the
            previous simulation exceeds the time remaining, the benchmark is
            suspended
        suspend_path : str or None, optional
            path to the suspend file if :literal:`time_limit` is not :literal:`None`
        """

        self.destroy()
        self.setup()

        ansatz_depth_temp = deepcopy(
            self.ansatz_depth
        )  # return to this value after benchmarking

        self.benchmarking = True

        suspend_path = "suspend" if suspend_path is None else suspend_path

        self.tracker = job_tracker(
            repeats,
            list(ansatz_depths)[-1],
            time_limit,
            self.MPI_COMM_WORLD,
            seed=self.seed,
            suspend_path=suspend_path,
        )

        previous_params = None

        first = not self.tracker.got_match

        while not self.tracker.complete:

            repeat, depth = self.tracker.get_job()
            self.set_seed(self.tracker.get_seed())
            self.ansatz_depth = depth
            self.set_depth(depth)

            if repeat == 1 or first:

                self.set_depth(depth)
                first = False

                if (
                    self.subcomms.get_subcomm_index() == 0
                    and verbose
                    and self.subcomms.SUBCOMM.Get_rank() == 0
                ):
                    print(f"Starting depth = {depth}:", flush=True)

            self.__pre()

            self.repeat = repeat

            if self.subcomms.get_subcomm_index() == 0:

                if (not param_persist) or (depth == 1):
                    self.variational_parameters = self.__gen_initial_params(
                        depth
                    )
                
                else:

                    if self.subcomms.SUBCOMM.Get_rank() == 0:
                        n_previous = len(self.tracker.results_dict[depth - 1])
                    else:
                        n_previous = None

                    n_previous = self.subcomms.SUBCOMM.bcast(n_previous, root = 0)

                    if n_previous > 0:

                        if self.subcomms.SUBCOMM.Get_rank() == 0:
                            if (
                                self.tracker.job_list[self.tracker.job_index][1]
                                != self.tracker.job_list[self.tracker.job_index - 1][1]
                            ) or (previous_params is None):
                                funs = [
                                    result["fun"]
                                    for result in self.tracker.results_dict[depth - 1]
                                ]
                                xs = [
                                    result["variational parameters"]
                                    for result in self.tracker.results_dict[depth - 1]
                                ]

                            previous_params = xs[np.argmin(funs)]
                        else:
                            previous_params = None

                        previous_params = self.subcomms.SUBCOMM.bcast(previous_params, root = 0)

                        self.variational_parameters = np.empty(
                            depth * self.total_params, dtype=np.float64
                            )

                        self.variational_parameters[
                            : len(previous_params)
                            ] = previous_params

                        new_params = self.__gen_initial_params(1)

                        self.variational_parameters[
                            -self.total_params :
                        ] = new_params

                    else:

                        self.variational_parameters = (
                            self.__gen_initial_params()
                        )

                if verbose and self.subcomms.SUBCOMM.Get_rank() == 0:
                    print(f"{repeat} of {repeats} at depth {depth}...", flush=True)

                self.execute()

                if verbose:
                    self.print_result()

                if filename is not None:
                    if first:
                        self.save(
                            ensure_path_and_extension(filename, "h5"),
                            f"{label}_{str(depth)}_{str(repeat)}",
                            action=save_action,
                        )
                    else:
                        self.save(
                            ensure_path_and_extension(filename, "h5"),
                            f"{label}_{str(depth)}_{str(repeat)}",
                            action="a",
                        )

                self.tracker.update(self.quop_result)
                first = False

            else:

                self.execute()
                self.tracker.update(None)

        self.benchmarking = False
        self.ansatz_depth = ansatz_depth_temp

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
            #prob = np.abs(self.final_state) ** 2
            return gather_array(
                np.abs(self.context.state)**2, self.unitaries[0].partition_table, self.subcomms.SUBCOMM
            )

    def save(self, file_name: str, config_name: str, action: str = "a"):
        """Write the :term:`final state`, :term:`observables` and results
        summary to a HDf5 file.

        Parameters
        ----------
        file_name : str
            file path to saved data
        config_name : str
            simulation identifier
        action : {'a', 'w'}, optional
            'a' to append or 'w' to overwrite, by default 'a'

        Notes
        -----

        Data is saved into a :literal:`*.h5` file with the following structure.

        ::

            ├── config_name
                ├── final_state 
                ├── observables

        The minimization result is saved in the 'minimize_result' attribute of
        'config_name' as a formatted string.

        Multiple configurations with a unique config_name can be stored in the
        same .h5 file. HDF5 files are supported in python by the `h5py
        <https://www.h5py.org/>`_ package. With it, a saved configuration can be
        accessed as follows:

        .. code-block:: python

            import h5py

            config_name = "my_simulation"

            f = h5py.File(file_name + ".h5", "r") final_state =
            np.array(f[config_name]['final_state']).view(np.complex128)
            eigenvalues =
            np.array(f[config_name]['eigenvalues']).view(np.complex128)
            observables =
            np.array(f[config_name]['observables']).view(np.float64)

            print(f["my_simulation"].attrs["minimize_result"])

        .. warning::

            The :literal:`"final_state"` and :literal:`"observables"` datasets are saved using Fortran
            subroutines which make use of parallel HDF5.

            The complex values of the final_state array are saved as a compound
            datatype consisting of contiguous double precision reals. This is
            equivalent to the np.complex128 NumPy datatype. To access this data
            without a loss of precision in python, the user must set the
            **view** of the NumPy array to np.complex128, rather than casting it
            to np.complex128 using the dtype keyword.

            Similarly, the observables array, which is saved as an array of
            double-precision reals, should have its view set to np.float64.
        """

        if self.subcomms.get_subcomm_index() != 0:
            return

        from quop_mpi.__lib import parallel_io

        if self.MPI_COMM_WORLD.Get_rank() == 0:

            import h5py

            self.config_name = config_name

            file_name = ensure_path_and_extension(file_name, "h5")
            File = h5py.File(file_name, action)

            # If the config_name already exists in the target file, add an underscore.
            duplicate = True
            while duplicate:
                if self.config_name in File:
                    self.config_name += "_"
                else:
                    duplicate = False

            config = File.create_group(self.config_name)

            if self.result is not None:
                config.attrs["minimize_result"] = str(self.result)

            File.create_dataset(
                f"{self.config_name}/initial_phases",
                data=self.variational_parameters,
                dtype=np.float64,
            )
            File.close()
        else:
            self.config_name = None

        file_name = self.subcomms.SUBCOMM.bcast(file_name, root=0)
        self.config_name = self.subcomms.SUBCOMM.bcast(self.config_name, root=0)

        parallel_io.io.save_dist_complex(
            file_name,
            f"{self.config_name}/",
            "final_state",
            "a",
            self.system_size,
            self.local_i_offset,
            self.context.state[: self.local_i],
            self.subcomms.SUBCOMM.py2f(),
        )

        parallel_io.io.save_dist_complex(
            file_name,
            f"{self.config_name}/",
            "initial_state",
            "a",
            self.system_size,
            self.local_i_offset,
            self.ansatz_initial_state[: self.local_i],
            self.subcomms.SUBCOMM.py2f(),
        )

        parallel_io.io.save_dist_real(
            file_name,
            f"{self.config_name}/",
            "observables",
            "a",
            self.system_size,
            self.local_i_offset,
            self.observables[: self.local_i],
            self.subcomms.SUBCOMM.py2f(),
        )

    def gen_initial_params(
        self, ansatz_depth:int = None
    ) -> np.ndarray[np.float64]:
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

    def __gen_initial_params(
        self, ansatz_depth: int =  None
    ) -> np.ndarray[np.float64]:
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
                param_iters[
                    self.param_map[i] : self.param_map[i + 1]
                ] = unitary.gen_initial_params()

        self.subcomms.SUBCOMM.Bcast([params, MPI.DOUBLE], 0)

        return params

    def __get_local_probabilities(self) -> np.ndarray[np.float64]:
        """Compute the probabilities of states local to each MPI process.

        Returns
        -------
        ndarray[float64]
            1-D array containing :meth:`~quop_mpi.Ansatz.local_i` state probabilities with
            global index offset :meth:`~quop_mpi.Ansatz.local_i_offset`
        """
        self.local_probabilities = (
            (np.abs(self.context.state[: self.local_i]) ** 2).astype(np.float64)
        )
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
            #self.state_norm = self.subcomms.SUBCOMM.allreduce(
            #    np.sum(self.__get_local_probabilities()), op=MPI.SUM
            #)
            #return self.state_norm

    def __get_expectation_value(self) -> float:
        """Compute the expectation value at :meth:`~quop_mpi.Ansatz.variational_parameters`.

        Returns
        -------
        float
            expectation value at :meth:`~quop_mpi.Ansatz.variational_parameters`
        """

        if self.sampling:
            return self.__sample_expectation_value()

        #self.__get_local_probabilities()

        #local_expectation = np.dot(self.local_probabilities, self.observables)

        #return np.real(self.subcomms.SUBCOMM.allreduce(local_expectation, op=MPI.SUM))

        return self.context.get_expectation_value()

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
        float or None]
            returns the objective function value at rank 0 in
            :meth:`~quop_mpi.Ansatz.MPI_COMM_WORLD`, None otherwise
        """
        self.stop = self.subcomms.SUBCOMM.bcast(self.stop, root=0)

        if not self.stop:

            self.variational_parameters = self.subcomms.SUBCOMM.bcast(
                variational_parameters, root=0
            )

            self.__evolve_state(self.variational_parameters)

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

    def __gen_log(self):
        """Create or open a log file."""

        self.n_log_fields = 6

        if self.MPI_COMM_WORLD.Get_rank() == 0:

            if os.path.exists(self.filename) and self.log_action == "a":
                self.logfile = open(self.filename, "a", newline="")
                self.logfile_csv = csv.writer(self.logfile)
            else:

                self.__create_new_logfile()

        self.log = True

    def __create_new_logfile(self):
        """Create a new log file, called by rank 0 at :meth:`~quop_mpi.Ansatz.MPI_COMM_WORLD`
        only."""

        headings = [
            "label",
            "system_size",
            "ansatz_depth",
            "repeat",
            "state_norm",
            "simulation_time",
            "MPI_nodes",
            "MPI_jacobian_evaluations",
        ]

        if self.sampling:
            headings.extend(
                ("total_shots", "minimum_sampled", "shots_to_global_minimum")
            )

        if self.optimiser_log is not None:
            headings.extend(iter(self.optimiser_log))

        self.logfile = open(self.filename, "w")
        self.logfile_csv = csv.writer(self.logfile)
        self.logfile_csv.writerow(headings)

    def __log_update(self):
        """Write simulation information to an active log file."""

        if self.MPI_COMM_WORLD.Get_rank() != 0:
            return

        log_output = [
            self.label,
            self.system_size,
            self.ansatz_depth,
            self.repeat,
            self.state_norm,
            self.time,
            self.subcomms.SUBCOMM.size,
            self.neval_mpi_jac,
        ]

        if self.sampling:
            log_output.extend(
                (
                    self.total_shots,
                    self.minimum_sampled,
                    self.shots_to_global_minimum,
                )
            )
        if self.optimiser_log is not None:
            log_output.extend(
                self.result[optimiser_log] for optimiser_log in self.optimiser_log
            )
        self.logfile_csv.writerow(log_output)

        self.logfile.flush()

    def __mpi_jacobian(self, x: np.ndarray[float]) -> Union[float, None]:
        """Compute the objective function gradient with parallel
        instances of the :class:`~quop_mpi.Ansatz` class.

        Parameters
        ----------
        x : ndarray[float]
            1-D real array of free variational parameters

        Returns
        -------
        float or None
            returns the objective function gradient to rank 0 in
            :meth:`~quop_mpi.Ansatz.MPI.COMM_WORLD`, None otherwise
        """
        self.subcomms.JACCOMM.barrier()
        self.stop = self.subcomms.JACCOMM.bcast(self.stop, 0)

        if self.stop:
            self.subcomms.JACCOMM.barrier()
            return

        self.variational_parameters = self.subcomms.JACCOMM.bcast(
            self.variational_parameters, 0
        )

        x = self.subcomms.JACCOMM.bcast(x, 0)

        if self.subcomms.JACCOMM.Get_rank() != 0:
            self.variational_parameters = self.__place_free_params(
                self.variational_parameters, x
            )

        partials = []
        if self.subcomms.JACCOMM.Get_rank() != 0:
            for var in self.var_map[self.subcomms.get_subcomm_index()]:
                self.jacobian.update_parameters()
                partials.append(self.jacobian.call(self.free_params[var]))

        opt_root = self.subcomms.get_subcomm_roots()[self.subcomms.colour]

        if self.subcomms.JACCOMM.Get_rank() == 0:
            jacobian = np.zeros(self.n_free_params, dtype=np.float64)
            reqs = []
            for root, mapping in zip(self.subcomms.get_subcomm_roots(), self.var_map):
                if root > 0:
                    for var in mapping:
                        self.MPI_COMM_WORLD.Recv(
                            [jacobian[var : var + 1], MPI.DOUBLE], source=root, tag=var
                        )

        elif self.subcomms.SUBCOMM.Get_rank() == 0:
            reqs = []
            jacobian = None
            for part, mapping in zip(partials, self.var_map[self.subcomms.get_subcomm_index()]):
                self.MPI_COMM_WORLD.Send(
                    [np.array([part]), MPI.DOUBLE], dest=0, tag=mapping
                )
        else:
            jacobian = None

        self.subcomms.JACCOMM.barrier()

        if self.record_objective:
            if self.subcomms.JACCOMM.Get_rank() == 0:
                self.n_evolutions = self.subcomms.JACCOMM.reduce(
                    self.n_evolutions, op=MPI.SUM, root=0
                )
            else:
                self.subcomms.JACCOMM.reduce(self.n_evolutions, op=MPI.SUM, root=0)
                self.n_evolutions = 0

        if self.subcomms.JACCOMM.Get_rank() == 0:

            self.neval_mpi_jac += 1
            return jacobian

        else:
            return None

    def __is_zero(self, x: float) -> bool:
        """Check if float is equivalent to zero up to double precision.

        Parameters
        ----------
        x : float
            a real number

        Returns
        -------
        bool
            wether :literal:`x` is functionally zero
        """
        return (x >= -np.finfo(np.float64).eps) and (x <= np.finfo(np.float64).eps)

    def set_free_params(
        self, function: Callable, free_params_dict: dict = None
    ):
        """Optimise over a subset of the :term:`free variational parameters
        <free parameters>`.

        Parameters
        ----------
        function : callable
            :term:`Free Parameters Function`
        free_params_dict : dict, optional
            :term:`FunctionDict` for the Free Parameters Function

        Examples
        --------

        A Free Params Function that restricts :term:`optimisation<optimiser>` to
        the last :term:`ansatz iteration<ansatz depth>`:

        .. code-block:: python

            def last_ansatz_iteration(total_params, ansatz_depth):
               return list(
                   range((ansatz_depth - 1) * total_params, ansatz_depth * total_params)
               )
        """
        self.__parse_function_dict__(free_params_dict, "free_params_dict")
        self.free_params_function = function

    def __gen_free_params(self):
        """Initial generation of free parameter indexes."""

        if not self.subcomms.in_subcomm():
            return

        if callable(self.free_params_function) and not isinstance(
            self.free_params_function, interface
        ):

            self.free_params_function = interface(
                [self], self.free_params_function, "free_params", self.subcomms.SUBCOMM
            )

        elif self.free_params_function is not None:

            self.free_params = self.free_params_function

        else:

            self.free_params = list(range(self.ansatz_depth * self.total_params))
            self.n_free_params = len(self.free_params)

    def __update_free_params(self):
        """Update free parameter indexes."""

        if isinstance(self.free_params_function, interface):

            self.free_params_function.update_parameters()

            self.free_params = self.free_params_function.call(
                *self.free_params_dict["args"], **self.free_params_dict["kwargs"]
            )

        else:

            self.free_params = list(range(self.ansatz_depth * self.total_params))

        self.n_free_params = len(self.free_params)

    def __get_free_params(self):
        return [self.variational_parameters[index] for index in self.free_params]

    def __place_free_params(
        self, all_params: np.ndarray[float], params: np.ndarray[float]
    ) -> np.ndarray[np.float64]:
        """Maps a subset of free parameters to the full set of variational
        parameters.

        Array :literal:`params` is mapped to positions in :literal:`all_params` based on indexes
        contained in the  :meth:`~quop_mpi.Ansatz.free_params` attribute.

        Parameters
        ----------
        all_params : ndarray[float]
            1-D real array of variational parameters
        params : ndarray[float]
            1-D real array of variational parameters

        Returns
        -------
        ndarray[float64]
            variational parameters with updated free parameters
        """

        for index, param in zip(self.free_params, params):
            all_params[index] = param
        return all_params

    def __free_params_objective(self, params: np.ndarray[float]) -> Union[float, None]:
        """Maps a subset of free parameters to the full set of variational
        parameters.

        Parameters
        ----------
        params : ndarray[float]
            1-D real array of variational parameters

        Returns
        -------
        float or None
            returns the objective function value to rank 0 in MPI.COMM_WORLD,
            None otherwise
        """

        if self.subcomms.SUBCOMM.Get_rank() != 0 or params is None:
            return self.__objective(None)

        self.variational_parameters = self.__place_free_params(
            self.variational_parameters, params
        )

        return self.__objective(self.variational_parameters)
