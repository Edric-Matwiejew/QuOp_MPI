import os
import sys
import pickle
import csv
import atexit
from warnings import warn
from copy import copy, deepcopy
from time import time
import numpy as np
from mpi4py import MPI
from .__utils.__interface import interface
from .__utils.__mpi import shrink_communicator, gather_array
import inspect

def print_method(func):
    def decorator(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(func.__name__)
        return func(*args, **kwargs)
    return decorator

def MPI_trace(cls):
    for name, method in inspect.getmembers(cls):
        if (not inspect.ismethod(method) and not inspect.isfunction(method)) or inspect.isbuiltin(method):
            continue
        setattr(cls, name, print_method(method))
    return cls

def forward_differences(variational_parameters, evaluate, h, var):
    x = np.array(variational_parameters, np.float64)
    #h = 1.4901161193847656e-08
    #h = 1.4901161193847656e-05
    expectation = evaluate(x)
    x[var] += h
    expectation_forward = evaluate(x)
    return (expectation_forward - expectation) / h


def central(variational_parameters,  evaluate, h, var):
    x = variational_parameters
    #h = 1.4901161193847656e-08
    expectation = evaluate(x)
    x_back = copy(x)
    x_forward = copy(x)
    x_back[var] -= h
    x_forward[var] += h
    expectation_back = evaluate(x_back)
    expectation_forward = evaluate(x_forward)
    return (expectation_forward - 2 * expectation + expectation_back) / h


#@MPI_trace
class Ansatz:

    """Used to define and simulate a quantum variational algorithm.

    :param system_size: Size of the quantum system, :math:`N`.
    :type system_size: integer

    :param MPI_communicator: An MPI4Py MPI communicator object.
    :type MPI_communicator: optional, default = 'MPI.COMM_WORLD'
    """

    def __init__(self, system_size, MPI_communicator=MPI.COMM_WORLD):

        self.system_size = system_size
        self.COMM = MPI_communicator

        # initialisation inputs
        self.COMM = MPI_communicator  # global MPI communicator, usually MPI.COMM_WORLD

        # variables that must be set by the 'pre' method of the child class
        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None
        self.partition_table = False
        self.observables = None
        self.observable_input = None
        self.observables_func = None
        self.variational_parameters = None

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

        # parameters linked to optional methods in the 'system' class
        self.observable_map_input = (
            None  # scalar transformation on the observable values
        )
        self.objective_map_input = None
        self.setup_log = False  # whether results will be recorded in a *.log file.

        # variables managed by the 'system' class
        self.stop = False  # synchronise ranks during optimisation

        self.COMM_OPT = None  # communicator used for optimisation
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

        self.COMM_JAC = None
        self.jac_ranks = None

        self.verbose_objective = False
        self.objective_cnt = 0
        self.record_objective = False

        self.n_evolutions = 0
        self.total_n_evolutions = []

        self.log = False

        self.set_parallel()

        self.setup_depth = True
        self.setup_parallel = True
        self.setup_unitaries = True
        self.setup_observables = True
        self.setup_initial_state = True
        self.setup_observable_map = False
        self.setup_objective_map = False
        self.setup_log = False
        self.setup_optimiser = True

        self.time_limit = None
        self.suspend_path = None
        self.available_time = None

        self.result = None

        self.seed = 0

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

        atexit.register(self.__exit)

    def __exit(self):
        if self.setup_called:
            self.destroy()
        self.COMM.barrier()

    def __pre(self):

        if self.setup_depth:
            self.__gen_depth()
            self.setup_depth = False

        #if self.setup_var_map:
        self.__update_var_map()
        #    self.setup_var_map = False

        if self.setup_observables:
            self.__gen_observables()
            self.setup_observables = False

        if self.setup_initial_state:
            self.__gen_initial_state()
            self.setup_initial_state = False

        if self.setup_observable_map:
            self.__gen_observable_map()
            self.setup_observable_map = False

        if self.setup_objective_map:
            self.__gen_objective_map()
            self.setup_objective_map = False

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

    def __post(self):

        self.variational_parameters = None


        # @TODO GET EXPECTATION REGARDLESS OF SAMPLING
        # @TODO warn if norm is not conserved.
        self.state_norm = self.__get_state_norm()
        if self.COMM.Get_rank() == 0:
            if self.result is not None:
                self.quop_result["system size"] = self.system_size
                self.quop_result["final objective function"] = self.result['fun']
                self.quop_result["qubits"] = np.log2(self.system_size)
                x = self.result['x']
                self.quop_result['variational parameters'] = (
                        (''.join([f'{str(" "):<36}{str(x[i:i+4])[1:-1]}\n' for i in range(0, len(x), 4)]))[36:-2]
                        )
                self.quop_result["execution time"] = self.time
                self.quop_result["optimiser convergence success"] = self.result['success']
                self.quop_result["final state norm"] = self.state_norm

        for method in self.post_execution_methods:
            method()

    def set_unitaries(self, unitaries):
        """Define the phase-shift and mixing unitaries used by the QVA. The
        untaries are passed in a python list in order of application from left
        to right.

        :param unitaries: List of unitaries corresponding to one application of the ansatz :math:`D=1`.
        :type unitaries: list, Unitary
        """

        self.unitaries = unitaries

        self.param_map = np.zeros(len(self.unitaries) + 1, int)

        for i, unitary in enumerate(self.unitaries):
            self.param_map[i + 1] = unitary.n_params

        self.total_params = np.sum(self.param_map)
        self.param_map = np.cumsum(self.param_map)

        self.reset = True

    def set_observables(self, function, kwargs=None):
        """Define the observables used during calculation of the objective
        function.

        :param function: A callable returning a local partition of :math:`\\text{diag}(\hat{Q})` or an integer specifying the index of the phase-shift unitary in the list passed to the :meth:`~Ansatz.set_observables` whose exponent contains :math:`\\text{diag}(\hat{Q})`.

        :type function: callable or integer

        :param kwargs: Keyword arguments to pass to the observable function during generation of the observables.
        :type kwargs: optional, default = None
        """

        if kwargs is None:
            kwargs = {}

        self.observable_input = [function, kwargs]

        self.setup_observables = True

    def set_optimiser(self, optimiser, optimiser_args=None, optimiser_log=None):
        """Defines the classical optimiser algorithm used, arguments passed to
        the optimiser and fields in the optimiser dictionary to write to the
        log file (when using :meth:`~system.log_results`). QuOp_MPI supports
        optimisers provided by SciPy through its minimize method 
        `minimize <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
        and optimisers provided by the `NLopt
        <http://nlopt.readthedocs.io/en/latest/>`_ package with respect to
        minimisation with scalar constraints through a SciPy-like interface.

        The default optimiser is the BFGS algorithm, which is set internally as follows:

        .. code-block:: python

            self.set_optimiser( 'scipy',
                    {'method':'BFGS','options':{'gtol':1e-3}},
                                ['fun','nfev','success'])


        :param optimiser: 'scipy' to use the SciPy, 'nlopt' to use NLopt, or a callable QuOp_MPI-compatible optimisation function.
        :type optimiser: string

        :param optimiser_args: Arguments to pass to the optimiser.
        :type optimiser_args: dictionary

        :param optimiser_log: Results of the optimisation process are stored in a dictionary. These values may be logged by passing a list of the corresponding keys.
        :type optimiser_log: array, string
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

        if self.parallel != "global":
            if "jac" in optimiser_args:
                self.jacobian_input = [copy(self.optimiser_args["jac"])]
                self.optimiser_args["jac"] = self.__mpi_jacobian
            else:
                self.jacobian_input = [forward_differences]
                self.optimiser_args["jac"] = self.__mpi_jacobian

        self.setup_optimiser = True

    def __parse_jacobian(self):

        self.jacobian = interface(
            [self],
            self.jacobian_input[0],
            "jacobian",
            self.COMM_OPT,
        )

    def set_depth(self, depth):
        """Define the circuit depth, :math:`D`.

        :param depth: Number of ansatz repetitions.
        :type depth: integer
        """
        if depth != self.ansatz_depth:
            self.ansatz_depth = int(depth)
            self.setup_depth = True

    def set_observable_map(self, func, kwargs=None):
        """Define a function that acts on the observable values during
        calculation of the expectation value. The function should take a Numpy
        array as its input, and return a Numpy array of the same size.

        :param func: Function to apply to the observable values.
        :type func: callable
        """
        if kwargs is None:
            kwargs = {}

        self.observable_map_input = [func, kwargs]

        self.setup_observable_map = True

    def __parse_observable_mapping(self):

        self.observable_map = interface(
            [self, self.unitaries],
            self.observable_map_input[0],
            "observable mapping",
            self.COMM_OPT,
        )

    def unset_observable_map(self):
        """Undo the function mapping defined by
        :meth:`~Ansatz.set_observable_map`."""
        self.observable_map_input = None

    def set_objective_map(self, func, kwargs=None):
        """Define a function that acts on the expectation value prior to it
        being passed to the optimiser. The function should take a float as its
        input and return a float.

        :param func: Function to apply to the expectation value.
        :type func: callable:

        :param kwargs: Keyword args to pass to the function.
        :type kwargs: optional, dictionary, default = None
        """

        if kwargs is None:
            kwargs = {}

        self.objective_map_input = [func, kwargs]

        self.setup_objective_map = True

    def __parse_objective_mapping(self):

        self.objective_map = interface(
            [self, self.unitaries],
            self.objective_map_input[0],
            "objective mapping",
            self.COMM_OPT,
        )

    def unset_objective_map(self):
        """Undo the function mapping defined by
        :meth:`~Ansatz.set_objective_map`."""
        self.objective_map_input = None

    def set_initial_state(self, function, kwargs=None):
        """Define the initial quantum state.

        :param function: A function that returns a local partition of the quantum state vector :math:`|\psi_0\\rangle_\\text{ANZ}`.
        :type function: callable

        :type kwargs: Keyword arguments passed to the function.
        :param kwargs: optional, dictionary, default = None
        """

        if kwargs is None:
            kwargs = {}

        self.initial_state_input = [function, kwargs]

        self.setup_initial_state = True

    def __parse_initial_state_function(self):

        self.initial_state_function = interface(
            [self, self.unitaries],
            self.initial_state_input[0],
            "initial state",
            self.COMM_OPT,
        )

    def set_sampling(self, sample_block_size, function = None, max_sample_iterations = 100, kwargs = None):
        """ Calculate the QVA objective funtion using simulated sampling. Samples are taken in blocks of `sample_block_size`.
        These are passed as a list of lists to `function`, which returns a value for :math:`\langle Q \\rangle` and a boolean
        that indicates wether the sampled result should be passed to the classical optimiser.

        If `function` is `None`, :math:`\langle Q \\rangle` is computed as the mean of `sample_block_size` size shots.

        :param sample_block_size: Number of shots taken between sucessive computation of :math:`\langle Q \\rangle`.
        :type sample_block_size: integer

        :param function: Callable that accepts a list of lists of sample values and returns :math:`\langle Q \\rangle` along with a boolean that indicates wether to pass :math:`\langle Q \\rangle` to the classical optimiser.
        :type function: callable, optional

        :param max_sample_iterations: Maximum number of sample blocks per :math:`\langle Q \\rangle`, if exceeded :math:`\langle Q \\rangle` is passed to the classical optimiser regardless of the `function` boolean output.
        :type max_sample_iterations: optional, default = 100

        :param kwargs: Dictionary of keyword arguments for `function`.
        :type kwargs: dictionary, optional, default = None

        """


        if kwargs is None:
            kwargs = {}

        if function is None:
            function = lambda samples: (np.mean(samples), True)

        self.sample_block_size = sample_block_size
        self.max_sample_iterations = max_sample_iterations
        self.sampling_function_input = [function, kwargs]

        if not self.setup_sampling: 
            self.pre_execution_methods.append(self.__pre_sampling)
            self.post_execution_methods.append(self.__post_sampling)

        self.setup_sampling = True

    def unset_sampling(self):
        self.setup_sampling = False
        self.sampling = False
        self.pre_execution_methods.remove(self.__pre_sampling)

    def __pre_sampling(self):

        self.minimum_sampled = np.inf
        self.total_shots = 0
        self.global_minimum_found = False
        self.shots_to_global_minimum = "not found"

        if self.COMM.Get_rank() == 0:
            print(f"Executing with simulated sampling.")

    def __post_sampling(self):

        if self.MPI_COMM.Get_rank() == 0:

            self.quop_result["sampling total shots"]  = self.total_shots
            self.quop_result["sampling minimum measured"] = self.minimum_sampled
            self.quop_result["sampling shots to minimum measured"] = self.shots_to_global_minimum
            self.quop_result["observables global minimum"] = self.global_minimum

    def __gen_sampling(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            self.sampling = True

            self.__parse_sampling_function()

            self.global_minimum = self.COMM_OPT.reduce(np.min(np.real(self.observables)), op = MPI.MIN)

    def __parse_sampling_function(self):

        self.sampling_function = interface(
            [self, self.unitaries],
            self.sampling_function_input[0],
            "sampling test function",
            self.COMM_OPT,
        )


    def __sample_expectation_value(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if self.COMM_OPT.Get_rank() == 0:
                self.samples = []
                self.sample_indexes = []
            else:
                self.samples = [None]
                self.sample_indexes = [None]

            for _ in range(self.max_sample_iterations):


                # Get the probability from each node in COMM_OPT
                self.__get_local_probabilities()
                total_local_probability = np.array([self.local_probabilities.sum()], dtype = np.float32)

                comm_opt_size = self.COMM_OPT.Get_size()

                if self.COMM_OPT.Get_rank() == 0:
                    process_probabilities = np.empty(comm_opt_size, dtype = np.float32)
                else:
                    process_probabilities = None
       
                self.COMM_OPT.Gather(
                        total_local_probability,
                        process_probabilities,
                        root = 0
                        )

                if self.COMM_OPT.Get_rank() == 0:

                    rank_samples = np.random.choice(
                            [i for i in range(comm_opt_size)],
                            size = self.sample_block_size,
                            replace = True,
                            p = process_probabilities
                            )

                    ranks, counts = np.unique(rank_samples, return_counts = True)

                    samples_per_rank = np.zeros(comm_opt_size, dtype = int)

                    for rank, count in zip(ranks, counts):
                        samples_per_rank[rank] = count

                else:
                    samples_per_rank = np.empty(comm_opt_size, dtype = int)

                self.COMM_OPT.Bcast(
                        samples_per_rank,
                        root = 0
                        )

                local_normed_probabilities = self.local_probabilities/self.local_probabilities.sum()

                local_samples_inds = np.random.choice(
                        [i for i in range(self.local_i)],
                        size = samples_per_rank[self.COMM_OPT.Get_rank()],
                        replace = True,
                        p = local_normed_probabilities,
                        ).astype(np.int32)

                local_samples = np.real(self.observables[local_samples_inds]).astype(np.float64)
        
                if self.COMM_OPT.Get_rank() == 0:
                    self.samples.append(np.empty(self.sample_block_size, dtype = np.float64))
                    self.sample_indexes.append(np.empty(self.sample_block_size, dtype = np.int32))

                self.COMM_OPT.Gatherv(
                        local_samples,
                        [self.samples[-1], samples_per_rank],
                        0)

                self.COMM_OPT.Gatherv(
                        local_samples_inds + self.local_i_offset,
                        [self.sample_indexes[-1], samples_per_rank],
                        0)

                if self.COMM_OPT.Get_rank() == 0:
                    self.sampling_function.update_parameters()
                    sampling_function_result = self.sampling_function.call()
                    self.total_shots += len(self.samples[-1])
                else:
                    sampling_function_result = None

                expectation, sample_test = self.COMM_OPT.bcast(
                        sampling_function_result,
                        root = 0,
                        )

                if self.COMM_OPT.Get_rank() == 0:
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


    def set_log(self, filename, label, action="a"):

        """Creates a CSV in which to save key QAOA results after a call to
        :meth:`~Ansatz.execute`.

        :param filename: Name of the CSV file.
        :type filename: string

        :param label: User-set identifier of the currently defined ansatz.
        :type label: string

        :param action: "a", append. "w", over-write.
        :type action: string, optional, default = "a"

        By default the following information is recorded:

        * label: User-defined system label.
        * p: :math:`p`.
        * objective_function: Final result of objective function minimization.
        * objective_evaluations: Number of objective function evaluations needed during optimisation.
        * optimization_success: If the minimizer converged to its target tolerances.
        * state_norm: Norm of the final state. This should always equal 1 (within the limits of double precision accuracy).
        * simulation_time: In-program simulation time.
        * MPI_nodes: Number of MPI processes.
        """

        self.filename = filename
        self.label = label
        self.log_action = action

        self.repeat = 1  # needed if logging results from the execute method

        self.setup_log = True

    def set_seed(self, seed):
        """Define an integer that is used to set the state of initial parameter
        :math:`\\theta` generation.

        :param seed: Sets the random seed.
        :type seed: integer
        """
        self.seed = seed

    def get_expectation_value(self):
        """
        :math:`\langle \hat{Q} \\rangle =  \langle \\boldsymbol{\\theta} | \hat{Q} | \\boldsymbol{\\theta} \\rangle`

        :return: The expectation value of the quality operator, returned to all MPI nodes.
        :rtype: float
        """

        if self.colours[self.COMM.Get_rank()] == 0:
            return self.__get_expectation_value()

    def objective(self, variational_parameters):
        """
        :math:`f(\\boldsymbol{\\theta}) = \langle \\boldsymbol{\\theta} | \hat{Q} | \\boldsymbol{\\theta} \\rangle` \
        - the function minimised by the calssical optimizer.


        :param variational_parameters: An array of length :math:`|\\boldsymbol{\\theta}| D`.
        :type variational_parameters: float, array

        """

        if self.colours[self.COMM.Get_rank()] == 0:
            return self.__objective(variational_parameters)

    def set_parallel(self, parallel="global", method = "forward", h=1.5e-8):

        self.jac_method = method
        self.parallel = parallel
        self.h = h
        self.reset = True

    def __check_comm_size(self):

        # if the number of MPI processes is greater than
        # system_size, resize the communicator

        busy_comm = False


        if self.colours[self.COMM.Get_rank()] != -1:

            if self.system_size // self.COMM_OPT.Get_size() == 0:

                newsize = self.system_size // 2

            else:

                newsize = 0

        else:

            newsize = 0

        newsize = self.COMM.allreduce(newsize, op=MPI.MAX)

        if newsize > 0:

            (
                self.colours,
                self.COMM_OPT,
                self.COMM_JAC,
                self.jac_ranks,
            ) = shrink_communicator(
                newsize,
                self.colours,
                self.COMM,
                self.COMM_OPT,
                self.COMM_JAC,
                self.jac_ranks,
            )

        while not busy_comm:

            if self.colours[self.COMM.Get_rank()] != -1:

                for unitary in self.unitaries:
                    if unitary.planner:
                        self.planner = unitary
                        self.planner._Unitary__plan(self.system_size, self.COMM_OPT)
                        break
                else:
                    self.planner = self.unitaries[0]
                    self.planner._Unitary__plan(self.system_size, self.COMM_OPT)

                self.alloc_local = self.planner.alloc_local
                self.local_i = self.planner.local_i
                self.local_i_offset = self.planner.local_i_offset
                self.final_state = self.planner.final_state
                self.partition_table = self.planner.partition_table

                if self.local_i == 0:
                    empty_rank = 1
                else:
                    empty_rank = 0

                empty_ranks = self.COMM_OPT.allreduce(empty_rank, op=MPI.SUM)

                if empty_ranks == 0:
                    newsize = 0
                else:
                    self.planner.destroy()
                    newsize = self.COMM_OPT.Get_size() - empty_ranks

            else:

                newsize = 0

            newsize = self.COMM.allreduce(newsize, op=MPI.MAX)

            if newsize > 0:

                (
                    self.colours,
                    self.COMM_OPT,
                    self.COMM_JAC,
                    self.jac_ranks,
                ) = shrink_communicator(
                    newsize,
                    self.colours,
                    self.COMM,
                    self.COMM_OPT,
                    self.COMM_JAC,
                    self.jac_ranks,
                )

            else:

                busy_comm = True

    def __get_nodes_dict(self):

        node_ID = MPI.Get_processor_name()

        if self.COMM.Get_rank() == 0:

            node_IDs = [node_ID]

            for i in range(1, self.COMM.Get_size()):
                node_IDs.append(self.COMM.recv(source=i))

            unique_IDs = list(set(node_IDs))
            rank_lists = [[] for _ in range(len(unique_IDs))]
            self.nodes_dict = dict(zip(unique_IDs, rank_lists))

            for i, ID in enumerate(node_IDs):
                self.nodes_dict[ID].append(i)

            self.nodes_dict = {
                k: self.nodes_dict[k]
                for k in sorted(self.nodes_dict.keys(), key=lambda x: self.nodes_dict[x][0])
            }

        else:

            self.COMM.send(node_ID, dest=0)
            self.nodes_dict = None

        self.nodes_dict = self.COMM.bcast(self.nodes_dict)
        self.n_nodes = len(self.nodes_dict)

    def __gen_communication_topology(self, processes, nodes):

        n_nodes = len(self.nodes_dict)
        nodes = 1 if nodes is None else nodes
        nodes = n_nodes if nodes > n_nodes else nodes
        processes = self.COMM.Get_size() if processes > self.COMM.Get_size() else processes

        if (len(self.nodes_dict) == nodes) and (processes == self.COMM.Get_size()):
            self.COMM_OPT = MPI.Comm.Dup(self.COMM)
            self.MPI_COMM = self.COMM_OPT
            self.colours = self.COMM.Get_size()*[0]
            self.parallel = "global"

        else:

            n_subcomms = n_nodes // nodes

            for key in list(self.nodes_dict.keys())[nodes * n_subcomms : n_nodes]:
                self.nodes_dict.pop(key)

            for key in self.nodes_dict.keys():
                while len(self.nodes_dict[key]) > processes:
                    self.nodes_dict[key].pop()

            self.comm_opt_mapping = [[] for i in range(n_subcomms)]

            for i, key in enumerate(self.nodes_dict.keys()):
                self.comm_opt_mapping[i % n_subcomms] += self.nodes_dict[key]

            self.comm_opt_roots = [min(m) for m in self.comm_opt_mapping]

            #self.var_map = [[] for _ in range(n_subcomms)]
            #for var in range(n_variational_parameters):
            #    self.var_map[var % n_subcomms].append(var)

            self.colours = np.full(self.COMM.Get_size(), -1, dtype=int)

            for i, m in enumerate(self.comm_opt_mapping):
                self.colours[m] = i

            self.jac_ranks = [ranks for m in self.comm_opt_mapping[1:] for ranks in m]
            self.jac_ranks.insert(0, 0)

            if self.colours[self.COMM.Get_rank()] != -1:

                self.COMM_OPT = MPI.Comm.Split(
                    self.COMM, self.colours[self.COMM.Get_rank()], self.COMM.Get_rank()
                )
                self.MPI_COMM = self.COMM_OPT

            world_group = MPI.Comm.Get_group(self.COMM)
            jac_group = MPI.Group.Incl(world_group, self.jac_ranks)
            self.COMM_JAC = self.COMM.Create_group(jac_group)

    def __update_var_map(self):

        if self.parallel != "global":
            self.var_map = [[] for _ in range(len(self.comm_opt_roots))]
            if self.colours[self.COMM.Get_rank()] != -1:
                for var in range(self.n_variational_parameters):
                    self.var_map[1:][var % (len(self.comm_opt_roots) - 1)].append(var)
        else:
            self.var_map = None

    def __gen_parallel(self):

        self.__get_nodes_dict()

        #n_variational_parameters = len(self.variational_parameters)

        if isinstance(self.parallel, str):

            if self.parallel in ["jacobian", "jacobian_local", "global"]:

                if self.parallel == "global":

                    processes = self.COMM.Get_size()
                    nodes = len(self.nodes_dict)

                elif self.parallel in ["jacobian", "jacobian_local"]:

                    nodes = 1
                    processes = np.min([len(self.nodes_dict[key]) for key in self.nodes_dict.keys()])
                 
                else:

                    QUOP_ERR = (
                        "Rank {}: Parallel scheme '{}' not recognised. Options are"
                        " 'jacobian', 'jacobian_local' and 'global'.".format(
                            self.COMM.Get_rank(), self.parallel
                            )
                        )

                    raise ValueError(QUOP_ERR)

        elif isinstance(self.parallel, tuple):
            processes, nodes = self.parallel
            if nodes is None:
                self.nodes_dict = {i // processes : [] for i in range(self.COMM.Get_size()) }
                if self.COMM.Get_size() == 1:
                    self.parallel = "global"
                else:
                    for i in range(self.COMM.Get_size()):
                        self.nodes_dict[i // processes] +=  [i]
                    self.parallel = "jacobian"

        self.__gen_communication_topology(processes, nodes)

    def __gen_unitaries(self):

        if self.colours[self.COMM.Get_rank()] != -1:
            for unitary in self.unitaries:
                if unitary is not self.planner:
                    unitary._Unitary__copy_plan(self.planner)

            for i, unitary in enumerate(self.unitaries):

                if unitary.operator_n_params == 0:
                    unitary.gen_operator()

                unitary.seed = self.seed + i

    def __gen_depth(self):

        self.n_variational_parameters = self.total_params * self.ansatz_depth

    def __gen_observable_map(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if self.observable_map_input is not None:
                self.__parse_observable_mapping()

    def __gen_initial_state(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if self.initial_state_input is None:
                from .state import equal

                self.set_initial_state(equal)

            self.__parse_initial_state_function()

            self.ansatz_initial_state = self.initial_state_function.call(
                **self.initial_state_input[1]
            )

    def __gen_observables(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if callable(self.observable_input[0]):

                self.parsed_observable_function = interface(
                    [self],
                    self.observable_input[0],
                    "observable",
                    self.COMM_OPT,
                )

                kwargs = self.observable_input[1]
                self.observables = self.parsed_observable_function.call(**kwargs)

                if self.observables.shape[0] != self.local_i:
                    self.observables = np.reshape(a, (self.local_i,))

            else:

                unitary = self.unitaries[self.observable_input[0]]

                if unitary.unitary_type == "diagonal":
                    self.observables = unitary.operator
                else:
                    RuntimeError(
                        "Rank {}: Cannot identify observables, no diagonal"
                        " unitary defined".format(self.COMM.Get_rank())
                    )

    def __gen_optimiser(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if self.optimiser is None:
                self.set_optimiser(
                    "scipy",
                    {"method": "BFGS", "options": {"gtol": 1e-3}},
                    ["fun", "nfev", "success"],
                )

            if self.jacobian_input is not None:
                self.__parse_jacobian()

    def __gen_objective_map(self):

        if self.COMM.Get_rank() == 0:
            if self.objective_map_input is not None:
                self.__parse_objective_mapping()

    def setup(self):

        if self.reset and not self.setup_called:
            self.seed += 1

            self.__gen_parallel()

            self.__check_comm_size()
            self.__gen_unitaries()

            self.setup_depth = True
            self.setup_observables = True
            self.setup_initial_state = True
            self.setup_observable_map = False
            self.setup_objective_map = False
            self.setup_optimiser = True

            self.reset = False
            self.setup_called = True

    def __post_log(self):

        if self.COMM.rank == 0:
            if self.log:
                self.logfile.close()

    def __post_unitaries(self):

        if self.colours[self.COMM.Get_rank()] != -1:
            for unitary in self.unitaries:
                if unitary.planned:
                    unitary.destroy()

    def __post_parallel(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if self.COMM_JAC is not None:

                MPI.Comm.Free(self.COMM_OPT)

                if self.COMM.Get_rank() in self.jac_ranks:
                    MPI.Comm.Free(self.COMM_JAC)

    def destroy(self):

        if self.reset and self.setup_called:

            if not self.benchmarking:
                if self.log:
                    self.__post_log()

            if not self.setup_unitaries:
                self.__post_unitaries()
                self.setup_unitaries = True

            if not self.setup_parallel:
                self.__post_parallel()
                self.setup_parallel = True

    def evolve_state(self, x):

        self.destroy()
        self.setup()
        self.__pre()

        self.__evolve_state(x)

        self.__post()

    def __evolve_state(self, x):
        """Compute :math:`U(\\boldsymbol{\\theta})_\\text{ANZ}|\psi_0\\rangle_\
        \text{ANZ}`.

        :param x: :math:`|\\boldsymbol{\\theta}| D` variational parameters.
        :type x: array, float
        """

        if isinstance(x, list):
            x = np.array(x, dtype=np.float64)

        if self.colours[self.COMM.Get_rank()] != -1:

            self.final_state[: self.local_i] = self.ansatz_initial_state[: self.local_i]

            params_split = np.split(x, self.ansatz_depth)

            for depth, params in enumerate(params_split):

                for i, unitary in enumerate(self.unitaries):

                    param_slice = params[self.param_map[i] : self.param_map[i + 1]]

                    if unitary.operator_n_params > 0:
                        evolution_parameter = param_slice[: -unitary.operator_n_params]
                        unitary.variational_parameters = param_slice[
                            unitary.unitary_n_params : :
                        ]

                        unitary.gen_operator()

                    else:
                        evolution_parameter = param_slice

                    unitary.initial_state[: self.local_i] = self.final_state[
                        : self.local_i
                    ]

                    unitary.propagate(evolution_parameter)

                    self.final_state[: self.local_i] = unitary.final_state[
                        : self.local_i
                    ]

            if self.COMM_OPT.Get_rank() == 0:
                self.n_evolutions += 1
            self.last_evaluated = copy(x)

    def evaluate(self, x):
        # returns the expectation value given variational parameters 'x'

        if not np.array_equal(self.last_evaluated, x):
            self.__evolve_state(x)
        return self.__get_expectation_value()

    def execute(self, variational_parameters=None):
        """Execute the QVA algorithm.

        :param variational_parameters: An array of length :math:`2 |\\boldsymbol{\\theta}|`.
        :type variational_parameters: float, array
        """

        if not self.benchmarking:

            self.destroy()

            if variational_parameters is not None:
                self.variational_parameters = np.array(variational_parameters, dtype = np.float64)
                self.set_depth(len(variational_parameters) // self.total_params)

            self.setup()
            self.__pre()

            if self.variational_parameters is None:
                self.variational_parameters = self.gen_initial_params(self.ansatz_depth)
        if self.colours[self.COMM.Get_rank()] != -1:

            self.stop = False
            self.n_evolutions = 0
            if self.colours[self.COMM.Get_rank()] == 0:

                self.objective_cnt = 0

                if self.COMM_OPT.Get_rank() == 0:

                    if self.record_objective:
                        self.total_n_evolutions = []

                    self.neval_mpi_jac = 0

                    self.time = time()

                    self.result = self.optimiser(
                        self.__objective,
                        self.variational_parameters,
                        **self.optimiser_args
                    )

                    self.stop = True

                    self.objective(None)

                    if self.parallel != "global":
                        self.__mpi_jacobian(None)

                    self.time = time() - self.time

                else:

                    while not self.stop:
                        self.__objective(self.variational_parameters)

                self.__post()

                if self.log:
                    self.__log_update()

            else:
                while not self.stop:
                    self.__mpi_jacobian(None)

                self.__post()

    def print_summary(self):
        """Print a summary of the last QuOp\_MPI simulation."""
        if self.COMM.Get_rank() == 0:
            print("\nQuOp_MPI Simulatuion Summary", flush = True)
            print("============================\n", flush = True)
            for key in self.quop_result.keys():
                printkey = f'{key}:'
                print(f'{printkey:36}{self.quop_result[key]}', flush = True)
            print('')

    def print_optimiser_result(self):
        """Print the optimisation result."""
        if self.COMM.Get_rank() == 0:
            print("\nOptimisation Result", flush = True)
            print("===================\n", flush = True)
            print(self.result, flush=True)

    def benchmark(
        self,
        ansatz_depths,
        repeats,
        param_persist=False,
        verbose=True,
        filename=None,
        label="test",
        save_action="a",
        time_limit=None,
        suspend_path=None,
    ):

        """This provides an method by which to study how a QVA algorithm
        performs with increases to the circuit depth, :math:`D`.

        :param ansatz_depths: iterable of :math:`D` values.
        :type ansatz_depths: integer, array

        :param repeats: The number of repeats at each value of :math:`D`.
        :type repeats: integer

        :param param_persist: If `True` the optimized :math:`\\boldsymbol{\\theta}` values which achieved the lowest objective function value  for all repeats at :math:`D` will be used as starting parameters for :math:`D + 1`.
        :type param_persist: boolean, optional

        The following parameters specify the output behaviours.

        :param verbose: If True, print current :math:`D`, repeat number and optimisation results.
        :type verbose: boolean, optional, default = True

        :param filename: Name of .h5 file in which to :meth:`~Ansatz.system.save` the evolved system.
        :type filename: string, optional, default = None

        :param label: If filename is specified, evolved systems will be saved as 'filename/label_p_repetition'.
        :type label: string, optional, default = 'test'

        :param save_action: Action taken during first .5 file write. "a", append. "w", over-write.
        :type save_action: string, optional, default = "a"

        :param time_limit: Total allocated in-program time, entered as "HH:MM:SS". If the time of the previous simulation exceeds the time remaining, the benchmark run is suspended.
        :param type time_limit: string, optional, default = None

        :param suspend_path: If `time_limit` is defined, benchmark progress is saved to this file with a '.pkl' extension.
        :type suspend_path: string, optional, default = "QuOp_benchmark_suspend_data"


        """

        ansatz_depth_temp = deepcopy(
            self.ansatz_depth
        )  # return to this value after benchmarking

        self.benchmarking = True

        if f'{suspend_path}.pkl' is None:
            suspend_path=f"QuOp_benchmark_suspend_data"

        if time_limit is not None:

            if isinstance(time_limit, str):
                self.available_time = sum(
                        int(x) * 60**i for i, x in enumerate(
                            reversed(time_limit.split(":"))
                            )
                        )
            else:
                self.available_time = time_limit

            if self.available_time <= 0:
                return

        runs = []
        for depth in ansatz_depths:
            for i in range(1, repeats + 1):
                runs.append([depth, i])

        if (time_limit is None) or (not os.path.exists(f'{suspend_path}.pkl')):
            previous_params = None
            first = True
            runstart = 0
            runcount = 0
        else:

            if self.COMM.Get_rank() == 0:
                resume_dict = pickle.load(open(f'{suspend_path}.pkl', "rb"))
            else:
                resume_dict = None

            resume_dict = self.COMM.bcast(resume_dict)

            if resume_dict["complete"]:
                
                if self.COMM.Get_rank() == 0:
                    print("Benchmark completed.", flush = True)

                return

            runstart = resume_dict["runstart"]
            runcount = resume_dict["runstart"]
            previous_params = resume_dict["previous_params"]
            first = resume_dict["first"]
            previous_params = resume_dict["previous_params"]
            best_p_params = resume_dict["best_p_params"]
            best_p_result = resume_dict["best_p_result"]

            save_action = "a"
            self.log_action = "a"

            self.set_depth(runs[runstart][0])
            #self.__pre_or_post()
            self.destroy()
            self.setup()
            self.__pre()

        #FLAG
        #for depth in ansatz_depths:
        for depth, i in runs[runstart:]:

            start_time = time()

            if i == 1:

                self.set_depth(depth)

                self.destroy()
                self.setup()
                self.__pre()
                #self.__pre_or_post()

                if self.colours[self.COMM.Get_rank()] == 0:

        #            if param_persist:
                    best_p_result = np.finfo(dtype=np.float64).max
                    result = None

                    if verbose:
                        if self.COMM_OPT.Get_rank() == 0:
                            print("Starting depth = {}:".format(depth), flush=True)

        #    for i in range(1, repeats + 1):

            if self.colours[self.COMM.Get_rank()] == 0:

                self.repeat = i

                if (not param_persist) or (previous_params is None):

                    self.variational_parameters = self.__gen_initial_params()

                else:

                    self.variational_parameters = np.empty(depth*self.total_params, dtype = np.float64)
                    self.variational_parameters[
                        : len(previous_params)
                    ] = previous_params
                    new_params = self.__gen_initial_params(1)
                    self.variational_parameters[len(previous_params) :] = new_params

                if verbose:
                    if self.COMM_OPT.Get_rank() == 0:
                        print("{} of {}...".format(i, repeats), flush=True)

                self.execute()

                if verbose:
                    self.print_summary()

                #if param_persist:

                if self.COMM.Get_rank() == 0:
                    result = self.result["fun"]
                    x = self.result["x"]
                else:
                    result = None
                    x = None

                result = self.COMM_OPT.bcast(result, root=0)
                x = self.COMM_OPT.bcast(x, root=0)

                if result < best_p_result:
                    best_p_result = result
                    best_p_params = copy(x)

                if filename is not None:

                    if first:
                        self.save(
                            filename,
                            label + "_" + str(depth) + "_" + str(i),
                            action=save_action,
                        )
                    else:
                        self.save(
                            filename,
                            label + "_" + str(depth) + "_" + str(i),
                            action="a",
                        )

                first = False

            else:

                self.execute()

            if self.colours[self.COMM.Get_rank()] == 0:
                if i == repeats:
                    if param_persist:
                        previous_params = best_p_params

            end_time = time()

            run_time = end_time - start_time

            runcount += 1
            if time_limit is not None:

                self.available_time -= run_time

                if runcount == len(runs):
                    complete = True
                else:
                    complete = False

                if self.COMM.Get_rank() == 0:
                    resume_dict = {
                        "runstart": runcount,
                        "previous_params": previous_params,
                        "best_p_params": best_p_params,
                        "first": first,
                        "previous_params":previous_params,
                        "best_p_params":best_p_params,
                        "best_p_result":best_p_result,
                        "complete": complete,
                    }
                
                    with open(f'{suspend_path}.pkl', "wb") as f:
                        pickle.dump(resume_dict, f)
                
                if self.available_time < 2 * run_time:
                    self.destroy()
                    return

        #if time_limit is not None:
        #    if self.COMM.Get_rank() == 0:
        #        with open(f'{suspend_path}.pkl', "wb") as f:
        #            pickle.dump({"complete":True}, f)

        self.benchmarking = False
        self.ansatz_depth = ansatz_depth_temp

    def get_final_state(self):
        """Gather :math:`|\\boldsymbol{\\theta}_f \\rangle_\\text{ANZ}` at the
        root MPI rank following a call to :meth:`~Ansatz.execute`,
        :meth:`~Ansatz.evolve_state` or `~Ansatz.benchmark`. If called after
        `~Ansatz.benchmark` the gathered state will correspond to the last
        performed simulation.

        :return: :math:`|\\boldsymbol{\\theta}_f\\rangle_\\text{ANZ}`
        :rtype: array, complex
        """
        if self.colours[self.COMM.Get_rank()] != -1:
            if self.colours[self.COMM.Get_rank()] == 0:
                return gather_array(
                    self.final_state, self.unitaries[0].partition_table, self.COMM_OPT
                )

    def get_probabilities(self):
        """Gather :math:`||\\boldsymbol{\\theta}_f \\rangle_\\text{ANZ}|^2` at
        the root MPI rank following a call to :meth:`~Ansatz.execute`,
        :meth:`~Ansatz.evolve_state` or `~Ansatz.benchmark`. If called after
        `~Ansatz.benchmark` the gathered probabilities will correspond to the
        last performed simulation.

        :return: :math:`||\\boldsymbol{\\theta}_f \\rangle_\\text{ANZ}|^2`.
        :rtype: array, float
        """
        if self.colours[self.COMM.Get_rank()] != -1:
            if self.colours[self.COMM.Get_rank()] == 0:
                prob = np.abs(self.final_state) ** 2
                return gather_array(
                    prob, self.unitaries[0].partition_table, self.COMM_OPT
                )

    def save(self, file_name, config_name, action="a"):
        """Write the final state, observables and execution results of the
        current configuration to a HDf5 file.

        :param file_name: Name of the file on disc.
        :type file_name: string

        :param file_name: Name of the saved configuration in the HDf5 file.
        :type file_name: string

        :param action: "a": append to an existing file or create a new file. "w": overwrite the file if it exists.
        :type action: string, optional

        Data is saved into a .h5 file with the following structure.

        ::

            file_name.h5
            ├── config_name
            │   ├── final_state
            │   ├── observables

        The minimization result is saved in the 'minimize_result' attribute of 'config_name' as a formatted string.

        Multiple configurations with a unique config_name can be stored in the same .h5 file.
        HDF5 files are supported in python by the `h5py <https://www.h5py.org/>`_ package. With it,
        a saved configuration can be accessed as follows:

        .. code-block:: python

            import h5py

            config_name = "my_simulation"

            f = h5py.File(file_name + ".h5", "r")
            final_state = np.array(f[config_name]['final_state']).view(np.complex128)
            eigenvalues = np.array(f[config_name]['eigenvalues']).view(np.complex128)
            observables = np.array(f[config_name]['observables']).view(np.float64)

            print(f["my_simulation"].attrs["minimize_result"])

        .. warning::
            The final_state and observables datasets are saved using Fortran
            subroutines which make use of parallel HDF5.

            The complex values of the final_state array are saved as a
            compound datatype consisting of contiguous double precision reals. This is equivalent to the np.complex128 NumPy datatype. To access this data without a
            loss of precision in python, the user must set the **view** of the NumPy array to np.complex128, rather than casting it to np.complex128 using the dtype keyword.

            Similarly, the observables array, which is saved as an array of double-precision
            reals, should have its view set to np.float64.
        """

        if self.colours[self.COMM.Get_rank()] == 0:

            from quop_mpi.__lib import fqwoa_mpi

            if self.COMM_OPT.Get_rank() == 0:

                import h5py

                self.config_name = config_name

                File = h5py.File(file_name + ".h5", action)

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
                    self.config_name + "/initial_phases",
                    data=self.variational_parameters,
                    dtype=np.float64,
                )
                File.close()
            else:
                self.config_name = None

            self.config_name = self.COMM_OPT.bcast(self.config_name, root=0)

            fqwoa_mpi.save_dist_complex(
                file_name,
                self.config_name + str("/"),
                "final_state",
                "a",
                self.system_size,
                self.local_i_offset,
                self.final_state[: self.local_i],
                self.COMM_OPT.py2f(),
            )

            fqwoa_mpi.save_dist_complex(
                file_name,
                self.config_name + str("/"),
                "initial_state",
                "a",
                self.system_size,
                self.local_i_offset,
                self.ansatz_initial_state[: self.local_i],
                self.COMM_OPT.py2f(),
            )

            fqwoa_mpi.save_dist_real(
                file_name,
                self.config_name + str("/"),
                "observables",
                "a",
                self.system_size,
                self.local_i_offset,
                self.observables[: self.local_i],
                self.COMM_OPT.py2f(),
            )

    def gen_initial_params(self, ansatz_depth=None):
        """Generate :math:`\\boldsymbol{\\theta}` using the parameter functions
        assocaited with each `unitary`.

        :param ansatz_depth: The number of ansatz iterations or quantum circuit 'depth' :math:`D`.
        :type ansatz_depth: optional, integer, default=None

        :return: :math:`|\\boldsymbol{\\theta}| D` variational parameters :math:`\\boldsymbol{\\theta}` where :math:`D` is defined by either the `ansatz_depth` argument or the current value of `Ansatz.ansatz_depth` (see :meth:`~Ansatz.set_depth`).
        :rtype: array, float
        """

        if ansatz_depth is None:
            params = self.__gen_initial_params()
        else:
            params = self.__gen_initial_params(ansatz_depth)

        if self.COMM.Get_rank() == 0:
            n_params = len(params)
        else:
            n_params = None

        n_params = self.COMM.bcast(n_params, 0)

        if self.colours[self.COMM.Get_rank()] != 0:
            params = np.empty(n_params, dtype=np.float64)

        self.COMM.Bcast([np.array(params, dtype = np.float64), MPI.DOUBLE], 0)

        return params

    def __gen_initial_params(self, ansatz_depth=None):

        if self.colours[self.COMM.Get_rank()] == 0:

            if ansatz_depth is None:
                ansatz_depth = self.ansatz_depth

            params = np.zeros(ansatz_depth * self.total_params, dtype=np.float64)

            if self.COMM.Get_rank() == 0:

                param_iterations = np.split(params, ansatz_depth)

                for param_iters in param_iterations:
                    for i, unitary in enumerate(self.unitaries):
                        unitary.seed += i + 1
                        param_iters[
                            self.param_map[i] : self.param_map[i + 1]
                        ] = unitary.gen_initial_params()

            self.COMM_OPT.Bcast([params, MPI.DOUBLE], 0)

            return params

    def __get_local_probabilities(self):

        self.local_probabilities = (
            np.abs(self.final_state[: self.local_i], dtype=np.float64) ** 2
        )
        return self.local_probabilities

    def __get_state_norm(self):

        if self.colours[self.COMM.Get_rank()] == 0:
            self.state_norm = self.COMM_OPT.allreduce(
                np.sum(self.__get_local_probabilities()), op=MPI.SUM
            )
            return self.state_norm

    def __get_expectation_value(self):

        # TEMP
        self.observables = self.unitaries[0].operator

        if self.sampling:
            return self.__sample_expectation_value()

        else:

            self.__get_local_probabilities()

            if self.observable_map_input is not None:

                local_expectation = np.dot(
                    self.local_probabilities,
                    self.observable_map(self.observables, **self.observable_map_input[1]),
                )
            else:

                local_expectation = np.dot(self.local_probabilities, self.observables)

            return np.real(self.COMM_OPT.allreduce(local_expectation, op=MPI.SUM))

    def __objective(self, variational_parameters):

        self.stop = self.COMM_OPT.bcast(self.stop, root=0)

        if not self.stop:

            self.variational_parameters = self.COMM_OPT.bcast(
                variational_parameters, root=0
            )

            self.__evolve_state(self.variational_parameters)

            self.expectation = self.get_expectation_value()

            if self.COMM.Get_rank() == 0:

                if self.objective_map_input is not None:
                    self.objective_map.update_parameters()
                    self.expectation = self.objective_map.call(
                        **self.objective_map_input[1]
                    )

                if self.verbose_objective:

                    self.objective_cnt += 1

                    print(
                        "Call # {}, f(x) = {}".format(
                            self.objective_cnt, self.expectation
                        ),
                        flush=True,
                    )

                if self.record_objective:
                    expectation = deepcopy(self.expectation)
                    self.objective_history.append(expectation)

                if self.record_objective:
                    self.total_n_evolutions.append(self.n_evolutions)
                return self.expectation

    def __gen_log(self):

        if self.colours[self.COMM.Get_rank()] == 0:

            self.n_log_fields = 6

            if self.COMM.Get_rank() == 0:
                if os.path.exists(self.filename + ".csv") and self.log_action == "a":
                    self.logfile = open(self.filename + ".csv", "a", newline="")
                    self.logfile_csv = csv.writer(self.logfile)
                else:

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
                        headings.append("total_shots")
                        headings.append("minimum_sampled")
                        headings.append("shots_to_global_minimum")

                    if self.optimiser_log is not None:
                        for optimiser_log in self.optimiser_log:
                            headings.append(optimiser_log)

                    self.logfile = open(self.filename + ".csv", "w")
                    self.logfile_csv = csv.writer(self.logfile)
                    self.logfile_csv.writerow(headings)

        self.log = True

    def __log_update(self):

        if self.COMM.Get_rank() == 0:

            log_output = [
                self.label,
                self.system_size,
                self.ansatz_depth,
                self.repeat,
                self.state_norm,
                self.time,
                self.COMM.size,
                self.neval_mpi_jac,
            ]

            if self.sampling:
                log_output.append(self.total_shots)
                log_output.append(self.minimum_sampled)
                log_output.append(self.shots_to_global_minimum)

            if self.optimiser_log is not None:
                for optimiser_log in self.optimiser_log:
                    log_output.append(self.result[optimiser_log])

            self.logfile_csv.writerow(log_output)

            self.logfile.flush()

    def __mpi_jacobian(self, x):

        self.COMM_JAC.barrier()
        self.stop = self.COMM_JAC.bcast(self.stop, 0)

        if self.stop:
            self.COMM_JAC.barrier()
            return

        self.variational_parameters = self.COMM_JAC.bcast(x, 0)

        partials = []
        if self.COMM.Get_rank() != 0:
            for var in self.var_map[self.colours[self.COMM.Get_rank()]]:
                self.jacobian.update_parameters()
                partials.append(self.jacobian.call(var))

        opt_root = self.comm_opt_roots[self.colours[self.COMM.Get_rank()]]

        if self.COMM.Get_rank() == 0:
            jacobian = np.zeros(len(self.variational_parameters), dtype=np.float64)
            reqs = []
            for root, mapping in zip(self.comm_opt_roots, self.var_map):
                if root > 0:
                    for var in mapping:
                        self.COMM.Recv(
                            [jacobian[var : var + 1], MPI.DOUBLE], source=root, tag=var
                        )

        elif self.COMM_OPT.Get_rank() == 0:
            reqs = []
            jacobian = None
            for part, mapping in zip(
                partials, self.var_map[self.colours[self.COMM.Get_rank()]]
            ):
                self.COMM.Send([np.array([part]), MPI.DOUBLE], dest=0, tag=mapping)
        else:
            jacobian = None

        self.COMM_JAC.barrier()

        if self.record_objective:
            if self.COMM_JAC.Get_rank() == 0:
                self.n_evolutions = self.COMM_JAC.reduce(
                    self.n_evolutions, op=MPI.SUM, root=0
                )
            else:
                self.COMM_JAC.reduce(self.n_evolutions, op=MPI.SUM, root=0)
                self.n_evolutions = 0

        if self.COMM.Get_rank() == 0:

            self.neval_mpi_jac += 1
            return jacobian

        else:
            return None

    def __is_zero(self, x):
        return (x >= -np.finfo(np.float64).eps) and (x <= np.finfo(np.float64).eps)
