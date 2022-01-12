import os
import csv
import atexit
from warnings import warn
from copy import copy, deepcopy
from time import time
import numpy as np
from mpi4py import MPI
from .__utils.__interface import interface
from .__utils.__mpi import shrink_communicator, gather_array
    
def forward_differences(variational_parameters, var, evaluate):
    x = variational_parameters
    h = 1.4901161193847656e-08
    expectation = evaluate(x)
    x[var] += h
    expectation_forward = evaluate(x)
    return (expectation_forward - expectation)/h
    
def central(variational_parameters, var, evaluate):
    x = variational_parameters
    h = 1.4901161193847656e-08
    expectation = evaluate(x)
    x_back = copy(x)
    x_forward = copy(x)
    x_back[var] -= h
    x_forward[var] += h
    expectation_back = evaluate(x_back)
    expectation_forward = evaluate(x_forward)
    return  (expectation_forward - 2 * self.expectation + expectation_back)/ h

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
        self.jacobian_input = None # for parallel jacobian evaluation
        self.var = None  # for parallel jacobian evaluation
        self.benchmarking = False  # indicates whether the benchmark method is running
        self.last_evaluated = np.empty(0) # last set of variational parameters passed to 'evolve_state'.

        self.pre_called = False
        self.post_called = False
        self.call_post = False

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

        self.seed = 0

        self.initial_state_parameters = [
            "partition_table",
            "system_size",
            "alloc_local",
            "local_i",
            "local_i_offset",
            "MPI_COMM",
        ]

        self.observable_map_parameters = [
            "observables",
            "system_size",
            "MPI_COMM",
            "partition_table",
        ]

        self.observable_parameters = [
            "partition_table",
            "system_size",
            "local_alloc",
            "local_i",
            "local_i_offset",
            "seed",
            "MPI_COMM",
        ]

        self.objective_map_parameters = [
            "expectation",
        ]
            
        self.jacobian_parameters = [
            "evaluate",
            "var",
            "variational_parameters",
            "system_size",
            "seed",
            "MPI_COMM",
        ]

        atexit.register(self.__exit)

    def __exit(self):
        if self.pre_called:
            self.post()
        self.COMM.barrier()

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

        self.setup_unitaries = True

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
            self.optimsier = optimiser

        self.optimiser_args = optimiser_args
        self.optimiser_log = optimiser_log

        if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
            if "jac" in optimiser_args:
                self.jacobian_input = [copy(self.optimiser_args["jac"])]
                self.optimiser_args["jac"] = self.__mpi_jacobian
            else:
                self.jacobian_input = [forward_differences]
                self.optimiser_args["jac"] = self.__mpi_jacobian
               
        self.setup_optimiser = True
        
    def __parse_jacobian(self):
        
        self.jacobian = interface(
            self,
            self.jacobian_input[0],
            self.jacobian_parameters,
            "jacobian",
            self.COMM_OPT
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
            self,
            self.observable_map_input[0],
            self.observable_map_parameters,
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
            self,
            self.objective_map_input[0],
            self.objective_map_parameters,
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
            self,
            self.initial_state_input[0],
            self.initial_state_parameters,
            "initial state",
            self.COMM_OPT,
        )

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

    def __check_setup_validity(self):
        """This method is called by execute, it determines whether the object
        needs to update any attributes or re-configure MPI communicators.

        First it checks the 'set' methods, each of these have an associated flag starting
        with 'setup'. \e.g. 'setup_initial_state = True' indicates that the initial state
        function should be re-parsed and called. For every 'set' method the flag is
        set from False to True as the last action of that method.

        Depending on the flags there are a number of actions that may or may not be called
        by the pre method.

        List of method --> setup flags:

            set_unitaries --> self.setup_unitaries
            set_depth --> setup_depth
            set_parallel --> setup_parallel
            set_unitaries --> setup_unitaries
            set_observables --> setup_observables
            set_initial_state --> setup_initial_state
            set_observable_map --> setup_observable_map
            set_objective_map --> setup_objective_map
            set_log --> setup_log
            set_optimiser --> setup_optimiser
        """

        reset = False

        if self.setup_depth and (self.parallel in ["jacobian", "jacobian_local"]):

            self.setup_parallel = True
            reset = True

        elif self.setup_depth:

            self.setup_initial_parameters = True
            reset = True

        if self.setup_parallel:

            self.setup_depth = True
            self.setup_parallel = True
            reset = True

        if self.setup_unitaries:

            self.setup_initial_parameters = True
            self.setup_parallel = True
            reset = True

        if self.setup_parallel:

            self.setup_unitaries = True
            self.setup_observables = True
            self.setup_initial_state = True
            self.setup_observable_map = True
            self.setup_objective_map = True
            self.reset = True

        if self.setup_log:
            reset = True

        if reset:

            if self.pre_called:
                self.call_post = True

            self.pre_called = False

    def __pre_or_post(self):

        if self.pre_called:

            self.__check_setup_validity()

            if self.call_post:
                self.post()

            if not self.pre_called:
                self.pre()

        elif not self.pre_called:
            self.pre()

    def set_parallel(self, parallel="global", method="forward", tol=1e-12):
        """Define the parallel scheme. Also defines the step-size and type of
        gradient approximation used if parallel evaluation of the objective
        function gradient is selected.

        Types of parallelisation methods include:

        * "global": The global MPI communicator simulates :math:`|\\boldsymbol{\\theta} \\rangle_\\text{ANZ}`.
        * "jacobian": The global MPI communicator is partitioned into a sub-communicator calculating :math:`|\\boldsymbol{\\theta} \\rangle_\\text{ANZ}` and sub-communicators calculating terms in the objective function gradient.
        * "jacobian-local": Create MPI sub-communicators as described above, but group MPI ranks based on CPU ID.

        Type of gradient approximations include:

        * "forward": finite difference approximation with using forward differences.
        * "center": finite difference approximation using central differences.

        :param parallel: Parallelisation scheme.
        :type parallel: optional, string, default = "global"

        :param method: Method used for gradient approximation in the case of "jacobian" or "jacobian-local" parallel schemes.
        :type method: optional, string, default = "forward"

        :param tol: Step size used during finite difference approximation of the objective function gradient.
        :type tol: optional, float, default = 1e-12
        """

        if parallel in ["jacobian", "jacobian_local", "global"]:

            self.parallel = parallel  # type of MPI parallelisation: "global", "jacobian" or "jacobian_local"
            self.jac_method = method
            self.jac_tol = tol

            if (parallel in ["jacobian", "jacobian_local"]) and (
                self.COMM.Get_size() == 1
            ):
                QUOP_ERR = (
                    "Rank {}: Parallel setting '{}' requires an MPI"
                    " communicator size greater than 1.".format(
                        self.COMM.Get_rank(), parallel
                    )
                )
                raise RuntimeError(QUOP_ERR)

        else:
            QUOP_ERR = (
                "Rank {}: Parallel scheme '{}' not recognised. Options are"
                " 'jacobian', 'jacobian_local' and 'global'.".format(
                    self.COMM.Get_rank(), parallel
                )
            )
            raise ValueError(QUOP_ERR)

        self.setup_parallel = True

    def __gen_parallel(self):

        busy_comm = False

        # parallel Jacobian not possible with one MPI process
        if self.COMM.Get_size() == 1:
            self.parallel = "global"

        # set up communication topology
        #   if self.COMM_OPT is None:

        if self.parallel == "global":
            self.COMM_OPT = MPI.Comm.Dup(self.COMM)
            self.MPI_COMM = self.COMM_OPT
            self.colours = [0] * self.COMM.Get_size()

        elif (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):

            self.__parallel_jacobian_communication_topology()
            self.n_jacobian_variables = len(self.variational_parameters)
            self.MPI_COMM = self.COMM_OPT

            # if the number of MPI processes is greater than
            # system_size, resize the communicator

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
        self.variational_parameters = np.empty(
            self.total_params * self.ansatz_depth, np.float64
        )

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
                    self,
                    self.observable_input[0],
                    self.observable_parameters,
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
                    {"method": "BFGS", 'options':{'gtol':1e-3}},
                    ["fun", "nfev", "success"],
                )
                
            if self.jacobian_input is not None:
                self.__parse_jacobian()

    def __gen_objective_map(self):

        if self.COMM.Get_rank() == 0:
            if self.objective_map_input is not None:
                self.__parse_objective_mapping()

    def pre(self):

        self.seed += 1

        if self.setup_depth:
            self.__gen_depth()
            self.setup_depth = False

        if self.setup_parallel:
            self.__gen_parallel()
            self.setup_parallel = False

        if self.setup_unitaries:
            self.__gen_unitaries()
            self.setup_unitaries = False

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

        if self.setup_log:
            self.__gen_log()
            self.setup_log = False

        self.pre_called = True
        self.call_post = False

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


    def post(self):

        if not self.benchmarking:
            if self.log:
                self.__post_log()

        if not self.setup_unitaries:
            self.__post_unitaries()

        if not self.setup_parallel:
            self.__post_parallel()

        self.post_called = True
        self.pre_called = False

    def evolve_state(self, x):
        """Compute :math:`U(\\boldsymbol{\\theta})_\\text{ANZ}|\psi_0\\rangle_\
        \text{ANZ}`.

        :param x: :math:`|\\boldsymbol{\\theta}| D` variational parameters.
        :type x: array, float
        """

        #TODO: handle call to evolve state where the depth as not been previously set.

        self.__pre_or_post()

        if isinstance(x,list):
            x = np.array(x, dtype = np.float64)

        if self.colours[self.COMM.Get_rank()] != -1:

            self.final_state[: self.local_i] = self.ansatz_initial_state[: self.local_i]

            params_split = np.split(x, self.ansatz_depth)

            for depth, params in enumerate(params_split):

                for i, unitary in enumerate(self.unitaries):

                    param_slice = params[self.param_map[i] : self.param_map[i + 1]]

                    if unitary.operator_n_params > 0:
                        evolution_parameter = param_slice[:-unitary.operator_n_params]
                        unitary.variational_parameters = param_slice[
                                unitary.unitary_n_params::
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
                self.last_evaluated = x
                
    def evaluate(self, x):
        # returns the expectation value given variational parameters 'x'
        if not np.array_equal(self.last_evaluated, x):
            self.evolve_state(x)
        return self.__get_expectation_value()
        
    def execute(self, variational_parameters=None):
        """Execute the QVA algorithm.

        :param variational_parameters: An array of length :math:`2 |\\boldsymbol{\\theta}|`.
        :type variational_parameters: float, array
        """

        if not self.benchmarking:

            if variational_parameters is not None:

                if (
                    not len(variational_parameters)
                    == self.ansatz_depth * self.total_params
                ):
                    self.set_depth(len(variational_parameters) // self.total_params)

                self.__pre_or_post()

                if self.colours[self.COMM.Get_rank()] != -1:

                    self.variational_parameters[:] = np.array(
                        variational_parameters, dtype=np.float64
                    )

            else:
                self.__pre_or_post()
                self.variational_parameters = self.__gen_initial_params(
                    self.ansatz_depth
                )

        if self.colours[self.COMM.Get_rank()] != -1:

            self.stop = False
            self.n_evolutions = 0

            if self.colours[self.COMM.Get_rank()] == 0:

                self.objective_cnt = 0

                if self.COMM_OPT.Get_rank() == 0:

                    if self.record_objective:
                        self.objective_history = []
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

                    if (self.parallel == "jacobian") or (
                        self.parallel == "jacobian_local"
                    ):
                        self.__mpi_jacobian(None)

                    self.time = time() - self.time

                else:

                    while not self.stop:
                        self.__objective(self.variational_parameters)

                if self.log:
                    self.__log_update()

            else:

                while not self.stop:
                    self.__mpi_jacobian(None)

    def print_optimiser_result(self):
        """Print the optimisation result."""
        if self.COMM.Get_rank() == 0:
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
        """

        ansatz_depth_temp = deepcopy(
            self.ansatz_depth
        )  # return to this value after benchmarking
        previous_params = None
        self.benchmarking = True
        first = True

        for depth in ansatz_depths:

            self.set_depth(depth)

            self.__pre_or_post()

            if self.colours[self.COMM.Get_rank()] == 0:

                if param_persist:
                    best_p_result = np.finfo(dtype=np.float64).max
                    result = None

                if verbose:
                    if self.COMM_OPT.Get_rank() == 0:
                        print("Starting depth = {}:".format(depth), flush=True)

            for i in range(1, repeats + 1):

                if self.colours[self.COMM.Get_rank()] == 0:

                    self.repeat = i

                    if (not param_persist) or first or (self.ansatz_depth == 1):

                        self.variational_parameters = self.__gen_initial_params()

                    else:

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
                        self.print_optimiser_result()

                    if param_persist:

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
                if param_persist:
                    previous_params = best_p_params

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
                    self.final_state,
                    self.unitaries[0].partition_table,
                    self.COMM_OPT,
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

        self.COMM.Bcast([params, MPI.DOUBLE], 0)

        return params

    def __gen_initial_params(self, ansatz_depth=None):

        if not self.benchmarking:
            if ansatz_depth is not None:
                self.set_depth(ansatz_depth)

            self.__pre_or_post()

        if ansatz_depth is None:
            ansatz_depth = self.ansatz_depth

        if self.colours[self.COMM.Get_rank()] == 0:

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
                np.sum(self.local_probabilities), op=MPI.SUM
            )
            return self.state_norm

    def __get_expectation_value(self):

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
            self.evolve_state(self.variational_parameters)

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

                    if self.optimiser_log is not None:
                        for optimiser_log in self.optimiser_log:
                            headings.append(optimiser_log)

                    self.logfile = open(self.filename + ".csv", "w")
                    self.logfile_csv = csv.writer(self.logfile)
                    self.logfile_csv.writerow(headings)

        self.log = True

    def __log_update(self):

        self.state_norm = self.__get_state_norm()

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

            if self.optimiser_log is not None:
                for optimiser_log in self.optimiser_log:
                    log_output.append(self.result[optimiser_log])

            self.logfile_csv.writerow(log_output)

            self.logfile.flush()

    def __parallel_jacobian_communication_topology(self):

        # NOTE: here "nodes" refers to physical compute units in MPI communicator self.COMM.

        node_ID = MPI.Get_processor_name()

        if self.COMM.Get_rank() == 0:
            node_IDs = [node_ID]

            for i in range(1, self.COMM.Get_size()):
                node_IDs.append(self.COMM.recv(source=i))

            unique_IDs = list(set(node_IDs))
            rank_lists = [[] for _ in range(len(unique_IDs))]
            nodes_dict = dict(zip(unique_IDs, rank_lists))

            for i, ID in enumerate(node_IDs):
                nodes_dict[ID].append(i)

            for key in nodes_dict:
                nodes_dict[key] = np.array(nodes_dict[key])

            nodes_dict = {
                k: nodes_dict[k]
                for k in sorted(nodes_dict.keys(), key=lambda x: nodes_dict[x][0])
            }

        else:
            self.COMM.send(node_ID, dest=0)
            nodes_dict = None

        nodes_dict = self.COMM.bcast(nodes_dict)

        n_nodes = len(nodes_dict)
        n_variational_parameters = len(self.variational_parameters)

        # the number of variational parameters per node
        parameters_per_node = (n_variational_parameters + 1) // n_nodes

        if (parameters_per_node == 0) and (self.parallel == "jacobian"):
            # If the number of nodes is greater than n_variational_parameters + 1
            # and distribution of state evolution is not constrained to individual nodes
            # then create subcommunicators across multiple physical nodes.

            self.comm_opt_mapping = [[] for _ in range(n_variational_parameters + 1)]
            for i, key in enumerate(nodes_dict.keys()):
                for rank in nodes_dict[key]:
                    self.comm_opt_mapping[i % (n_variational_parameters + 1)].append(
                        rank
                    )

        elif (parameters_per_node == 0) and (self.parallel == "jacobian_local"):
            # If the number of nodes is greater than n_variational_parameters + 1
            # and state evolution is constrained to individual nodes, then create
            # a subcommunicator per variational parameter and ignore the remaining
            # nodes.

            self.comm_opt_mapping = []
            for key in nodes_dict.keys():
                self.comm_opt_mapping.append(nodes_dict[key])

        else:
            # If n_variational_parameters + 1 is greater than the number of nodes
            # then create a subcommunicator per node and distribute the parameters
            # evenly (as possible).

            self.comm_opt_mapping = []

            for key in nodes_dict.keys():

                if parameters_per_node == 1:
                    # If n_variational_parameters + 1 is equal to the number of MPI processes, then
                    # create a sub-communicator per node.
                    self.comm_opt_mapping.append(nodes_dict[key])
                    continue

                if parameters_per_node > len(nodes_dict[key]):
                    # If parameters_per_node is greater than the number of ranks at a node
                    # then create subcommnicators containing individual ranks.
                    for rank in nodes_dict[key]:
                        self.comm_opt_mapping.append([rank])
                    continue

                for part in np.array_split(nodes_dict[key], parameters_per_node):
                    # If parameters_per_node is less than the number of ranks at a node
                    # then divide those ranks into parameters_per_node subcommunicators.
                    self.comm_opt_mapping.append(part)

        # Colours specify membership to a particular self.COMM_OPT
        # comm_opt_roots is the self.COMM rank of the root process in each self.COMM_OPT
        self.colours = np.full(self.COMM.Get_size(), -1)
        self.comm_opt_roots = []
        for i, comm in enumerate(self.comm_opt_mapping):
            self.comm_opt_roots.append(min(comm))
            self.colours[comm] = i

        self.comm_opt_roots.sort()

        if self.colours[self.COMM.Get_rank()] != -1:

            self.COMM_OPT = MPI.Comm.Split(
                self.COMM,
                self.colours[self.COMM.Get_rank()],
                self.COMM.Get_rank(),
            )

        self.var_map = [[] for _ in range(len(self.comm_opt_mapping))]
        for var in range(n_variational_parameters):
            # The subcommunicator containing self.COMM rank 0 is used to compute the objective function
            # it is not assigned variables for gradient calculations.
            self.var_map[1:][var % (len(self.comm_opt_mapping) - 1)].append(var)

        # Create self.COMM_JAC, a communicator containing the self.COMM_OPT used to calculate the gradient
        # values and the root process of the sub communicator responsible for calls to the
        # objective function.
        self.jac_ranks = [
            rank for rank in range(self.COMM.Get_size()) if self.colours[rank] != 0
        ]
        self.jac_ranks.insert(0, 0)

        world_group = MPI.Comm.Get_group(self.COMM)
        jac_group = MPI.Group.Incl(world_group, self.jac_ranks)
        self.COMM_JAC = self.COMM.Create_group(jac_group)

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
                partials.append(self.jacobian.call())

        opt_root = self.comm_opt_roots[self.colours[self.COMM.Get_rank()]]

        if self.COMM.Get_rank() == 0:
            jacobian = np.zeros(len(self.variational_parameters), dtype=np.float64)
            reqs = []
            for root, mapping in zip(self.comm_opt_roots, self.var_map):
                if root > 0:
                    for var in mapping:
                        self.COMM.Recv(
                            [jacobian[var : var + 1], MPI.DOUBLE],
                            source=root,
                            tag=var,
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
