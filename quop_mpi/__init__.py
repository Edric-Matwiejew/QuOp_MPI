import sys
import os
import csv
from time import time
from importlib import import_module
import numpy as np
from mpi4py import MPI
from quop_mpi.__utils.__interface import interface
from quop_mpi.__utils.__mpi import shrink_communicator, gather_array


I = np.complex(0,1)

class ansatz(object):

    def __init__(self, system_size, MPI_communicator = MPI.COMM_WORLD, parallel = "global"):

        rank = MPI_communicator.Get_rank() # confirm argument is an MPI4Py communicator

        self.system_size = system_size
        self.COMM = MPI_communicator

        # initialisation inputs
        self.COMM = MPI_communicator # global MPI communicator, usually MPI.COMM_WORLD

        if parallel in ["jacobian", "jacobian_local", "global"]:
            self.parallel = parallel # type of MPI parallelisation: "global", "jacobian" or "jacobian_local"
        else:
            QUOP_ERR = "Rank {}: Parallel scheme '{}' not recognised. Options are 'jacobian', 'jacobian_local' and 'global'.".format(self.COMM.Get_rank(), parallel)
            raise ValueError(QUOP_ERR)

        # variables that must be set by the 'pre' method of the child class
        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None
        self.observables = None
        self.observables_func = None

        # can be set using methods in the system class
        # but default values are used if not set
        self.ansatz_depth = 1 # ansatz circuit depth
        self.initial_state_type = None
        self.optimiser = None # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize

        # parameters linked to optional methods in the 'system' class
        self.observable_map = None # scalar tranformation on the output of the objective function
        self.log = False # wether results will be recorded in a *.log file.

        # variables managed by the 'system' class
        self.stop = False # synchronise ranks durring optimisation
        self.COMM_OPT = None # communicator used for optimisation
        self.expectation = None # expectation value of the system
        self.initial_state_input = None
        self.initial_state = None # initial state before algorithm evolution
        self.final_state = None # quantum state durring and after simulation
        self.benchmarking = False # indicates wether the benchmark method is running
        self.pre_called = False
        self.post_called = False

        self.initial_state_parameters = [
                'system_size',
                'alloc_local',
                'local_i',
                'local_i_offset',
                'MPI_COMM'
                ]

        self.observable_map_parameters = [
                'observables',
                'system_size',
                'MPI_COMM',
                'partition_table'
                ]

    def __del__(self):
        self.post()

    def set_unitaries(self, unitaries, index):

        self.unitaries = unitaries

        self.param_map = np.zeros(len(self.unitaries) + 1, int)

        for i, unitary in enumerate(self.unitaries):
            self.param_map[i + 1] = unitary.n_params
            unitary.seed = i

        self.total_params = np.sum(self.param_map)
        self.param_map = np.cumsum(self.param_map)

        self.observable_index = index

        if self.unitaries[i].unitary_type != "diagonal":
            RuntimeError("Rank {}: Unitary containing observable values is not diagonal.".format(self.COMM.Get_rank()))

    def set_optimiser(self, optimiser, optimiser_args = {}, optimiser_log = None):
        """
        Defines the classical optimiser algorithm used, arguments passed to the optimiser and fields in the optimiser dictionary to write to the log file (when using :meth:`~system.log_results`). QuOp_MPI supports optimisers provided by SciPy through its minimize method `minimize <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ and optimisers provided by the `NLopt <http://nlopt.readthedocs.io/en/latest/>`_ package with respect to minmisation with scalar constraints through a SciPy-like interface.


        The default optimiser is the BFGS algorithm provided SciPy, which is set on instantiation of the :class:`~system` class as follows:

        .. code-block:: python

            self.set_optimiser( 'scipy',
                    {'method':'BFGS','tol':1e-3},
                                ['fun','nfev','success'])


        :param optimiser: 'scipy' to use the SciPy or 'nlopt' to use NLopt.
        :type optimiser: string

        :param optimiser_args: Arguments to pass to the optimiser.
        :type optimiser_args: dictionary

        :param optimiser_log: Results of the optimisation process are stored in a dictionary. These values may be logged by passing a list of the corresponding keys.
        :type optimiser_log: array, string
        """

        if optimiser == 'scipy':
            from scipy.optimize import minimize as sp_minimize
            self.optimiser = sp_minimize
        elif optimiser == 'nlopt':
            from quop_mpi.__util._nlopt_wrap import minimize as nlopt_minimize
            self.optimiser = nlopt_minimize

        self.optimiser_args = optimiser_args
        self.optimiser_log = optimiser_log

        if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
            if not "jac" in self.optimiser_args:
                self.optimiser_args["jac"] = self.__mpi_jacobian

    def set_depth(self, depth):
        self.ansatz_depth = depth

    def set_observable_map(self, **kwargs):
        self.observable_map_input = [func, kwargs]

    def __parse_observable_mapping(self):

        self.observable_map = interface(
                self,
                self.observable_map_input[0],
                self.observable_map_parameters,
                "observable mapping",
                self.COMM)

    def unset_observable_map(self):
        self.observable_map_input = None

    def set_initial_state(self, function, **kwargs):
        self.initial_state_input = [function, kwargs]

    def __parse_initial_state_function(self):

        self.initial_state_function = interface(
                self,
                self.initial_state_input[0],
                self.initial_state_parameters,
                "initial state",
                self.COMM)

    def set_log(self, filename, label, action = "a"):

        self.log = True
        self.filename = filename
        self.label = label
        self.log_action = "a"

        self.repeat = 1 # needed if logging results from the execute method

    def get_expectation_value(self):
        """
        :math:`\langle Q \\rangle =  \langle \\vec{\gamma}, \\vec{t}|Q|\\vec{\gamma}, \\vec{t} \\rangle`

        :return: The expectation value of the quality matrix operator, returned to all MPI nodes.
        :rtype: float
        """

        if self.colours[self.COMM.Get_rank()] == 0:
            return self.__get_expectation_value()

    def objective(self, variational_parameters):
        """
        :math:`f(\\vec{\gamma}, \\vec{t}) = \langle \\vec{\gamma}, \\vec{t} | Q |\\vec{\gamma}, \\vec{t} \\rangle` \
        - the function minimised by the calssical optimizer.


        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array

        """

        if self.colours[self.COMM.Get_rank()] == 0:
            return self.__objective(variational_parameters)

    def pre(self):

        busy_comm = False
        while not busy_comm:

            self.variational_parameters = np.empty(self.total_params * self.ansatz_depth, np.float64)

            # parallel jacobian not possible with one MPI process
            if self.COMM.Get_size() == 1:
                self.parallel = "global"

            # set up communication topology
            if self.COMM_OPT is None:

                if (self.parallel == "global"):
                    self.COMM_OPT = self.COMM
                    self.MPI_COMM = self.COMM
                    self.colours = [0]*self.COMM.Get_size()

                elif (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):

                    self.__parallel_jacobian_communication_topology()
                    self.n_jacobian_variables = len(self.variational_parameters)

            # if the number of MPI processes is greater than
            # system_size, resize the communicator

            if self.colours[self.COMM.Get_rank()] != -1:

                if self.system_size // self.COMM_OPT.Get_size() == 0:

                    newsize = self.system_size // 2

                    self.colours, self.COMM_OPT = shrink_communicator(
                            newsize,
                            self.colours,
                            self.COMM_OPT)

                    if self.colours[self.COMM.Get_rank()] == -1:
                        busy_comm = True

            if self.colours[self.COMM.Get_rank()] != -1:

                # find if any of the unitaries should be used for planning
                # TODO check that there are not planner of different type

                for unitary in self.unitaries:
                    if unitary.planner:
                        planner = unitary
                        planner.plan(self.system_size, self.COMM_OPT)
                        break
                else:
                    planner = self.unitaries[0]
                    planner.plan(self.system_size, self.COMM_OPT)

                self.alloc_local = planner.alloc_local
                self.local_i = planner.local_i
                self.local_i_offset = planner.local_i_offset
                self.final_state = planner.final_state

                if self.local_i == 0:
                    empty_rank = 1
                else:
                    empty_rank = 0

                empty_ranks = self.COMM_OPT.allreduce(
                        empty_rank,
                        op = MPI.SUM)

                if empty_ranks == 0:
                    busy_comm = True
                else:
                    planner.destroy()

                if not busy_comm:
                    newsize = self.COMM_OPT.Get_size() - empty_ranks
                    self.colours, self.COMM_OPT = shrink_communicator(
                            newsize,
                            self.colours,
                            self.COMM_OPT)

                if self.colours[self.COMM.Get_rank()] == -1:
                    busy_comm = True

        if self.colours[self.COMM.Get_rank()] != -1:

            for unitary in self.unitaries:
                if unitary is not planner:
                    unitary.copy_plan(planner)

            if self.observable_map is not None:
                self.__parse_observable_mapping()

            if self.initial_state_input is None:
                from quop_mpi.states import equal
                self.set_initial_state(equal)

            self.__parse_initial_state_function()

            self.initial_state = self.initial_state_function.call(
                    **self.initial_state_input[1]
                    )

            if self.observable_index is None:

                for i, unitary in enumerate(self.unitaries):
                    if unitary.unitary_type == "diagonal":
                        self.observable_index = i
                        break
                else:
                    RuntimeError("Rank {}: Cannot identify observables, no diagonal unitary defined".format(self.COMM.Get_rank()))


            for i, unitary in enumerate(self.unitaries):

                if unitary.operator_n_params == 0:
                    unitary.gen_operator()
                    if self.observable_index == i:
                        self.observables = np.real(unitary.operator)

            if self.optimiser is None:
                self.set_optimiser( 'scipy',
                    {'method':'BFGS','tol':1e-5},
                                ['fun','nfev','success'])

            if self.log:
                self.__gen_log()

        self.pre_called = True

    def post(self):

        for unitary in self.unitaries:
            unitary.destroy()

        self.pre_called = False

        if self.colours[self.COMM.Get_rank()] != -1:
            if self.COMM_OPT.Get_size() < self.COMM.Get_size():
                MPI.Comm.Free(self.COMM_OPT)

        self.COMM_OPT = None

    def evolve_state(self, x):

        if not self.pre_called:
            self.pre()

        cnt = 0
        if self.colours[self.COMM.Get_rank()] != -1:

            self.final_state[:self.local_i] = self.initial_state

            params_split = np.split(x, self.ansatz_depth)

            for depth, params in enumerate(params_split):

                for i, (param_group, unitary) in enumerate(zip(params, self.unitaries)):

                    if unitary.operator_n_params > 0:

                        operator_parameters = param_group[:-1]
                        evolution_parameter = param_group[0]

                        unitary.gen_operator(operator_parameters)

                    else:
                        evolution_parameter = param_group

                    if not self.__is_zero(evolution_parameter):
                        cnt += 1
                        unitary.initial_state[:self.local_i] = self.final_state[:self.local_i]

                        unitary.propagate(evolution_parameter)

                        self.final_state[:self.local_i] = unitary.final_state[:self.local_i]

    def execute(self):
        """
        Execute the QAOA-like algorithm.

        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array
        """
        if self.pre_called:
            if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
                if not (self.n_jacobian_variables == len(self.variational_parameters)):
                    self.post()
                    self.pre()
        else:
            self.pre()

        if not self.benchmarking:
            self.variational_parameters = self.get_initial_params(self.ansatz_depth)

        if self.colours[self.COMM.Get_rank()] != -1:

            self.stop = False

            if self.colours[self.COMM.Get_rank()] == 0:

                if self.COMM_OPT.Get_rank() == 0:

                    self.time = time()

                    self.result = self.optimiser(
                            self.__objective,
                            self.variational_parameters,
                            **self.optimiser_args)

                    self.stop = True

                    self.objective(None)

                    if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
                        self.__mpi_jacobian(None)

                    self.time = time() - self.time

                else:

                    while not self.stop:
                        self.__objective(self.variational_parameters)

            else:

                while not self.stop:
                    self.__mpi_jacobian(None)


            if self.log:
                self.__log_update()

    def print_optimiser_result(self):
        """
        Print the optimization result.
        """
        if self.COMM.Get_rank() == 0:
            print(self.result)

    def benchmark(
            self,
            ansatz_depths,
            repeats,
            param_persist = False,
            verbose = True,
            filename = None,
            label = 'test',
            save_action = "a"):

        """
        This provides an easy method by which to study how a QAOA algorithm
        performs with increases to the circuit depth, :math:`p`.

        :param ps: List of :math:`p` values.
        :type ps: integer, array

        :param repeats: The number of repeats at each value of :math:`p`.
        :type repeats: integer

        The following four parameters specify the starting conditions for each repeat and :math:`p`.
        Of these `param_func` is required as :math:`(\\vec{\gamma},\\vec{t})` grows with :math:`p`.

        :param param_func: Method returning starting :math:`(\\vec{\gamma},\\vec{t})`.
        :type param_func: callable

        Example:

        .. code-block:: python

            def x0(p, seed):
                return np.random.uniform(low = 0, high = 1, size = 2*p)

        :param qual_func: Method compatible with :meth:`~system.set_qualities`.
        :type qual_func: callable, optional, default = None

        :param state_func: Method to generate a distributed inital state, compatible with :meth:`~system.set_initial_state`.
        :type state_func: callable, optional, default = None

        :param param_persist: If `True` the optimized :math:`(\\vec{\gamma},\\vec{t})` values which achieved the lowest objective function value  for all repeats at :math:`p` will be used as starting parameters for :math:`p + 1`.
        :type param_persist: boolean, optional

        The following parameters specify the output behaviours.

        :param verbose: If True, print current :math:`p`, repitition number and optimization results.
        :type verbose: boolean, optional, default = True

        :param filename: Name of .h5 file in which to :meth:`~qwoa.system.save` the evolved system.
        :type filename: string, optional, default = None

        :param label: If filename is specified, evolved systems will be saved as 'filename/label_p_repetition'.
        :type label: string, optional, default = 'test'

        :param save_action: Action taken durring first .5 file write. "a", append. "w", over-write.
        :type save_action: string, optional, default = "a"
        :param kwargs: Keyword arguments to pass to the :meth:`~system.set_qualities` method.
        :type kwargs: dictionary, optional

        .. note::
            The `param_func`, `qual_func` and `state_func` must have the keyword argument 'seed'. This allows for a repeatable variation if :math:`(\\vec{\gamma}, \\vec{t}), q_i` and :math:`| s \\rangle` with each repetition at the same :math:`p`.
        """

        if not self.pre_called:
            self.pre()

        if self.colours[self.COMM.Get_rank()] != -1:

            self.benchmarking = True

            self.rank = self.COMM_OPT.Get_rank()

            first = True

            itter = 0

            previous_params = None

            ansatz_depth_temp = self.ansatz_depth # return to this value after benchmarking
            for depth in ansatz_depths:
                self.ansatz_depth = depth
                if param_persist:
                    best_p_result = np.finfo(dtype=np.float64).max
                    result = None

                if verbose:
                    if self.COMM_OPT.Get_rank() == 0:
                        print('Starting depth = ' + str(depth) + ':')

                for i  in range(1, repeats + 1):

                    self.repeat  = i

                    if (not param_persist) or first:

                        self.variational_parameters = self.get_initial_params(depth)

                    else:

                        new_parameters = self.get_initial_params(1)

                        self.variational_parameters = np.append(previous_params, new_parameters)

                    if verbose:
                        if self.COMM_OPT.Get_rank() == 0:
                            print(str(i) + ' of ' + str(repeats) + '...')

                    self.execute()

                    if param_persist:

                        if self.rank == 0:
                            result = self.result['fun']
                            x = self.result['x']
                        else:
                            result = None
                            x = None

                        result = self.COMM_OPT.bcast(result, root = 0)
                        x = self.COMM_OPT.bcast(x, root = 0)

                        if result < best_p_result:
                            best_p_result = result
                            best_p_params = x

                    if self.COMM_OPT.Get_rank() == 0:
                        if verbose:
                            print(self.result,flush=True)

                    if filename is not None:
                        if first:
                            self.save(filename, label + '_' + str(depth) + '_' + str(i), action = save_action)
                        else:
                            self.save(filename, label + '_' + str(depth) + '_' + str(i), action = "a")

                if param_persist:
                    previous_params = best_p_params
                    first = False

            self.post()
            self.benchmarking = False
            self.ansatz_depth = ansatz_depth_temp

    def get_final_state(self):
        if self.colours[self.COMM.Get_rank()] != -1:
            if self.colours[self.COMM.Get_rank()] == 0:
                return gather_array(
                        self.final_state,
                        self.unitaries[0].partition_table,
                        self.COMM_OPT)

    def get_probabilities(self):
        if self.colours[self.COMM.Get_rank()] != -1:
            if self.colours[self.COMM.Get_rank()] == 0:
                prob = np.abs(self.final_state)**2
                return gather_array(
                        prob,
                        self.unitaries[0].partition_table,
                        self.COMM_OPT)

    def save(self, file_name, config_name, action = "a", verbose = False):
        """
        Write the final state, qualities and execution results
        of the current configuration to a HDf5 file.

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
            qualities = np.array(f[config_name]['qualities']).view(np.float64)

            print(f["my_simulation"].attrs["minimize_result"])

        .. warning::
            The final_state and qualities datasets are saved using Fortran
            subroutines which make use of parallel HDF5.

            The complex values of the final_state array are saved as a
            compound datatype consisting of contiguous double precision reals. This is equivalent to the np.complex128 NumPy datatype. To access this data without a
            loss of precision in python, the user must set the **view** of the NumPy array to np.complex128, rather than casting it to np.complex128 using the dtype keyword.

            Similarly, the qualities array, which is saved as an array of double-precision
            reals, should have its view set to np.float64.
        """

        if self.colours[self.COMM.Get_rank()] == 0:

            import h5py
            from quop_mpi.__lib import fqwoa_mpi

            if self.COMM_OPT.Get_rank() == 0:

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

                File.create_dataset(self.config_name + "/initial_phases", data = self.variational_parameters, dtype = np.float64)
                File.close()

            self.config_name = self.COMM_OPT.bcast(config_name, root = 0)

            fqwoa_mpi.save_dist_complex(
                    file_name,
                    self.config_name + str("/"),
                    "final_state",
                    "a",
                    self.system_size,
                    self.local_i_offset,
                    self.final_state[:self.local_i],
                    self.COMM_OPT.py2f())

            fqwoa_mpi.save_dist_real(
                    file_name,
                    self.config_name + str("/"),
                    "observables",
                    "a",
                    self.system_size,
                    self.local_i_offset,
                    self.observables[:self.local_i],
                    self.COMM_OPT.py2f())

            self.COMM_OPT.Barrier()

    def get_initial_params(self, ansatz_depth = None):

        if not self.pre_called:
            self.pre()

        if self.colours[self.COMM.Get_rank()] != -1:

            if ansatz_depth is None:
                ansatz_depth = self.ansatz_depth

            params = np.zeros(ansatz_depth*self.total_params)

            if self.COMM_OPT.Get_rank() == 0:

                param_iterations = np.split(params, ansatz_depth)

                for param_iters in param_iterations:
                    for i, unitary in enumerate(self.unitaries):
                        unitary.seed += i + 1
                        param_iters[self.param_map[i]:self.param_map[i+1]] = unitary.get_initial_params()

            self.COMM_OPT.Bcast([params, MPI.DOUBLE], 0)

            return params

    def __get_local_probabilities(self):
        """
        :math:`\\vec{p} = ( \langle s_i|\\vec{\gamma}, \\vec{t} \\rangle` ), i=0,N-1

        :return: Probability vector corresponding the the local `self.final_state` partition.
        :rtype: array, float
        """
        self.local_probabilities = np.abs(self.final_state[:self.local_i], dtype = np.float64)**2
        return self.local_probabilities

    def __get_state_norm(self):
        """
        Check that :math:`\langle \\vec{\gamma}, \\vec{t}|\\vec{\gamma}, \\vec{t} \\rangle = 1`.
        The result is returned to each MPI rank and should be equal to 1 within the limits of double machine precision. This is used to check for state validity.

        :return: Norm of the current `self.final_state`.
        :rtype: float
        """
        if self.colours[self.COMM.Get_rank()] == 0:
            self.state_norm = self.COMM_OPT.allreduce(np.sum(self.local_probabilities), op = MPI.SUM)
            return self.state_norm

    def __get_expectation_value(self):

        self.__get_local_probabilities()

        if self.observable_map is not None:

            local_expectation = np.dot(
                    self.local_probabilities,
                    self.observable_map(
                        self.observables,
                        *self.observable_map_args,
                        **self.observable_map_kwargs
                    ))
        else:

            local_expectation = np.dot(self.local_probabilities, self.observables)

        return self.COMM_OPT.allreduce(local_expectation, op = MPI.SUM)


    def __objective(self, variational_parameters):


        self.stop = self.COMM_OPT.bcast(self.stop, root = 0)

        if not self.stop:

            self.variational_parameters = self.COMM_OPT.bcast(variational_parameters, root = 0)

            self.evolve_state(self.variational_parameters)

            self.expectation = self.get_expectation_value()
            return self.expectation

    def __gen_log(self):
        """
        Creates a .csv in which to save key QAOA results after a call to :meth:`~system.execute`.

        :param filename: Name of the .csv file.
        :type filename: string

        :param label: User-set identifier of the currently defined system.
        :type label: string

        :param action: "a", append. "w", over-write.
        :type action: string, optional, default = "a"

        Once called, addtional calls to :meth:`~system.log_update` will save the following information:

        * label: User-defined system label.
        * p: :math:`p`.
        * objective_function: Final result of objective function minimization.
        * objective_evaluations: Number of objective function evalutions needed durring optimisation.
        * optimization_success: If the minimizer converged to its target tolerances.
        * state_norm: Norm of the final state. This should always equal 1 (within the limits of double precision accuracy).
        * simulation_time: In-program simultion time.
        * MPI_nodes: Number of mpi processes.
        """

        self.n_log_fields = 6

        if self.COMM.Get_rank() == 0:
            if (os.path.exists(self.filename + ".csv") and self.log_action == "a"):
                self.logfile = open(self.filename + ".csv", "a", newline='')
                self.logfile_csv = csv.writer(self.logfile)
            else:

                headings = ['label','system_size','ansatz_depth','repeat','state_norm','simulation_time','MPI_nodes']

                if self.optimiser_log is not None:
                    for optimiser_log in self.optimiser_log:
                        headings.append(optimiser_log)

                self.logfile = open(self.filename + ".csv", "w")
                self.logfile_csv = csv.writer(self.logfile)
                self.logfile_csv.writerow(headings)

    def __log_update(self):
        """
        Update a .csv log of QAOA algorithm performance, instantiated by :meth:`~system.log_results`.
        """
        self.state_norm = self.__get_state_norm()

        if self.COMM.Get_rank() == 0:

            log_output = [
                    self.label,
                    self.system_size,
                    self.ansatz_depth,
                    self.repeat,
                    self.state_norm,
                    self.time,
                    self.COMM.size
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
                node_IDs.append(self.COMM.recv(source = i))

            unique_IDs = list(set(node_IDs))
            rank_lists = [[] for _ in range(len(unique_IDs))]
            nodes_dict = dict(zip(unique_IDs, rank_lists))

            for i, ID in enumerate(node_IDs):
                nodes_dict[ID].append(i)

            for key in nodes_dict:
                nodes_dict[key] = np.array(nodes_dict[key])

        else:
                self.COMM.send(node_ID, dest = 0)
                nodes_dict = None

        nodes_dict = self.COMM.bcast(nodes_dict)

        n_nodes = len(nodes_dict)
        n_variational_parameters = len(self.variational_parameters)

        # the number of variational parameters per node
        parameters_per_node = (n_variational_parameters + 1)//n_nodes

        if (parameters_per_node == 0) and (self.parallel == "jacobian"):
        # If the number of nodes is greater than n_variational_parameters + 1
        # and distribution of state evolution is not constrained to individual nodes
        # then create subcommunicators across multiple physical nodes.

            self.comm_opt_mapping = [[] for _ in range(variables + 1)]
            for i, key in enumerate(nodes_dict.keys()):
                for rank in nodes_dict[key]:
                    self.comm_opt_mapping[i % variables].append(rank)

        elif (parameters_per_node == 0) and (self.parallel == "jacobian_local"):
        # If the number of nodes is greater than n_variational_parameters + 1
        # and state evolution is constrained to individual nodes, then create
        # a subcommunicator per variational parameter and ignore the remianing
        # nodes.

            self.comm_opt_mapping = []
            for node, parameter in zip(nodes_dict.keys(), list(range(n_variational_parameters))):
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
                    comm_opt_mapping.append(part)

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
                    self.COMM.Get_rank())

        self.var_map = [[] for _ in range(len(self.comm_opt_mapping))]
        for var in range(n_variational_parameters):
        # The subcommunicator containing self.COMM rank 0 is used to compute the objective function
        # it is not assigned variables for gradient calculations.
            self.var_map[1:][var % len(self.comm_opt_mapping) - 1].append(var)

        # Create self.COMM_JAC, a communicator containing the self.COMM_OPT used to calculate the gradient
        # values and the root process of the sub communicator responsible for calls to the
        # objective function.
        jac_ranks = [rank for rank in range(self.COMM.Get_size()) if self.colours[rank] != 0]
        jac_ranks.insert(0,0)

        world_group = MPI.Comm.Get_group(self.COMM)
        jac_group = MPI.Group.Incl(world_group, jac_ranks)
        self.COMM_JAC = self.COMM.Create_group(jac_group)

    def __mpi_jacobian(self, x, tol = 1e-13):

        self.COMM_JAC.barrier()

        self.stop = self.COMM_JAC.bcast(self.stop, 0)

        if self.stop:
            self.COMM_JAC.barrier()
            return

        x = self.COMM_JAC.bcast(x, 0)

        # if the jacobian is called before evaluation of the objective function
        if self.colours[self.COMM.Get_rank()] == 0:

            if self.expectation is None:

                self.evolve_state(x)
                self.expectation = self.__get_expectation_value()

        self.expectation = self.COMM_JAC.bcast(self.expectation, 0)

        x_jac_temp = np.empty(len(x))
        partials = []

        if  self.COMM.Get_rank() != 0:

            h = np.abs(np.min(x)*np.sqrt(tol))

            if self.__is_zero(h):
                h = 1.4901161193847656e-08

            for var in self.var_map[self.colours[self.COMM.Get_rank()]]:
                x_jac_temp[:] = x
                x_jac_temp[var] += h
                self.evolve_state(x_jac_temp)
                partials.append((self.__get_expectation_value() - self.expectation)/h)

        opt_root = self.comm_opt_roots[self.colours[self.COMM.Get_rank()]]

        if self.COMM.Get_rank() == 0:
            jacobian = np.zeros(len(self.variational_parameters), dtype = np.float64)
            reqs = []
            for root, mapping in zip(self.comm_opt_roots, self.var_map):
                if root > 0:
                    for var in mapping:
                        self.COMM.Recv([jacobian[var:var+1], MPI.DOUBLE],  source = root, tag = var)

        elif self.COMM_OPT.Get_rank() == 0:
            reqs = []
            jacobian = None
            for part, mapping in zip(partials, self.var_map[self.colours[self.COMM.Get_rank()]]):
                self.COMM.Send([np.array([part]), MPI.DOUBLE], dest = 0, tag = mapping)
        else:
            jacobian = None

        self.COMM_JAC.barrier()

        if self.COMM.Get_rank() == 0:
            return jacobian
        else:
            return None

    def __is_zero(self, x):
        return (x >= -np.finfo(np.float64).eps) and  (x <= np.finfo(np.float64).eps)

class phase_and_mixer(ansatz):

    def __init__(self, system_size, MPI_communicator = MPI.COMM_WORLD, parallel = "global"):

        super().__init__(system_size, MPI_communicator, parallel = "global")

        self.operator_function = None
        self.param_function = None

    def set_qualities(self, operator_function, **kwargs):
        self.operator_function = operator_function
        self.operator_kwargs = kwargs

    def set_params(self, param_function, **kwargs):
        self.param_function
        self.param_kwargs = kwargs

    def _pre(self):

        if self.operator_function is None:
            raise RuntimeError("Rank {}: Solution qualities not defined.".format(self.rank))

        if self.param_function is None:
            from quop_mpi.params import uniform
            self.set_params(uniform)


