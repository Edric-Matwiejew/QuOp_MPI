from mpi4py import MPI
import h5py
import numpy as np
from scipy.optimize import minimize as sp_minimize
from scipy import sparse
import sys
import os
import quop_mpi.fqwoa_mpi as fqwoa_mpi
import quop_mpi.fMPI as fMPI
import quop_mpi.mixers_mpi as mixers_mpi
from quop_mpi.nlopt_wrap import minimize as nlopt_minimize
from time import time
import csv

I = np.complex(0,1)

class system(object):
    """
    Provides a framework for the simulation of a QAOA-like algorithm
    in parallel using MPI. To do so, this class must be used to form a
    subclass containing an 'evolve_state' method:

    .. code-block:: python

        class(system):

            def __init__(self, system_size, MPI_communicator):

                ...

            def evolve_state(self, gammas, ts):

                self.initial_state = self.final_state

                self.final_state = ...

    The 'evolve_state' method must modify self.final_state in place.
    self.final_state and self.initial_state must be distributed over an MPI
    communicator and described by the following variables:

    * local_i: The number of array indices stored at a given rank.
    * local_i_offset: the offset of those local indices in the globally distributed array
    * alloc_local: The size of the local array partition. This may be greater than local_i, but the local array partition must be fully defined by the first local_i values.

    self.qualities must be defined by local_i, local_i_offset and be of size local_i.

    QWAO_MPI contains two :class:`system` subclasses: :class:`qaoa` and :class:`qwoa`.
    """
    def __init__(self, MPI_communicator, parallel = "global"):

        # initialisation inputs
        rank = MPI_communicator.Get_rank() # confirm argument is an MPI4Py communicator
        self.COMM = MPI_communicator # global MPI communicator, usually MPI.COMM_WORLD

        if parallel in ["jacobian", "jacobian_local", "global"]:
            self.parallel = parallel # type of MPI parallelisation: "global", "jacobian" or "jacobian_local"
        else:
            QUOP_ERR = "Parallel scheme '{}' not recognised. Options are 'jacobian', 'jacobian_local' and 'global'.".format(parallel)
            raise self._quop_raise_error(ValueError, QUOP_ERR)

        # must be set at using methods in the system class
        self.variational_parameters = None # parameters tuned via classical optimisation

        # variables that must be set by the 'pre' method of the child class
        self.system_size = None
        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None
        self.observables = None

        # can be set using methods in the system class
        # but default values are used if not set
        self.ansatz_depth = None # ansatz circuit depth
        self.initial_state_type = None
        self.optimiser = None # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize

        # parameters linked to optional methods in the 'system' class
        self.observable_map = None # scalar tranformation on the output of the objective function
        self.log = False # wether results will be recorded in a *.log file.

        # variables managed by the 'system' class
        self.stop = False # synchronise ranks durring optimisation
        self.COMM_OPT = None # communicator used for optimisation
        self.expectation = None # expectation value of the system
        self.initial_state = None # initial state before algorithm evolution
        self.final_state = None # quantum state durring and after simulation
        self.benchmarking = False # indicates wether the benchmark method is running
        self.pre_called = False
        self.post_called = False

    def set_observables(self, func, *args, **kwargs):
        """
        Sets the local, :math:`q_i`. Ideally, each local partition of :math:`\\vec{q}` should be generated in parallel. As such :meth:`~qwoa.qualities` accepts a function whose first three arguments are the size of the distributed qwoa state,
        the number of locally stored input elements and the offset of these elements relative to the 0-index of the distributed array. Example quality functions are included in :mod:`~qwoa_mpi.qualities`.

        :param func: Function with which to generate the local :math:`q_i`.
        :type func: callable

        :param args: Extra arguments to pass to the quality function.
        :type args: array, optional
        """

        self.observables_func = func
        self.observable_func_args = args
        self.observable_func_kwargs = kwargs

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
            self.optimiser = sp_minimize
        elif optimiser == 'nlopt':
            self.optimiser = nlopt_minimize

        self.optimiser_args = optimiser_args
        self.optimiser_log = optimiser_log

        if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
            if not "jac" in self.optimiser_args:
                self.optimiser_args["jac"] = self._mpi_jacobian

    def set_observable_mapping(self, func, *args, **kwargs):
        self.objective_map = func
        self.objective_map_args = args
        self.objective_map_kwargs = kwargs

    def unset_observable_mapping(self):
        self.objective_map = None
        self.objective_map_args = []
        self.objective_map_kwargs = {}

    def set_initial_state(self, name = None, basis_states = None, state = None, normalized = True):
        """
        Set :math:`| s \\rangle`. This can be done in several mutually exclusive ways:

        * `name` keyword.
            * 'equal' - an equal superposition accross all :math:`|s_i\\rangle`.
            * 'localized' - :math:`|s_0\\rangle = 1`.
            * 'split' - :math:`| s_0 \\rangle, | s_2 \\rangle = \\frac{1}{2}`.
        * `vertices` keyword.
            * Pass an MPI rank specific array of unique :math:`i` such that :math:`\\text{local_i} \leq i < \\text{local_i + local_i_offset}`. :math:`| s \\rangle` will be initialized as an equal superposition across the specified :math:`| s_i \\rangle`.
        * `state` keyword.
            * Fully specify :math:`|s\\rangle`. Pass an MPI rank specific array of :math:`x_i \in \mathbb{C}` such that :math:`x_i    \geq 0' and  :math:`\\text{local_i} \leq i < \\text{local_i + local_i_offset}`. If keyword `normalized` is True, this state will be normalized.


        :param name: Name of a pre-defined initial state.
        :type name: str, optional

        :param vertices: Specify an equal superposition over a set of :math:`|s_i\\rangle`.
        :type verticies: array, integer, optional


        :state: Initialize :math:`|s\\rangle` in a user-defined generalized state.
        :param: array, float, optional


        :state normalized: If normalized is `True` and an argument to `state` is specified, normalize the input state.
        :param normalized: boolean, optional

        """

        def state_check(state_type):
            if self.initial_state_type is not None:
                QUOP_WRN = "Initial state redefined, switching to '{}'.".format(state_type)
                raise Warning(QUOP_WRN)
            self.initial_state_type = state_type


        if name is not None:
            self.name = name
            state_check("name")
        elif basis_states is not None:
            self.basis_states = basis_states
            state_check("basis_states")
        elif state is not None:
            self.state = state
            self.normalized = normalized
            state_check("state")

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
            return self._get_expectation_value()

    def objective(self, variational_parameters):
        """
        :math:`f(\\vec{\gamma}, \\vec{t}) = \langle \\vec{\gamma}, \\vec{t} | Q |\\vec{\gamma}, \\vec{t} \\rangle` \
        - the function minimised by the calssical optimizer.


        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array

        """

        if self.colours[self.COMM.Get_rank()] == 0:
            return self._objective(variational_parameters)

    def pre(self):
        raise NotImplementedError("Simulation setup method 'pre' not defined by system subclass.")

    def post(self):
        pass

    def execute(self, x, post = True):
        """
        Execute the QAOA-like algorithm.

        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array
        """

        self._set_parameters(x)

        if self.pre_called:
            if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
                if not (self.n_jacobian_variables == len(self.variational_parameters)):
                    self._post()
                    self._pre()
        else:
            self._pre()

        if self.colours[self.COMM.Get_rank()] != -1:

            self.stop = False

            if self.colours[self.COMM.Get_rank()] == 0:

                if self.COMM_OPT.Get_rank() == 0:

                    self.time = time()

                    self.result = self.optimiser(
                            self.objective,
                            self.variational_parameters,
                            **self.optimiser_args)

                    self.stop = True
                    self.objective(None)

                    if (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):
                        self._mpi_jacobian(None)

                    self.time = time() - self.time

                else:

                    while not self.stop:
                        self.objective(self.variational_parameters)

            else:

                while not self.stop:
                    self._mpi_jacobian(None)


            if self.log:
                self._log_update()

            if post:
                self._post()

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
            param_func,
            obs_func = None,
            param_persist = False,
            verbose = True,
            filename = None,
            label = 'test',
            save_action = "a",
            *args,
            **kwargs):

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

        if self.colours[self.COMM.Get_rank()] != -1:

            first = True

            itter = 0

            previous_params = None

            for depth in ansatz_depths:

                if param_persist:
                    best_p_result = np.finfo(dtype=np.float64).max
                    result = None

                if verbose:
                    if self.COMM_OPT.Get_rank() == 0:
                        print('Starting depth = ' + str(depth) + ':')

                for i  in range(1, repeats + 1):

                    self.repeat  = i

                    np.random.seed(i + itter)

                    if (not param_persist) or first:

                        if self.rank == 0:
                            self._set_parameters(param_func(depth, seed = i + itter))
                            self.parameter_groups = int(len(self.variational_parameters) / depth)
                        else:
                            self.variational_parameters = None

                        self.variational_parameters = self.comm.bcast(self.variational_parameters, root = 0)

                    else:

                        if self.COMM_OPT.Get_rank() == 0:
                            new_parameters = param_func(1,seed = i + itter)
                        else:
                            new_parameters = None

                        new_parameters =  self.COMM_OPT.bcast(new_parameters, root = 0)
                        self.variational_parameters = np.append(self.variational_parameters, new_parameters)

                    if obs_func is not None:
                        self._gen_observables(obs_func, seed = i + itter, *args, **kwargs)
                    #if state_func is not None:
                    #    self.initial_state = state_func(p, seed = i + itter)

                    if verbose:
                        if self.comm.Get_rank() == 0:
                            print(str(i) + ' of ' + str(repeats) + '...')

                    self.execute(self.variational_parameters, post = False)

                    itter += 1

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
                            self.save(filename, label + '_' + str(p) + '_' + str(i), action = save_action)
                        else:
                            self.save(filename, label + '_' + str(p) + '_' + str(i), action = "a")

                if param_persist:
                    previous_params = best_p_params
                    first = False

            self._post()

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

    def _set_parameters(self, x):

        self.variational_parameters = np.array(x, dtype = np.float64)

        if self.variational_parameters.ndim != 1:
            QUOP_ERR = "Provided parameter list not 1-dimensional."
            raise self._quop_raise_error(TypeError, QUOP_ERR)

    def _gen_initial_state(self):

        # In some cases rank 0 might have no local indicies.
        if self.local_i > 0:
            rank = self.COMM_OPT.Get_rank()
        else:
            rank = self.COMM_OPT.Get_size()

        self.lowest_rank = self.COMM_OPT.allreduce(rank, op = MPI.MIN)

        if self.initial_state_type == "name":

            if self.name == "equal":
                self.initial_state = np.ones(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
            elif self.name == "localized":
                self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
                if self.COMM_OPT.Get_rank() == self.lowest_rank:
                    self.initial_state[0] = 1.0
            elif self.name == "split":
                self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
                if self.COMM_OPT.Get_rank() == self.lowest_rank:
                    self.initial_state[0:2] = 1.0/np.sqrt(2.0, dtype = np.float64)
            else:
                QUOP_ERR = "Initial state name = '{}' not recognised. Options are 'equal', 'localized' and 'split'".format(self.name)
                raise self._quop_raise_error(ValueError, QUOP_ERR)

        elif self.initial_state_type == "basis_states":

            self.initial_state = np.zeros(self.alloc_local, np.complex128)

            n_basis_states = len(self.basis_states) #self.COMM_OPT.allreduce(np.float64(len(vertices)), op = MPI.SUM)
            for state in self.basis_states:
                if (state > self.local_i_offset) and (state <= self.local_i_offset + self.local_i):
                    self.initial_state[vertex] = 1.0/np.sqrt(total_verticies, dtype = np.float64)

        elif self.initial_state_type == "state":

            self.initial_state = np.zeros(self.alloc_local, dtype = np.complex128)

            self.initial_state[0:self.local_i] = np.array(self.state[self.local_i_offset:self.local_i_offset + self.local_i], np.complex128)

            if not normalized:
                normalization = self.COMM_OPT.allreduce(np.dot(np.conjugate(state), state), op = MPI.SUM)
                self.initial_state = self.initial_state/np.sqrt(normalization)

        else:
            QUOP_ERR = "Initial state type '{}' not recognised. Possible types are 'name', 'basis_states' or 'state'.".format(self.initial_state_type)
            raise self._quop_raise_error(ValueError, QUOP_ERR)

    def _gen_observables(self, obs_func, *args, **kwargs):

        self.observables = obs_func(
                self.system_size,
                self.local_i,
                self.local_i_offset)

        if (self.observables.ndim != 1) and (len(self.observables) != self.local_i):
            QUOP_ERR = "Output of obs_func is not a numpy array of length {}.".format(self.local_i)
            raise self._quop_raise_error(TypeError, QUOP_ERR)

    def _get_local_probabilities(self):
        """
        :math:`\\vec{p} = ( \langle s_i|\\vec{\gamma}, \\vec{t} \\rangle` ), i=0,N-1

        :return: Probability vector corresponding the the local `self.final_state` partition.
        :rtype: array, float
        """
        self.local_probabilities = np.abs(self.final_state[:self.local_i], dtype = np.float64)**2
        return self.local_probabilities

    def _get_state_norm(self):
        """
        Check that :math:`\langle \\vec{\gamma}, \\vec{t}|\\vec{\gamma}, \\vec{t} \\rangle = 1`.
        The result is returned to each MPI rank and should be equal to 1 within the limits of double machine precision. This is used to check for state validity.

        :return: Norm of the current `self.final_state`.
        :rtype: float
        """
        if self.colours[self.COMM.Get_rank()] == 0:
            self.state_norm = self.COMM_OPT.allreduce(np.sum(self.local_probabilities), op = MPI.SUM)
            return self.state_norm

    def _get_expectation_value(self):

        self._get_local_probabilities()

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


    def _objective(self, variational_parameters):


        self.stop = self.COMM_OPT.bcast(self.stop, root = 0)

        if not self.stop:

            self.variational_parameters = self.COMM_OPT.bcast(variational_parameters, root = 0)

            self.evolve_state(self.variational_parameters)

            self.expectation = self.get_expectation_value()

            return self.expectation

    def _gen_log(self):
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

                self.logfile = open(filename + ".csv", "w")
                self.logfile_csv = csv.writer(self.logfile)
                self.logfile_csv.writerow(headings)

    def _log_update(self):
        """
        Update a .csv log of QAOA algorithm performance, instantiated by :meth:`~system.log_results`.
        """
        self.state_norm = self._get_state_norm()

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

    def _parallel_jacobian_communication_topology(self):

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

    def _pre(self):

        if self.variational_parameters is None:
            QUOP_ERR = "Variational parameters not defined."
            raise self._quop_raise_error(RuntimeError, QUOP_ERR)

        if self.observables_func is None:
            QUOP_ERR = "Observables function not defined."
            raise self._quop_raise_error(RuntimeError, QUOP_ERR)

        # set up communication topology
        if self.parallel == "global":

            self.COMM_OPT = self.COMM

            self.colours = [0]*self.COMM.Get_size()

        elif (self.parallel == "jacobian") or (self.parallel == "jacobian_local"):

            self._parallel_jacobian_communication_topology()
            self.n_jacobian_variables = len(self.variational_parameters)

        if self.colours[self.COMM.Get_rank()] != -1:

            self.pre()

            partition_parameters = [self.system_size, self.alloc_local, self.local_i, self.local_i_offset]
            parameter_names = ["system_size", "alloc_local", "local_i", "local_i_offset"]

            for parameter, name in zip(partition_parameters, parameter_names):
                if parameter is None:
                    QUOP_ERR = "MPI partition parameter '{}' not defined by 'pre' method of system subclass.".format(name)
                    raise self._quop_raise_error(RuntimeError, QUOP_ERR)

            if self.observables is None:

                self._gen_observables(
                        self.observables_func,
                        self.observable_func_args,
                        self.observable_func_kwargs)

            if self.initial_state_type is None:
                self.set_initial_state(name = "equal")

            self._gen_initial_state()

            if self.optimiser is None:
                self.set_optimiser( 'scipy',
                    {'method':'BFGS','tol':1e-5},
                                ['fun','nfev','success'])

            if self.ansatz_depth is None:
                self.ansatz_depth = 1

            if self.log:
                self._gen_log()

            self.pre_called= True

    def _post(self):
        self.post()

    def _mpi_jacobian(self, x, tol = 1e-13):

        self.COMM_JAC.barrier()

        self.stop = self.COMM_JAC.bcast(self.stop, 0)

        if self.stop:
            self.COMM_JAC.barrier()
            return

        x = self.COMM_JAC.bcast(x, 0)
        self.expectation = self.COMM_JAC.bcast(self.expectation, 0)

        x_jac_temp = np.empty(len(x))
        partials = []

        if  self.COMM.Get_rank() != 0:
            #xs, ts = np.split(x,2)
            #self.evolve_state(xs, ts)
            #expectation = self.expectation()
            h = np.abs(np.min(x)*np.sqrt(tol))
            if (h >= -np.finfo(np.float64).eps) and  (h <= np.finfo(np.float64).eps):
                h = 1.4901161193847656e-08
            for var in self.var_map[self.colours[self.COMM.Get_rank()]]:
                x_jac_temp[:] = x
                x_jac_temp[var] += h
                self.evolve_state(x_jac_temp)
                partials.append((self._get_expectation_value() - self.expectation)/h)

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

    def _quop_raise_error(self, PythonError, message):

        if self.COMM.Get_rank() == 0:
            raise PythonError(message)
        else:
            exit()

class qaoa(system):
    """
    Subclass of :class:`system`, this provides for the instantiation of a QAOA system distributed over an MPI communicator and the execution of this algorithm in parallel. Evolution of the QAOA system involves high-precision approximation of the action of the time-evolution operator on the QAOA state vector. This allows
    for the use of arbitrary mixing operators, :math:`W`, but is less efficient than
    :class:`qwoa` which makes use of a fast Fourier transform instead. If the user wishes to simulate the dynamics of a QAOA-like algorithm with a circulant mixing operator use of the :class:`qwoa` class is recommended. By default this class uses a hypercube mixing operator, :math:`W`, of size :math:`2^n \\times 2^n`.

    :param n_qubits: The number of qubits :math:`n`. For the :class:`~qaoa` class this sets the dimension of the mixing unitary, :math:`U_{\\text{W}}.`
    :type n_qubits: integer

    :param comm: MPI communicator objected created by mpi4py.
    :type comm: MPI communicator
    """
    def __init__(self, n_qubits, MPI_communicator, parallel = "global"):

        super().__init__(MPI_communicator, parallel = "global")

        self.n_qubits = n_qubits

        self.system_size = 2**n_qubits
        self.precision = "dp"

        self.mixing_operator_set_type = None

    def _generate_partition_table(self, N, MPI_communicator):

        flock = MPI_communicator.Get_size()

        partition_table = np.zeros(flock + 1, dtype = np.int32)
        for i in range(flock + 1):
            partition_table[i] = i * N / flock + 1

        remainder = N - partition_table[flock]

        for i in range(remainder):
            partition_table[flock - i % flock : flock + 1] += 1

        return partition_table

    def _csr_local_slice(self, W, MPI_communicator):

        if (sparse.issparse(W) and not sparse.isspmatrix_csr(W)):
            W_temp = W.tocsr()
        elif not sparse.issparse(W):
            try:
                W_temp = sparse.csr_matrix(W)
            except:
                print("Unable to convert input operator to csr_matrix.")
        else:
            W_temp = W

        W_row_starts = W_temp.indptr[self.lb:self.ub + 1].copy()
        W_col_indexes = W_temp.indices[W_row_starts[0]:W_row_starts[-1]].copy() + 1
        W_values = -I * W_temp.data[W_row_starts[0]:W_row_starts[-1]].copy()
        W_row_starts += 1

        return W_row_starts, W_col_indexes, W_values

    def set_graph(self, scipy_csr = None, function = mixers_mpi.hypercube):
        """
        Sets the operator, :math:`W`, used by the mixing unitary, :math:`U_{\\text{W}}`. This can be a :math:`2^n \\times 2^n` SciPy sparse CSR array, an array of :math:`2^n \\times 2^n` SciPy sprase CSR arrays or a python method which generates the mixing operator in parallel using MPI. An array of mixers allows for the simulation of mixing operators consisting of sequential non-commutative operators (:math:`U_{\\text{W}_1}, U_{\\text{W}_2},...`) each parameterised by the same :math:`t_i`. By default, this method generates a :math:`2^n \\times 2^n` sized hypercube using the 'method' option.

        To produce the mixing operator in parallel pass a method which takes the following arguments:
            * the number of qubits (integer)
            * The lower bound of the local row-wise partition of :math:`W` (as given by self.partition_table).
            * The upper bound of the local row-wise parition of :math:`W` (as given by self.partition_table).

        This method must return arrays (or an array of arrays) which describe the (MPI rank) local row-wise partition of the distributed CSR array(s) in the SciPy sparse CSR format: indptr, indices and values. See the SciPy CSR `documentation <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ for more information on the CSR sparse format.


        :param scipy_csr: SciPy sparse matrix, or array of SciPy sprase matrices. Must be of size :math:`2^n \\times 2^n`.
        :type scipy_csr: complex, SciPy sparse array

        :param method: A method which produces a :math:`2^n \\times 2^n` CSR mixing opertor(s) in parallel using MPI, as described above.
        :param method: callable, optional

        """

        if scipy_csr is not None:

            self.W = scipy_csr
            self.mixing_operator_set_type = "scipy_csr"

        else:

            self.W = function
            self.mixing_operator_set_type = "function"


    def _gen_mixing_operator(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            if self.mixing_operator_set_type == "scipy_csr":

                if type(self.W) is list:

                    self.W_row_starts = []
                    self.W_col_indexes = []
                    self.W_values = []

                    for w in self.W:

                        w_row_starts, w_col_indexes, w_values = self._csr_local_slice(
                                w,
                                self.comm)

                        self.W_row_starts.append(w_row_starts)
                        self.W_col_indexes.append(w_col_indexes)
                        self.W_values.append(w_values)

                else:

                    self.W_row_starts, self.W_col_indexes, self.W_values = self._csr_local_slice(
                            self.W,
                            self.comm)

            else:

                self.W_row_starts, self.W_col_indexes, self.W_values = self.W(
                        self.n_qubits,
                        self.lb + 1,
                        self.ub)

                self.W_values *= -I

            if isinstance(self.W_row_starts[0], np.ndarray):

                self.W_num_rec_inds = []
                self.W_rec_disps = []
                self.W_num_send_inds = []
                self.W_send_disps = []
                self.W_local_col_inds = []
                self.W_rhs_send_inds = []
                self.one_norms = []
                self.num_norms = []

                for w_row_starts, w_col_indexes, w_values in zip(self.W_row_starts, self.W_col_indexes, self.W_values):

                    w_num_rec_inds, w_rec_disps, w_num_send_inds, w_send_disps = fMPI.rec_a(
                               self.system_size,
                               w_row_starts,
                               w_col_indexes,
                               self.partition_table,
                               self.comm.py2f())

                    self.W_num_rec_inds.append(w_num_rec_inds)
                    self.W_rec_disps.append(w_rec_disps)
                    self.W_num_send_inds.append(w_num_send_inds)
                    self.W_send_disps.append(w_send_disps)

                    w_local_col_inds, w_rhs_send_inds = fMPI.rec_b(
                            self.system_size,
                            np.sum(self.W_num_send_inds),
                            w_row_starts,
                            w_col_indexes,
                            w_num_rec_inds,
                            w_rec_disps,
                            w_num_send_inds,
                            w_send_disps,
                            self.partition_table,
                            self.COMM_OPT.py2f())

                    self.W_local_col_inds.append(w_local_col_inds)
                    self.W_rhs_send_inds.append(w_rhs_send_inds)

                    one_norms, num_norms = fMPI.one_norm_series(
                            self.system_size,
                            w_row_starts,
                            w_col_indexes,
                            -I * w_values,
                            w_num_rec_inds,
                            w_rec_disps,
                            w_num_send_inds,
                            w_send_disps,
                            w_local_col_inds,
                            w_rhs_send_inds,
                            self.partition_table,
                            self.COMM_OPT.py2f())

                    self.one_norms.append(one_norms)
                    self.num_norms.append(num_norms)

            else:

                self.W_num_rec_inds, self.W_rec_disps, self.W_num_send_inds, self.W_send_disps = fMPI.rec_a(
                           self.system_size,
                           self.W_row_starts,
                           self.W_col_indexes,
                           self.partition_table,
                           self.COMM_OPT.py2f())

                self.W_local_col_inds, self.W_rhs_send_inds = fMPI.rec_b(
                        self.system_size,
                        np.sum(self.W_num_send_inds),
                        self.W_row_starts,
                        self.W_col_indexes,
                        self.W_num_rec_inds,
                        self.W_rec_disps,
                        self.W_num_send_inds,
                        self.W_send_disps,
                        self.partition_table,
                        self.COMM_OPT.py2f())

                self.one_norms, self.num_norms = fMPI.one_norm_series(
                        self.system_size,
                        self.W_row_starts,
                        self.W_col_indexes,
                        self.W_values,
                        self.W_num_rec_inds,
                        self.W_rec_disps,
                        self.W_num_send_inds,
                        self.W_send_disps,
                        self.W_local_col_inds,
                        self.W_rhs_send_inds,
                        self.partition_table,
                        self.COMM_OPT.py2f())

    def pre(self):

        if self.colours[self.COMM.Get_rank()] != -1:

            self.rank = self.COMM_OPT.Get_rank()

            self.partition_table = self._generate_partition_table(self.system_size, self.COMM_OPT)

            self.lb = self.partition_table[self.rank] - 1
            self.ub = self.partition_table[self.rank + 1] - 1

            self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
            self.local_i_offset = self.partition_table[self.rank] - 1

            self.alloc_local = self.local_i

            if self.mixing_operator_set_type is None:
                self.set_graph()

            self._gen_mixing_operator()

    def evolve_state(self, gammas_ts):
        """
        Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """

        if self.colours[self.COMM.Get_rank()] != -1:

            self.final_state = self.initial_state

            if isinstance(self.W_row_starts[0], np.ndarray):

                for gamma, t in zip(gammas_ts[::2], gammas_ts[1::2]):

                    self.final_state = np.multiply(np.exp(-I * np.gamma * self.observables), self.final_state)

                    for i in range(len(self.W_row_starts)):

                        self.final_state = fMPI.step(
                                self.system_size,
                                self.local_i,
                                self.W_row_starts[i],
                                self.W_col_indexes[i],
                                self.W_values[i],
                                self.W_num_rec_inds[i],
                                self.W_rec_disps[i],
                                self.W_num_send_inds[i],
                                self.W_send_disps[i],
                                self.W_local_col_inds[i],
                                self.W_rhs_send_inds[i],
                                np.abs(t),
                                self.final_state,
                                self.partition_table,
                                self.num_norms[i],
                                self.one_norms[i],
                                self.COMM_OPT.py2f(),
                                self.precision)

            else:

                for gamma, t in zip(gammas_ts[::2], gammas_ts[1::2]):

                    self.final_state = np.multiply(np.exp(-I * gamma * self.observables), self.final_state)

                    self.final_state = fMPI.step(
                            self.system_size,
                            self.local_i,
                            self.W_row_starts,
                            self.W_col_indexes,
                            self.W_values,
                            self.W_num_rec_inds,
                            self.W_rec_disps,
                            self.W_num_send_inds,
                            self.W_send_disps,
                            self.W_local_col_inds,
                            self.W_rhs_send_inds,
                            np.abs(t),
                            self.final_state,
                            self.partition_table,
                            self.num_norms,
                            self.one_norms,
                            self.COMM_OPT.py2f(),
                            self.precision)

class qwoa(system):
    """
    The :class:`qwoa` class provides for the instantiation a QWAO configuration
    distributed over an MPI communicator and the execution of the QWAO algorithm in parallel.
    Evolution of the :class:`qwoa` state occurs via calls to the compiled Fortran library
    'fqwoa_mpi', which makes use of MPI enabled FFTW (Fastest Fourier Transform in the West).

    :param system_size: The number of qubits or dimension of the system operators.
    :type system_size: integer

    :param MPI_communicator: An MPI communicator provided via MPI4Py.
    :type MPI_communicator: MPI communicator.

    :param qubits: If qubits is True, system_size is the number of qubits, producing a system of size :math:`2^n`. Otherwise system_size is equal to :math:`n`, allowing for simulations with a non-integer number of qubits.
    """
    def __init__(self, system_size, MPI_communicator, qubits = True, parallel = "global"):

        super().__init__(MPI_communicator, parallel)

        self.system_size = system_size

        if qubits:
            self.system_size = 2**system_size
            self.n_qubits = system_size
        else:
            self.system_size = system_size
            self.n_qubits = np.log(self.system_size)/np.log(2.0)

        self.graph_array = None

    def set_graph(self, graph_array = None):
        """
        Given a 1D array representing the first row of a circulant matrix,
        this returns a 1D array of matrix eigenvalues corresponding to
        a row-wise partitioning of that matrix over the active MPI communicator.

        :param graph_array: The first row of a circulant matrix.
        :type graph_array: float, array
        """

        if graph_array is None:
            self.graph_array = np.ones(self.system_size, dtype = np.float64)
            self.graph_array[0] = 0
        else:
            self.graph_array = np.array(graph_array, dtype = np.float64)

    def _gen_lambdas(self):

        self.lambdas = fqwoa_mpi.graph_eigenvalues(self.graph_array,self.local_o,self.local_o_offset)

    def pre(self):


        self.rank = self.COMM_OPT.Get_rank()

        # When performing a parallel 1D-FFT using FFTW it may be the case that
        # the transformed array is distributed on the MPI communicator differently
        # from the input. fqwoa.mpi_local_size determines the size needed at each
        # MPI node to accommodate for this. Along with the number of the actual array
        # elements stored at each node and their offset relative to the 0-index
        # of the distributed array.

        local_sizes = fqwoa_mpi.mpi_local_size(self.system_size, self.COMM_OPT.py2f())

        self.alloc_local = local_sizes[0]
        self.local_i = local_sizes[1]
        self.local_i_offset = local_sizes[2]
        self.local_o = local_sizes[3]
        self.local_o_offset = local_sizes[4]

        self.final_state = np.empty(self.alloc_local, np.complex128)

        self.dummy_gammas = np.empty(1, dtype = np.float64)
        self.dummy_ts = np.empty(1, dtype = np.float64)
        self.dummy_observables = np.empty(1, dtype = np.float64)
        self.dummy_lambdas = np.empty(1, dtype = np.float64)

        """
        Calls FFTW subroutines which set up the ancillary data structures needed to
        efficiently perform 1D parallel Fourier and inverse Fourier transforms.
        """

        if self.initial_state_type == None:
            self.set_initial_state(name = "equal")

        self._gen_initial_state()

        fqwoa_mpi.qwoa_state(
                self.system_size,
                self.dummy_gammas,
                self.dummy_ts,
                self.dummy_observables,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.COMM_OPT.py2f(),
                1)

        if self.graph_array is None:
            self.set_graph()

        self._gen_lambdas()

    def evolve_state(self, gammas_ts):
        """
        Evolves the qwoa.initial_state to the qwoa.final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """

        if self.colours[self.COMM.Get_rank()] != -1:

            fqwoa_mpi.qwoa_state(
                    self.system_size,
                    gammas_ts[::2],
                    np.abs(gammas_ts[1::2]),
                    self.observables,
                    self.lambdas,
                    self.initial_state,
                    self.final_state,
                    self.COMM_OPT.py2f(),
                    0)


    def post(self):
        """
        Deallocates/frees ancillary arrays and pointers needed by FFTW.
        """

        if self.colours[self.COMM.Get_rank()] == -1:

            fqwoa_mpi.qwoa_state(
                    self.system_size,
                    self.dummy_gammas,
                    self.dummy_ts,
                    self.dummy_observables,
                    self.dummy_lambdas,
                    self.initial_state,
                    self.final_state,
                    self.COMM_OPT.py2f(),
                    -1)

    def save(self, file_name, config_name, action = "a"):
        """
        Save a QWAO system as described by :meth:`system.save`. This also saves the eigenvalues of the circulant mixing operator.

        :param file_name: Name of the file on disc.
        :type file_name: string

        :param file_name: Name of the saved configuration in the HDF5 file.
        :type file_name: string

        :param action: "a": append to an existing file or create a new file. "w": overwrite the file if it exists.
        :type action: string, optional

        .. warning::
            Information given in :meth:`~system.save` corresponding to correctly loading complex-valued data from a .h5 file created by QWAO_MPI
            also applies to the saved eigenvalues.
        """

        I
        if self.colours[self.COMM.Get_rank()] == 0:
            super().save(
                    file_name,
                    config_name,
                   action)

            fqwoa_mpi.save_dist_complex(
                    file_name,
                    self.config_name + str("/"),
                    "eigenvalues",
                    "a",
                    self.system_size,
                    self.local_o_offset,
                    self.lambdas[:self.local_o],
                    self.COMM_OPT.py2f())

            self.COMM_OPT.Barrier()

#class ansatz(system):

