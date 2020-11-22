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
    def __init__(self):

        self.log = False
        self.graph_set = False

        # Set-up the default optimiser.
        self.stop = False
        self.set_optimiser( 'scipy',
                {'method':'BFGS','tol':1e-3},
                            ['fun','nfev','success'])

    def set_optimiser(self, optimiser, optimiser_args, optimiser_log = None):
        """
        Defines the classical optimiser algorithm used, arguments passed to the optimiser and fields in the optimiser dictionary to write to the log file (when using :meth:`~system.log_results`). QuOp_MPI supports optimisers provided by SciPy through its minimize method `minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>` and optimisers provided by the `NLopt <https://nlopt.readthedocs.io/en/latest/>` package with respect to minmisation with scalar constraints through a SciPy-like interface.


        The default optimiser is the BFGS algorithm provided SciPy, which is set on instantiation of the :class:`~system` class as follows:

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

    def get_probabilities(self):
        """
        :math:`\\vec{p} = ( \langle s_i|\\vec{\gamma}, \\vec{t} \\rangle` ), i=0,N-1

        :return: Probability vector corresponding the the local `self.final_state` partition.
        :rtype: array, float
        """
        self.probabilities = np.abs(self.final_state[:self.local_i])**2
        return self.probabilities

    def get_state_norm(self):
        """
        Check that :math:`\langle \\vec{\gamma}, \\vec{t}|\\vec{\gamma}, \\vec{t} \\rangle = 1`.
        The result is returned to each MPI rank and should be equal to 1 within the limits of double machine precision. This is used to check for state validity.

        :return: Norm of the current `self.final_state`.
        :rtype: float
        """
        self.state_norm = self.comm.allreduce(np.sum(self.probabilities), op = MPI.SUM)
        return self.state_norm

    def expectation(self):
        """
        :math:`\langle Q \\rangle =  \langle \\vec{\gamma}, \\vec{t}|Q|\\vec{\gamma}, \\vec{t} \\rangle`

        :return: The expectation value of the quality matrix operator, returned to all MPI nodes.
        :rtype: float
        """
        self.get_probabilities()
        local_expectation = np.dot(self.probabilities, self.qualities)
        return self.comm.allreduce(local_expectation, op = MPI.SUM)

    def objective(self, gammas_ts):
        """
        :math:`f(\\vec{\gamma}, \\vec{t}) = \langle \\vec{\gamma}, \\vec{t} | Q |\\vec{\gamma}, \\vec{t} \\rangle` \
        - the function minimised by the calssical optimizer.


        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array

        """

        # During optimization the root processes controls parallel evaluation
        # through the passing of the self.stop parameter.

        self.stop = self.comm.bcast(self.stop, root = 0)

        if not self.stop:
            self.gammas_ts = self.comm.bcast(gammas_ts, root = 0)
            gammas, ts = np.split(self.gammas_ts, 2)
            self.evolve_state(gammas, ts)
            expectation = self.expectation()
            return expectation

    def execute(self, gammas_ts, seed = 0):
        """
        Execute the QAOA-like algorithm.

        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array
        """
        self.time = time()
        self.gammas_ts = gammas_ts

        self.gammas_ts = self.comm.bcast(self.gammas_ts, root = 0)

        self.p = len(gammas_ts)//2

        self.stop = False

        if not self.graph_set:
            self.set_graph()

        if self.comm.Get_rank() == 0:
            self.result = self.optimiser(self.objective,self.gammas_ts,**self.optimiser_args)
            self.stop = True
            self.objective(self.gammas_ts)
        else:
            while not self.stop:
                self.objective(self.gammas_ts)

        self.time = time() - self.time

        if self.log:
            self.log_update()

    def print_result(self):
        """
        Print the optimization result.
        """
        if self.comm.Get_rank() == 0:
            print(self.result)

    def set_initial_state(self, name = None, vertices = None, state = None, normalized = False):
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

        # In some cases rank 0 might have no local indicies.
        if self.local_i > 0:
            rank = self.comm.Get_rank()
        else:
            rank = self.comm.Get_size()

        self.lowest_rank = self.comm.allreduce(rank, op = MPI.MIN)

        if name == "equal":
            self.initial_state = np.ones(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
        elif name == "localized":
            self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
            if self.comm.Get_rank() == self.lowest_rank:
                self.initial_state[0] = 1.0
        elif name == "split":
            self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
            if self.comm.Get_rank() == self.lowest_rank:
                self.initial_state[0:2] = 1.0/np.sqrt(2.0)
        elif vertices is not None:
            self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.system_size))
            total_verticies = self.comm.allreduce(np.float64(len(vertices)), op = MPI.SUM)
            for vertex in vertices:
                self.initial_state[vertex] = 1.0/np.sqrt(total_verticies)
        elif state is not None:
            self.initial_state = np.zeros(self.alloc_local, dtype = np.complex128)
            self.initial_state[0:self.local_i] = np.array(state, np.complex128)
            if not normalized:
                normalization = self.comm.allreduce(np.sum(np.multiply(np.conjugate(state), state)), op = MPI.SUM)
                self.initial_state = self.initial_state/np.sqrt(normalization)

    def set_qualities(self, func, *args, **kwargs):
        """
        Sets the local, :math:`q_i`. Ideally, each local partition of :math:`\\vec{q}` should be generated in parallel. As such :meth:`~qwoa.qualities` accepts a function whose first three arguments are the size of the distributed qwoa state,
        the number of locally stored input elements and the offset of these elements relative to the 0-index of the distributed array. Example quality functions are included in :mod:`~qwoa_mpi.qualities`.

        :param func: Function with which to generate the local :math:`q_i`.
        :type func: callable

        :param args: Extra arguments to pass to the quality function.
        :type args: array, optional
        """
        self.qualities = func(self.system_size, self.local_i, self.local_i_offset, *args, **kwargs)

    def benchmark(
            self,
            ps,
            repeats,
            param_func,
            qual_func = None,
            state_func = None,
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
            The `param_func`, qual_func` and `state_func` must have the keyword argument 'seed'. This allows for a repeatable variation if :math:`(\\vec{\gamma}, \\vec{t}), q_i` and :math:`| s \\rangle` with each repetition at the same :math:`p`.
        """

        # param persist and qual func warning.
        first = True

        itter = 0

        previous_params = None
        for p in ps:
            if param_persist:
                best_p_result = np.finfo(dtype=np.float64).max
                result = None

            if verbose:
                if self.comm.Get_rank() == 0:
                    print('Starting p = ' + str(p) + ':')

            for i  in range(1, repeats + 1):

                np.random.seed(i + itter)

                if (not param_persist) or first:

                    if self.rank == 0:
                        self.gammas_ts = param_func(p, seed = i + itter)
                    else:
                        self.gammas_ts = None

                    self.gammas_ts = self.comm.bcast(self.gammas_ts, root = 0)

                else:

                    gammas, ts = np.split(previous_params, 2)
                    if self.rank == 0:
                        gamma_t = param_func(1,seed = i + itter)
                    else:
                        gamma_t = None

                    gamma_t = self.comm.bcast(gamma_t, root = 0)
                    self.gammas_ts = np.append(np.append(gammas, gamma_t[0]), np.append(ts, gamma_t[1]))

                if qual_func is not None:
                    self.set_qualities(qual_func, seed = i + itter, *args, **kwargs)
                if state_func is not None:
                    self.initial_state = state_func(p, seed = i + itter)

                if verbose:
                    if self.comm.Get_rank() == 0:
                        print(str(i) + ' of ' + str(repeats) + '...')

                self.execute(self.gammas_ts)

                itter += 1

                if param_persist:

                    if self.rank == 0:
                        result = self.result['fun']
                        x = self.result['x']
                    else:
                        result = None
                        x = None

                    result = self.comm.bcast(result, root = 0)
                    x = self.comm.bcast(x, root = 0)

                    if result < best_p_result:
                        best_p_result = result
                        best_p_params = x

                if self.comm.Get_rank() == 0:
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

    def log_results(self, filename, label, action = "a"):
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
        * qubits: Number of qubits.
        * p: :math:`p`.
        * objective_function: Final result of objective function minimization.
        * objective_evaluations: Number of objective function evalutions needed durring optimisation.
        * optimization_success: If the minimizer converged to its target tolerances.
        * state_norm: Norm of the final state. This should always equal 1 (within the limits of double precision accuracy).
        * simulation_time: In-program simultion time.
        * MPI_nodes: Number of mpi processes.
        """
        self.label = label
        self.log = True
        self.n_log_fields = 7

        if self.comm.Get_rank() == 0:
            if (os.path.exists(filename + ".csv") and action == "a"):
                self.logfile = open(filename + ".csv", "a", newline='')
                self.logfile_csv = csv.writer(self.logfile)
            else:
                headings = ['label','qubits','system_size','p','state_norm','simulation_time','MPI_nodes']
                if self.optimiser_log is not None:
                    for optimiser_log in self.optimiser_log:
                        headings.append(optimiser_log)

                self.logfile = open(filename + ".csv", "w")
                self.logfile_csv = csv.writer(self.logfile)
                self.logfile_csv.writerow(headings)

    def log_update(self):
        """
        Update a .csv log of QAOA algorithm performance, instantiated by :meth:`~system.log_results`.
        """
        self.state_norm = self.get_state_norm()

        if self.comm.Get_rank() == 0:

            log_output = [
                    self.label,
                    self.n_qubits,
                    self.system_size,
                    self.p,
                    self.state_norm,
                    self.time,
                    self.comm.size
                    ]

            if self.optimiser_log is not None:
                for optimiser_log in self.optimiser_log:
                    log_output.append(self.result[optimiser_log])

            self.logfile_csv.writerow(log_output)

            self.logfile.flush()

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
            │   ├── eigenvalues
            │   ├── qualitites

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

        self.config_name = config_name

        if self.comm.Get_rank() == 0:

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

            File.create_dataset(self.config_name + "/initial_phases", data = self.gammas_ts, dtype = np.float64)
            File.close()

        self.config_name = self.comm.bcast(self.config_name, root = 0)

        fqwoa_mpi.save_dist_complex(
                file_name,
                self.config_name + str("/"),
                "final_state",
                "a",
                self.system_size,
                self.local_i_offset,
                self.final_state[:self.local_i],
                self.comm.py2f())

        fqwoa_mpi.save_dist_real(
                file_name,
                self.config_name + str("/"),
                "qualities",
                "a",
                self.system_size,
                self.local_i_offset,
                self.qualities[:self.local_i],
                self.comm.py2f())

        self.comm.Barrier()

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
    def __init__(self, n_qubits, MPI_communicator):

        super().__init__()

        self.n_qubits = n_qubits
        self.system_size = 2**n_qubits
        self.comm = MPI_communicator
        self.rank = self.comm.Get_rank()
        self.precision = "sp"
        self.graph_set = False

        self.partition_table = self._generate_partition_table(self.system_size, self.comm)

        self.lb = self.partition_table[self.rank] - 1
        self.ub = self.partition_table[self.rank + 1] - 1

        self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
        self.local_i_offset = self.partition_table[self.rank] - 1

        self.alloc_local = self.local_i

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

    def set_graph(self, scipy_csr = None, method = mixers_mpi.hypercube):
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

            W = scipy_csr

            if type(W) is list:

                self.W_row_starts = []
                self.W_col_indexes = []
                self.W_values = []

                for w in W:

                    w_row_starts, w_col_indexes, w_values = self._csr_local_slice(
                            w,
                            self.comm)

                    self.W_row_starts.append(w_row_starts)
                    self.W_col_indexes.append(w_col_indexes)
                    self.W_values.append(w_values)

            else:

                self.W_row_starts, self.W_col_indexes, self.W_values = self._csr_local_slice(
                        W,
                        self.comm)

        else:

            self.W_row_starts, self.W_col_indexes, self.W_values = method(
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
                        self.comm.py2f())

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
                        self.comm.py2f())

                self.one_norms.append(one_norms)
                self.num_norms.append(num_norms)

        else:

            self.W_num_rec_inds, self.W_rec_disps, self.W_num_send_inds, self.W_send_disps = fMPI.rec_a(
                       self.system_size,
                       self.W_row_starts,
                       self.W_col_indexes,
                       self.partition_table,
                       self.comm.py2f())

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
                    self.comm.py2f())

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
                    self.comm.py2f())

        self.graph_set = True

    def evolve_state(self, gammas, ts):
        """
        Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """
        self.final_state = self.initial_state

        if isinstance(self.W_row_starts[0], np.ndarray):

            for gamma, t in zip(gammas, ts):

                self.final_state = np.multiply(np.exp(-I * np.abs(gamma) * self.qualities), self.final_state)

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
                            self.comm.py2f(),
                            self.precision)

        else:

            for gamma, t in zip(gammas, ts):

                self.final_state = np.multiply(np.exp(-I * np.abs(gamma) * self.qualities), self.final_state)

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
                        self.comm.py2f(),
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
    def __init__(self, system_size, MPI_communicator, qubits = True):

        super().__init__()

        self.system_size = system_size

        if qubits:
            self.system_size = 2**system_size
            self.n_qubits = system_size
        else:
            self.system_size = system_size
            self.n_qubits = np.log(self.system_size)/np.log(2.0)

        self.comm = MPI_communicator
        self.rank = self.comm.Get_rank()

        # When performing a parallel 1D-FFT using FFTW it may be the case that
        # the transformed array is distributed on the MPI communicator differently
        # from the input. fqwoa.mpi_local_size determines the size needed at each
        # MPI node to accommodate for this. Along with the number of the actual array
        # elements stored at each node and their offset relative to the 0-index
        # of the distributed array.

        local_sizes = fqwoa_mpi.mpi_local_size(self.system_size, self.comm.py2f())

        self.alloc_local = local_sizes[0]
        self.local_i = local_sizes[1]
        self.local_i_offset = local_sizes[2]
        self.local_o = local_sizes[3]
        self.local_o_offset = local_sizes[4]

        self.final_state = np.empty(self.alloc_local, np.complex128)

        self.dummy_gammas = np.empty(1, dtype = np.float64)
        self.dummy_ts = np.empty(1, dtype = np.float64)
        self.dummy_qualities = np.empty(1, dtype = np.float64)
        self.dummy_lambdas = np.empty(1, dtype = np.float64)

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

        self.lambdas = fqwoa_mpi.graph_eigenvalues(self.graph_array,self.local_o,self.local_o_offset)
        self.graph_set = True

    def plan(self):
        """
        Calls FFTW subroutines which set up the ancillary data structures needed to
        efficiently perform 1D parallel Fourier and inverse Fourier transforms.
        """
        fqwoa_mpi.qwoa_state(
                self.system_size,
                self.dummy_gammas,
                self.dummy_ts,
                self.dummy_qualities,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.comm.py2f(),
                1)

    def evolve_state(self, gammas, ts):
        """
        Evolves the qwoa.initial_state to the qwoa.final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """
        fqwoa_mpi.qwoa_state(
                self.system_size,
                np.abs(gammas),
                np.abs(ts),
                self.qualities,
                self.lambdas,
                self.initial_state,
                self.final_state,
                self.comm.py2f(),
                0)

    def destroy_plan(self):
        """
        Deallocates/frees ancillary arrays and pointers needed by FFTW.
        """
        fqwoa_mpi.qwoa_state(
                self.system_size,
                self.dummy_gammas,
                self.dummy_ts,
                self.dummy_qualities,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.comm.py2f(),
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
                self.comm.py2f())

        self.comm.Barrier()
