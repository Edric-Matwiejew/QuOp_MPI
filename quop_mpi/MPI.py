from mpi4py import MPI
import h5py
import numpy as np
from scipy.optimize import basinhopping, Bounds
import sys
import os
import quop_mpi.fqwao_mpi as fqwao_mpi
import quop_mpi.fMPI as fMPI

I = np.complex(0,1)

class system(object):
    """
    Provides a framework for the simulation of a QAOA-like algorithm
    in parallel using MPI. To do so this class must be used to form a
    subclass containing an 'evolve_state' method:

    .. code-block:: python

        class(system):

            def __init__(self, n_qubits, MPI_communicator):

                ...

            def evolve_state(self, gammas, ts):

                self.initial_state = self.final_state

                self.final_state = ...

    The 'evolve_state' method must modify self.final_state in place.
    self.final_state and self.initial_state must be distributed over an MPI
    communicator and described by the following variables:

    * local_i: The number of array indicies stored at a given rank.
    * local_i_offset: the offset of those local indcies in the globally distributed array
    * alloc_local: The size of the local array partition. This may be greater than local_i, but the local array partition must be fully defined by the first local_i values.

    self.qualities must be defined by local_i, local_i_offset and be of size local_i.

    QWAO_MPI contains two :class:`system` subclasses: :class:`qaoa` and :class:`qwao`.
    """
    def __init__(self):

        # set variables used by state_success.
        self.quality_cutoff = 0.9
        self.success_target = 2.0/3.0

    def get_probabilities(self):
        """
        Calculate the probability distributed of the current self.final_state.
        """
        self.probabilities = np.abs(self.final_state[:self.local_i])**2
        return self.probabilities

    def get_state_norm(self):
        """
        Calculate the norm of the current self.final_state. The value is returned
        to each MPI rank and can be used to check for state validity.
        """
        self.state_norm = self.comm.allreduce(np.sum(self.probabilities), op = MPI.SUM)
        return self.state_norm

    def expectation(self):
        """
        Returns the expectation of the qwao final state to all MPI processes.
        """
        self.get_probabilities()
        local_expectation = np.dot(self.probabilities, self.qualities)
        return self.comm.allreduce(local_expectation, op = MPI.SUM)

    def objective(self, gammas_ts, stop):
        """
        Objection funtion to minimise as part of the QWAO algorithm.

        :param gammas_ts: Starting angles, an array of size :math:`2 \\times p`.
        :type gammas_ts: float, array

        """

        self.stop = self.comm.bcast(stop, root = 0)

        if not self.stop:
            self.gammas_ts = self.comm.bcast(gammas_ts, root = 0)
            gammas, ts = np.split(self.gammas_ts, 2)
            self.evolve_state(gammas, ts)
            expectation = self.expectation()
            return (np.abs(self.max_quality) - expectation)/np.abs(self.max_quality)

    def execute(self, gammas_ts, seed = 0):
        """
        Execute the QWAO algorithm.

        :param gammas_ts: Starting angles, an array of size :math:`2 \\times p`.
        :type gammas_ts: float, array
        """
        self.gammas_ts = gammas_ts
        self.p = len(gammas_ts)//2

        bounds = Bounds(0, np.inf)

        self.stop = False
        if self.comm.Get_rank() == 0:
            self.result = basinhopping(
                    self.objective,
                    gammas_ts,
                    stepsize = 0.1,
                    niter = 100,
                    seed = 1,
                    minimizer_kwargs = {'method':'L-BFGS-B','bounds':bounds,'args':(self.stop,)})
            self.stop = True
            self.objective(gammas_ts, self.stop)
        else:
            while not self.stop:
                self.objective(gammas_ts,self.stop)

        if self.log:
            self.state_success(self.quality_cutoff, self.success_target)
            self.log_update()

    def print_result(self):
        if self.comm.Get_rank() == 0:
            print(self.result)

    def set_initial_state(self, name = None, vertices = None, state = None, normalized = False):
        """
        Sets the initial state used in the QWAO algorithm.
        """

        if name == "equal":
            self.initial_state = np.ones(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.size))
        elif name == "localized":
            self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.size))
            if self.comm.Get_rank() == 0:
                self.initial_state[0] = 1.0
        elif name == "split":
            self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.size))
            if self.comm.Get_rank() == 0:
                self.initial_state[0:2] = 1.0/np.sqrt(2.0)
        elif vertices is not None:
            self.initial_state = np.zeros(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.size))
            total_verticies = comm.allreduce(np.float64(len(vertices)), op = MPI.SUM)
            for vertex in vertices:
                self.initial_state[vertex] = 1.0/np.sqrt(total_verticies)
        elif state is not None:
            self.initial_state = np.zeros(self.alloc_local, dtype = np.complex128)
            self.initial_state[0:self.local_i] = np.array(state, np.complex128)
            if not normalized:
                normalization = self.comm.allreduce(np.sum(np.multiply(np.conjugate(state), state)), op = MPI.SUM)
                self.initial_state = self.initial_state/np.sqrt(normalization)

    def set_qualities(self, func, sign = "positive", *args, **kwargs):
        """
        Sets the qualities in the QWAO algorithm. As the array of qualities is
        equal to the size of the QWAO state, ideally the qualities should be
        generated in parallel. As such :meth:`~qwao.qualities` accepts a function
        whose first three arguments are the size of the distributed qwao state,
        the number of locally stored stored input elements and the offset of these
        elements relative to the 0-index of the distributed array. Example quality
        functions are included in :mod:`~qwao_mpi.qualities`.

        :param func: Function with which to generate the local qualities.
        :type func: callable

        :param args: Extra arguments to pass to the quality function.
        :type args: array, optional
        """
        self.qualities = func(self.size, self.local_i, self.local_i_offset, *args, **kwargs)

        if sign == "negative":
            self.max_quality = self.comm.allreduce(np.min(self.qualities), op = MPI.MIN)
        else:
            self.max_quality = self.comm.allreduce(np.max(self.qualities), op = MPI.MAX)

    def state_success(self, quality_cutoff, success_target):
        """
        A rough method for judging the effectiveness of an QAOA algorithm.

        :param quality_cutoff: Between 0 and 1. 0.9 corresponds to seeking solutions with a quality in the top 10%.
        :type quality_cutoff: float

        :param success_target: Target cummulative chance of measuring a result above the quality_cutoff.

        :return: success, success_probability
        :rtype: boolean, float

        With 'quality_cutoff = 0.9', 'success_target = 2/3' and return values of 'True', 0.7 and 5'. The algorithm
        has succeeded in having a greater than :math:`\\frac{2}{3}` chance of measuring a state corresponding to a solution
        in the top 10%.
        """
        self.quality_cutoff = quality_cutoff
        self.success_target = success_target
        self.success_probability = 0.0

        for i, prob in enumerate(self.probabilities):
            if self.qualities[i]/self.max_quality >= self.quality_cutoff:
                self.success_probability += prob

        self.success_probability = self.comm.allreduce(self.success_probability, op = MPI.SUM)

        if self.success_probability >= self.success_target:
            self.success = True
        else:
            self.success = False

        return self.success, self.success_probability

    def benchmark(
            self,
            ps,
            repeats,
            param_func,
            qual_func = None,
            state_func = None,
            early_stopping = False,
            verbose = True,
            filename = None,
            label = 'test',
            save_action = "a",
            **kwargs):

        """
        A convience method provided for users who wish to see how a QAOA algorithm
        performs with increases to the circuit depth, :math:`p`.

        :param ps: List of :math:`p` values.
        :type ps: integer, array

        :param repeats: The number of repeats at each value of :math:`p`.
        :type repeats: integer

        The following three parameters allow the user to specify functions to generate
        unique starting conditions with each repeat. Of these `param_func` is required
        as :math:`(\\vec{\gamma},\\vec{t})` grows with :math:`p`.

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

        :param early_stopping: If True, stop repeated evaluation for a given :math:`p` if :meth:`~system.state_success` returns 'self.success' as True.
        :type early_stopping: boolean, optional, default = False

        :param verbose: If True, print current :math:`p`, repitition number and optimization results.
        :type verbose: boolean, optional, default = True

        :param filename: Name of .h5 file in which to :meth:`~qwao.system.save` the evolved system.
        :type filename: string, optional, default = None

        :param label: If filename is specified, evolved systems will be saved as 'filename/label_p_repetition'.
        :type label: string, optional, default = 'test'

        :param save_action: Action taken durring first .5 file write. "a", append. "w", over-write.
        :type save_action: string, optional, default = "a"

        :param kwargs: Keyword arguments to pass to the :meth:`~system.set_qualities` method.
        :type kwargs: dictionary, optional
        """

        first = True

        for p in ps:

            if verbose:
                if self.comm.Get_rank() == 0:
                    print('Starting p = ' + str(p) + ':')

            for i  in range(1, repeats + 1):

                np.random.seed(i)

                self.gammas_ts = param_func(p, seed = i)

                if qual_func is not None:
                    self.set_qualities(qual_func, seed = i)
                if state_func is not None:
                    self.initial_state = state_func(p, seed = i)

                if verbose:
                    if self.comm.Get_rank() == 0:
                        print(str(i) + ' of ' + str(repeats) + '...')

                self.execute(self.gammas_ts)

                if self.comm.Get_rank() == 0:
                    if verbose:
                        print(self.result)

                if early_stopping and (self.success_probability >= self.success_target):
                    break

                if filename is not None:
                    if first:
                        self.save(filename, label + '_' + str(p) + '_' + str(i), action = save_action)
                    else:
                        self.save(filename, label + '_' + str(p) + '_' + str(i), action = "a")

    def log_success(self, filename, label, action = "a"):
        """
        Creates a .csv in which to save key QAOA results after a call to :meth:`~system.execute`.

        :param filename: Name of the .csv file.
        :type filename: string

        :param label: User-set identifier of the currently defined system.
        :type label: string

        :param action: "a", append. "w", over-write.
        :type action: string, optional, default = "a"

        One called, addtional calls to :meth:`~system.log_update` will save the following information:

        * label: User-defined system label.
        * qubits: Number of qubits.
        * p: :math:`p`.
        * quality_cutoff: As defined by :meth:`~system.state_success`.
        * objective_function: Final result of objective function minimization.
        * state_norm: Norm of the final state. This should always equal 1 (within the limits of double precision accuracy).
        """
        self.label = label
        self.log = True

        if self.comm.Get_rank() == 0:
            if (os.path.exists(filename + ".csv") and action == "a"):
                self.logfile = open(filename + ".csv", "a")
            else:
                self.logfile = open(filename + ".csv", "w")
                self.logfile.write(
                        'label,qubits,p,quality_cutoff,success_probability,success,quality_cutoff,objective_function,state_norm\n')

    def log_update(self):
        """
        Update a .csv log of QAOA algorithm performance, instantiated by :meth:`~system.log_success`.
        """
        self.state_norm = self.get_state_norm()

        if self.comm.Get_rank() == 0:

            self.logfile.write('{},{},{},{},{},{},{},{},{}\n'.format(
                self.label,
                self.n_qubits,
                self.p,
                self.quality_cutoff,
                self.success_probability,
                self.success,
                self.quality_cutoff,
                self.result['fun'],
                self.state_norm))

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

        Data it saved into a .h5 file with the following structure.

        ::

            file_name.h5
            ├── config_name
            │   ├── final_state
            │   ├── eigenvalues
            │   ├── qualitites
            │   ├── minimize_results
            │   │   ├── result_field_1
            │   │   ├── result_field_2
            │   │   ├── result_field_3
            │   │   ├── ...

        Multiple configurations with a unique config_name can be stored in the same .h5 file.
        HDF5 files are supported in python by the `h5py <https://www.h5py.org/>`_ package. With it,
        a saved configuration can be accessed as follows:

        .. code-block:: python

            import h5py

            f = h5py.File(file_name + ".h5", "r")
            final_state = np.array(f[config_name]['final_state']).view(np.complex128)
            eigenvalues = np.array(f[config_name]['eigenvalues']).view(np.complex128)
            qualities = np.array(f[config_name]['qualities']).view(np.float64)

            minimize_result = {}
            for key in f[config_name]['minimize_result'].keys():
                minimize_result[key] = np.array(f[config_name]['minimize_result'][key])

        .. warning::
            The final_state and qualities datasets are saved using Fortran
            subroutines which make use of parallel HDF5.

            The complex values of the final_state array are saved as a
            compound datatype consisting of contiguous double precision reals. This is
            equivalent to the np.complex128 numpy datatype. To access this data without a
            loss of precision in python the user must set the **view** of the numpy array
            to np.complex128, rather than casting it to np.complex128 using the dtype keyword.

            Similarly the qualities array, which is saved as an array of double precision
            reals, should have its view set to np.float64.
        """

        self.config_name = config_name

        if self.comm.Get_rank() == 0:

            File = h5py.File(file_name + ".h5", "a")

            # If the config_name already exists in the target file, add an underscore.
            duplicate = True
            while duplicate:
                if self.config_name in File:
                    self.config_name += "_"
                else:
                    duplicate = False

            config = File.create_group(self.config_name)
            minimize_result = config.create_group("minimize_result")

            for key in self.result.keys():

                try:

                    minimize_result.create_dataset(key, data = self.result.get(key))
                except:

                    if verbose:
                        print("No native HDF5 type for " + \
                                str(type(self.result.get(key))) + \
                                ". Minimization result field " + \
                                key + "  not saved.", file = sys.stderr)

            File.create_dataset(self.config_name + "/initial_phases", data = self.gammas_ts, dtype = np.float64)
            File.close()

        self.config_name = self.comm.bcast(self.config_name, root = 0)

        fqwao_mpi.save_dist_complex(
                file_name,
                self.config_name + str("/"),
                "final_state",
                action,
                self.size,
                self.local_i_offset,
                self.final_state[:self.local_i],
                self.comm.py2f())

        fqwao_mpi.save_dist_real(
                file_name,
                self.config_name + str("/"),
                "qualities",
                "a",
                self.size,
                self.local_i_offset,
                self.qualities,
                self.comm.py2f())

        self.comm.Barrier()

class qaoa(system):
    """
    Subclass of :class:`system`, this provides for the instantiation of a QAOA system
    distributed over an MPI communicator and the execution of this algorithm in
    parallel. Evolution of the QAOA system involves high-precision approximation
    of the action of the time-evolution operator on the QAOA state vector. This allows
    for use of arbitrary mixing operators, :math:`H_c`, but is less effcient than
    :class:`qwao` which makes use of a fast Fourier transfrom instead. If the user
    wishes to simulate the dynamics of a QAOA-like algorithm with a circulant mixing
    operator use of the :class:`qwao` class is recommend.

    :param Hc: Arbitrary :math:`N \\times N` mixing operator, :math:`H_c`.
    :type Hc: SciPy sparse CSR matrix

    :param comm: MPI communicator objected created by mpi4py.
    :type comm: MPI communicator
    """
    def __init__(self, Hc, comm):

        super().__init__()

        self.size = Hc.shape[0]
        self.n_qubits = int(np.log(self.size)/np.log(2.0))
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.precision = "dp"

        self.partition_table = self._generate_partition_table(self.size, self.comm)

        self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
        self.local_i_offset = self.partition_table[self.rank] - 1

        self.alloc_local = self.local_i

        self.Hc_row_starts, self.Hc_col_indexes, self.Hc_values = self._csr_local_slice(
                Hc,
                self.partition_table,
                self.comm)

        self.Hc_num_rec_inds, self.Hc_rec_disps, self.Hc_num_send_inds, self.Hc_send_disps = fMPI.rec_a(
                   self.size,
                   self.Hc_row_starts,
                   self.Hc_col_indexes,
                   self.partition_table,
                   self.comm.py2f())

        self.Hc_local_col_inds, self.Hc_rhs_send_inds = fMPI.rec_b(
                self.size,
                np.sum(self.Hc_num_send_inds),
                self.Hc_row_starts,
                self.Hc_col_indexes,
                self.Hc_num_rec_inds,
                self.Hc_rec_disps,
                self.Hc_num_send_inds,
                self.Hc_send_disps,
                self.partition_table,
                self.comm.py2f())

        self.one_norms, self.num_norms = fMPI.one_norm_series(
                self.size,
                self.Hc_row_starts,
                self.Hc_col_indexes,
                -I * self.Hc_values,
                self.Hc_num_rec_inds,
                self.Hc_rec_disps,
                self.Hc_num_send_inds,
                self.Hc_send_disps,
                self.Hc_local_col_inds,
                self.Hc_rhs_send_inds,
                self.partition_table,
                self.comm.py2f())

    def _generate_partition_table(self, N, MPI_communicator):

        flock = MPI_communicator.Get_size()

        partition_table = np.zeros(flock + 1, dtype = np.int32)
        for i in range(flock + 1):
            partition_table[i] = i * N / flock + 1

        remainder = N - partition_table[flock]

        for i in range(remainder):
            partition_table[flock - i % flock : flock + 1] += 1

        return partition_table

    def _csr_local_slice(self, Hc, partition_table, MPI_communicator):

        rank = MPI_communicator.Get_rank()

        lb = partition_table[rank] - 1
        ub = partition_table[rank + 1] - 1

        Hc_row_starts = Hc.indptr[lb:ub + 1]
        Hc_col_indexes = Hc.indices[Hc_row_starts[0]:Hc_row_starts[-1]] + 1
        Hc_values = Hc.data[Hc_row_starts[0]:Hc_row_starts[-1]]
        Hc_row_starts += 1

        return Hc_row_starts, Hc_col_indexes, Hc_values

    def evolve_state(self, gammas, ts):
        """
        Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continous-time quantum walk times.
        :type ts: float, array
        """
        self.final_state = self.initial_state

        for gamma, t in zip(gammas, ts):

            self.final_state = np.multiply(np.exp(-I * gamma * self.qualities), self.final_state)

            self.final_state = fMPI.step(
                    self.size,
                    self.local_i,
                    self.Hc_row_starts,
                    self.Hc_col_indexes,
                    -I * self.Hc_values,
                    self.Hc_num_rec_inds,
                    self.Hc_rec_disps,
                    self.Hc_num_send_inds,
                    self.Hc_send_disps,
                    self.Hc_local_col_inds,
                    self.Hc_rhs_send_inds,
                    t,
                    self.final_state,
                    self.partition_table,
                    self.num_norms,
                    self.one_norms,
                    self.comm.py2f(),
                    self.precision)

class qwao(system):
    """
    The :class:`qwao` class provides for the instantiation a QWAO configuration
    distributed over an MPI communicator and the execution of the QWAO algorithm in parallel.
    Evolution of the :class:`qwao` state occurs via calls to the compiled Fortran library
    'fqwao_mpi', which makes use of MPI enabled FFTW (Fastest Fourier Transform in the West).

    :param n_qubits: The number of qubits, :math:`n`, total distributed system is of size :math:`n^2`.
    :type n_qubits: integer

    :param MPI_communicator: An MPI communicator provided via MPI4Py.
    :type MPI_communicator: MPI communicator.
    """
    def __init__(self, n_qubits, MPI_communicator):

        super().__init__()

        self.n_qubits = n_qubits
        self.size = 2**n_qubits
        self.comm = MPI_communicator

        # When performing a parallel 1D-FFT using FFTW it may be the case that
        # the transformed array is distributed on the MPI communicator differently
        # from the input. fqwao.mpi_local_size determines the size needed at each
        # MPI node to accomodate for this. Along with the number of actual array
        # elements stored at each node and their offset relative to the 0-index
        # of the distributed array.

        local_sizes = fqwao_mpi.mpi_local_size(self.size, self.comm.py2f())

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

    def set_graph(self, graph_array):
        """
        Given a 1D array representing the first row of a circulant matrix,
        this returns a 1D array of matrix eigenvalues corresponding to a
        a row-wise partitioning of that matrix over the active MPI communicator.

        :param graph_array: The first row of a circulant matrix.
        :type graph_array: float, array
        """

        self.graph_array = graph_array
        self.lambdas = np.zeros(self.local_o, np.complex)

        for i in range(self.local_o_offset, self.local_o_offset + self.local_o):
            for j in range(len(graph_array)):
                self.lambdas[i - self.local_o_offset] = self.lambdas[i - self.local_o_offset] \
                        + np.exp((2.0j*np.pi*i)/float(len(graph_array)))**(j)*graph_array[j]

    def plan(self):
        """
        Calls FFTW subroutines which set up the ancillary data structures needed to
        efficiently perform 1D parallel Fourier and inverse Fourier transforms.
        """
        fqwao_mpi.qwao_state(
                self.size,
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
        Evolves the qwao.initial_state to the qwao.final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continous-time quantum walk times.
        :type ts: float, array
        """
        fqwao_mpi.qwao_state(
                self.size,
                gammas,
                ts,
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
        fqwao_mpi.qwao_state(
                self.size,
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
        Save a QWAO system as described by :meth:`system.save`. This also saves the
        eigenvalues of the circlant mixing operator.

        :param file_name: Name of the file on disc.
        :type file_name: string

        :param file_name: Name of the saved configuration in the HDf5 file.
        :type file_name: string

        :param action: "a": append to an existing file or create a new file. "w": overwrite the file if it exists.
        :type action: string, optional

        .. warning::
            Information given in :meth:`~system.save` corresponding to correctly loading complex valued data from a .h5 file created by QWAO_MPI
            also applies to the saved eigenvalues.
        """

        super().save(
                file_name,
                config_name,
                action)

        fqwao_mpi.save_dist_complex(
                file_name,
                self.config_name + str("/"),
                "eigenvalues",
                "a",
                self.size,
                self.local_i_offset,
                self.lambdas,
                self.comm.py2f())
