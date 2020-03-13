from mpi4py import MPI
import h5py
import numpy as np
from scipy.optimize import basinhopping, Bounds
import sys
import os
import quop_mpi.fqwoa_mpi as fqwoa_mpi
import quop_mpi.fMPI as fMPI
from time import time

I = np.complex(0,1)

class system(object):
    """
    Provides a framework for the simulation of a QAOA-like algorithm
    in parallel using MPI. To do so, this class must be used to form a
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

    * local_i: The number of array indices stored at a given rank.
    * local_i_offset: the offset of those local indices in the globally distributed array
    * alloc_local: The size of the local array partition. This may be greater than local_i, but the local array partition must be fully defined by the first local_i values.

    self.qualities must be defined by local_i, local_i_offset and be of size local_i.

    QWAO_MPI contains two :class:`system` subclasses: :class:`qaoa` and :class:`qwoa`.
    """
    def __init__(self):

        # set default variables used by state_success.
        self.quality_cutoff = 0.9
        self.success_target = 2.0/3.0
        self.log = False

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
        :math:` \langle Q \\rangle =  angle \\vec{\gamma}, \\vec{t}|Q|\\vec{\gamma}, \\vec{t} \\rangle`

        :return: The expectation value of the quality matrix operator, returned to all MPI nodes.
        :rtype: float
        """
        self.get_probabilities()
        local_expectation = np.dot(self.probabilities, self.qualities)
        return self.comm.allreduce(local_expectation, op = MPI.SUM)

    def objective(self, gammas_ts, stop):
        """
        :math:`f(\\vec{\gamma}, \\vec{t}) = \\frac{q_{max} - \langle \\vec{\gamma}, \\vec{t} | Q |\\vec{\gamma}, \\vec{t} \\rangle}{q_{max}}` \
        - the function minimised by the calssical optimizer.


        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array

        """

        # During optimization the root processes controls parallel evaluation
        # through the passing of the self.stop parameter.

        self.stop = self.comm.bcast(stop, root = 0)

        if not self.stop:
            self.gammas_ts = self.comm.bcast(gammas_ts, root = 0)
            gammas, ts = np.split(self.gammas_ts, 2)
            self.evolve_state(gammas, ts)
            expectation = self.expectation()
            return (np.abs(self.max_quality) - expectation)/np.abs(self.max_quality)

    def execute(self, gammas_ts, seed = 0):
        """
        Execute the QAOA-like algorithm.

        :param gammas_ts: An array of length :math:`2 p`, :math:`(\\vec{\gamma},\\vec{t})`.
        :type gammas_ts: float, array
        """
        self.time = time()
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

        self.time = time() - self.time

        if self.log:
            self.state_success(self.quality_cutoff, self.success_target)
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
            * Fully specify :math:`|s\\rangle`. Pass an MPI rank specific array of :math:`x_i \in \mathbb{C}` such that :math:`x_i    \geq 0' and  :math:``\\text{local_i} \leq i < \\text{local_i + local_i_offset}`. If keyword `normalized` is True, this state will be normalized.

        :param name: Name of a pre-defined initial state.
        :type name: str, optional

        :param vertices: Specify an equal superposition over a set of :math:`|s_i\\rangle`.
        :type verticies: array, integer, optional

        :state: Initialize :math:`|s\\rangle` in a user-defined generalized state.
        :param: array, float, optional

        :state normalized: If normalized is True and an argument to `state`is specified, normalize the input state.l
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
            total_verticies = self.comm.allreduce(np.float64(len(vertices)), op = MPI.SUM)
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
        Sets the local, :math:`q_i`. Ideally, each local partition of :math:`\\vec{q}` should be generated in parallel. As such :meth:`~qwoa.qualities` accepts a function whose first three arguments are the size of the distributed qwoa state,
        the number of locally stored input elements and the offset of these elements relative to the 0-index of the distributed array. Example quality functions are included in :mod:`~qwoa_mpi.qualities`.

        :param func: Function with which to generate the local :math:`q_i`.
        :type func: callable

        :param args: Extra arguments to pass to the quality function.
        :type args: array, optional
        """
        self.qualities = func(self.size, self.local_i, self.local_i_offset, *args, **kwargs)

        if sign == "negative":
            if len(self.qualities) == 0:
                local_max = np.finfo(np.float64).max
            else:
                local_max = np.min(self.qualities)
            self.max_quality = self.comm.allreduce(local_max, op = MPI.MIN)
        else:
            if len(self.qualities) == 0:
                local_max = np.finfo(np.float64).min
            else:
                local_max = np.max(self.qualities)
            self.max_quality = self.comm.allreduce(local_max, op = MPI.MAX)

    def state_success(self, quality_cutoff, success_target):
        """
        A rough method for judging the effectiveness of a QAOA algorithm.

        :param quality_cutoff: Between 0 and 1. 0.9 corresponds to seeking solutions with quality in the top 10%.
        :type quality_cutoff: float

        :param success_target: Target cumulative chance of measuring a result above the quality_cutoff.

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
            *args,
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

        first = True

        for p in ps:

            if verbose:
                if self.comm.Get_rank() == 0:
                    print('Starting p = ' + str(p) + ':')

            for i  in range(1, repeats + 1):

                np.random.seed(i)

                self.gammas_ts = param_func(p, seed = i)

                if qual_func is not None:
                    self.set_qualities(qual_func, seed = i, *args, **kwargs)
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

        Once called, addtional calls to :meth:`~system.log_update` will save the following information:

        * label: User-defined system label.
        * qubits: Number of qubits.
        * p: :math:`p`.
        * quality_cutoff: As defined by :meth:`~system.state_success`.
        * objective_function: Final result of objective function minimization.
        * state_norm: Norm of the final state. This should always equal 1 (within the limits of double precision accuracy).
        * time: In-program simultion time.
        """
        self.label = label
        self.log = True

        if self.comm.Get_rank() == 0:
            if (os.path.exists(filename + ".csv") and action == "a"):
                self.logfile = open(filename + ".csv", "a")
            else:
                self.logfile = open(filename + ".csv", "w")
                self.logfile.write(
                        'label,qubits,p,quality_cutoff,success_probability,success,quality_cutoff,objective_function,state_norm,time\n')

    def log_update(self):
        """
        Update a .csv log of QAOA algorithm performance, instantiated by :meth:`~system.log_success`.
        """
        self.state_norm = self.get_state_norm()

        if self.comm.Get_rank() == 0:

            self.logfile.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                self.label,
                self.n_qubits,
                self.p,
                self.quality_cutoff,
                self.success_probability,
                self.success,
                self.quality_cutoff,
                self.result['fun'],
                self.state_norm,
                self.time))

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

        fqwoa_mpi.save_dist_complex(
                file_name,
                self.config_name + str("/"),
                "final_state",
                "a",
                self.size,
                self.local_i_offset,
                self.final_state[:self.local_i],
                self.comm.py2f())

        fqwoa_mpi.save_dist_real(
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
    Subclass of :class:`system`, this provides for the instantiation of a QAOA system distributed over an MPI communicator and the execution of this algorithm in parallel. Evolution of the QAOA system involves high-precision approximation of the action of the time-evolution operator on the QAOA state vector. This allows
    for the use of arbitrary mixing operators, :math:`W`, but is less efficient than
    :class:`qwoa` which makes use of a fast Fourier transform instead. If the user wishes to simulate the dynamics of a QAOA-like algorithm with a circulant mixing operator use of the :class:`qwoa` class is recommended.

    :param W: Arbitrary :math:`N \\times N` mixing operator, :math:`W`.
    :type W: SciPy sparse CSR matrix

    :param comm: MPI communicator objected created by mpi4py.
    :type comm: MPI communicator
    """
    def __init__(self, W, comm):

        super().__init__()

        self.size = W.shape[0]
        self.n_qubits = int(np.log(self.size)/np.log(2.0))
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.precision = "dp"

        self.partition_table = self._generate_partition_table(self.size, self.comm)

        self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
        self.local_i_offset = self.partition_table[self.rank] - 1

        self.alloc_local = self.local_i

        self.W_row_starts, self.W_col_indexes, self.W_values = self._csr_local_slice(
                W,
                self.partition_table,
                self.comm)

        self.W_num_rec_inds, self.W_rec_disps, self.W_num_send_inds, self.W_send_disps = fMPI.rec_a(
                   self.size,
                   self.W_row_starts,
                   self.W_col_indexes,
                   self.partition_table,
                   self.comm.py2f())

        self.W_local_col_inds, self.W_rhs_send_inds = fMPI.rec_b(
                self.size,
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
                self.size,
                self.W_row_starts,
                self.W_col_indexes,
                -I * self.W_values,
                self.W_num_rec_inds,
                self.W_rec_disps,
                self.W_num_send_inds,
                self.W_send_disps,
                self.W_local_col_inds,
                self.W_rhs_send_inds,
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

    def _csr_local_slice(self, W, partition_table, MPI_communicator):

        rank = MPI_communicator.Get_rank()

        lb = partition_table[rank] - 1
        ub = partition_table[rank + 1] - 1

        W_row_starts = W.indptr[lb:ub + 1]
        W_col_indexes = W.indices[W_row_starts[0]:W_row_starts[-1]] + 1
        W_values = W.data[W_row_starts[0]:W_row_starts[-1]]
        W_row_starts += 1

        return W_row_starts, W_col_indexes, W_values

    def evolve_state(self, gammas, ts):
        """
        Evolves the QAOA initial_state to its final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """
        self.final_state = self.initial_state

        for gamma, t in zip(gammas, ts):

            self.final_state = np.multiply(np.exp(-I * gamma * self.qualities), self.final_state)

            self.final_state = fMPI.step(
                    self.size,
                    self.local_i,
                    self.W_row_starts,
                    self.W_col_indexes,
                    -I * self.W_values,
                    self.W_num_rec_inds,
                    self.W_rec_disps,
                    self.W_num_send_inds,
                    self.W_send_disps,
                    self.W_local_col_inds,
                    self.W_rhs_send_inds,
                    t,
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

    :param n_qubits: The number of qubits, :math:`n`, the total distributed system is of size :math:`n^2`.
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
        # from the input. fqwoa.mpi_local_size determines the size needed at each
        # MPI node to accommodate for this. Along with the number of the actual array
        # elements stored at each node and their offset relative to the 0-index
        # of the distributed array.

        local_sizes = fqwoa_mpi.mpi_local_size(self.size, self.comm.py2f())

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
        this returns a 1D array of matrix eigenvalues corresponding to
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
        fqwoa_mpi.qwoa_state(
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
        Evolves the qwoa.initial_state to the qwoa.final_state.

        :param gammas: Quality-proportional phase shifts.
        :type gammas: float, array

        :param ts: Continuous-time quantum walk times.
        :type ts: float, array
        """
        fqwoa_mpi.qwoa_state(
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
        fqwoa_mpi.qwoa_state(
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
                self.size,
                self.local_i_offset,
                self.lambdas,
                self.comm.py2f())
