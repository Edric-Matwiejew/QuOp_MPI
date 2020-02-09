from mpi4py import MPI
import h5py
import numpy as np
from scipy.optimize import basinhopping
import sys
import qwao_mpi.fqwao_mpi as fqwao_mpi
import qwao_mpi.fMPI as fMPI

class system(object):

    def expectation(self):
        """
        Returns the expectation of the qwao final state to all MPI processes.
        """
        probs = np.abs(self.final_state[:self.local_i])**2
        local_expectation = np.dot(probs, self.qualities)
        return self.comm.allreduce(local_expectation, op = MPI.SUM)

    def objective(self, gammas_ts):
        """
        Objection funtion to minimise as part of the QWAO algorithm. The NLOpt module
        requires that a gradient variable be included, even if the optimizers being
        used are gradient free, which is the case for those used in QWAO_MPI.

        :param gammas_ts: Starting angles, an array of size :math:`2 \\times p`.
        :type gammas_ts: float, array

        :param grad: Placeholder gradient variable for the NLOpt library, unused.
        :type grad: None
        """
        gammas, ts = np.split(gammas_ts, 2)
        self.evolve_state(gammas, ts)
        expectation = self.expectation()
        return (self.max_quality - expectation)/np.float64(self.max_quality)

    def execute(self, gammas_ts):
        """
        Execute the QWAO algorithm.

        :param gammas_ts: Starting angles, an array of size :math:`2 \\times p`.
        :type gammas_ts: float, array
        """

        self.gammas_ts = gammas_ts

        self.result = basinhopping(
                self.objective,
                gammas_ts,
                stepsize = 0.01,
                niter = 100,
                seed = self.size,
                minimizer_kwargs = {'method':'Nelder-Mead'})

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
            self.initial_state[:] = np.array(state, np.complex128)
            if not normalized:
                nomalization = comm.allreduce(np.sum(np.multiply(np.conjugate(state), state)), op = MPI.SUM)
                self.initial_state = self.initial_state/np.sqrt(normalized)

    def set_qualities(self, func, *args, **kwargs):

        """
        Sets the qualities in the QWAO algorithm. As the array of qualities is
        equal to the size of the QWAO state, ideally the qualities should be
        generated in parallel. As such :meth:`~qwao.qualities` accepts a function
        whose first three arguments are the size of the distributed qwao state,
        the number of locally stored stored input elements and the offset of these
        elements relative to the 0-index of the distributed array. Example quality
        functions are included in :mod:`~qwao_mpi.qualities`.

        :param func: Function with which to generate the local qualities.
        :method type: callable

        :param args: Extra arguments to pass to the quality function.
        :type args: array, optional
        """
        self.qualities = func(self.size, self.local_i, self.local_i_offset, *args, **kwargs)
        self.max_quality = self.comm.allreduce(np.max(self.qualities), op = MPI.MAX)



    def save(self, file_name, config_name, action = "a"):

        """
        Write the final state, eigenvalues, qualities and execution results
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
            The final_state, eigenvalues and qualities datasets are saved using Fortran
            subroutines which make use of parallel HDF5.

            The complex values of the final_state and eigenvalues arrays are saved as a
            compound datatype consisting of contiguous double precision reals. This is
            equivalent to the np.complex128 numpy datatype. To access this data without a
            loss of precision in python the user must set the **view** of the numpy array
            to np.complex128, rather than casting it to np.complex128 using the dtype keyword.

            Similarly the qualities array, which is saved as an array of double precision
            reals, should have its view set to np.float64.
        """
        fqwao_mpi.save_dist_complex(
                file_name,
                config_name + str("/"),
                "final_state",
                action,
                self.size,
                self.local_i_offset,
                self.final_state[:self.local_i],
                self.comm.py2f())

        fqwao_mpi.save_dist_complex(
                file_name,
                config_name + str("/"),
                "eigenvalues",
                "a",
                self.size,
                self.local_i_offset,
                self.lambdas,
                self.comm.py2f())

        fqwao_mpi.save_dist_real(
                file_name,
                config_name + str("/"),
                "qualities",
                "a",
                self.size,
                self.local_i_offset,
                self.qualities,
                self.comm.py2f())

        self.comm.Barrier()

        if self.comm.Get_rank() == 0:
            File = h5py.File(file_name + ".h5", "a")
            config = File[config_name]
            minimize_result = config.create_group("minimize_result")
            for key in self.result.keys():
                try:
                    minimize_result.create_dataset(key, data = self.result.get(key))
                except:
                    none = None
                    print("No native HDF5 type for " + str(type(self.result.get(key))) + ". Minimization result field " + key + "  not saved.",
                            file = sys.stderr)
            File.create_dataset(config_name + "/initial_phases", data = self.gammas_ts, dtype = np.float64)
            File.create_dataset(config_name + "/graph_array", data = self.graph_array, dtype = np.float64)
            File.close()


class qaoa(system):

    def __init__(self, Hc, comm):

        self.size = Hc.shape[0]
        self.n_qubits = np.log(self.size)/np.log(2.0)
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.precision = "dp"

        self.partition_table = self.generate_partition_table(self.size, self.comm)

        self.local_i = self.partition_table[self.rank + 1] - self.partition_table[self.rank]
        self.local_i_offset = self.partition_table[self.rank] - 1

        self.Hc_row_starts, self.Hc_col_indexes, self.Hc_values = self.csr_local_slice(
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

        self.one_norms, self.p = fMPI.one_norm_series(
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

    def generate_partition_table(self, N, MPI_communicator):

        flock = MPI_communicator.Get_size()

        partition_table = np.zeros(flock + 1, dtype = np.int32)
        for i in range(flock + 1):
            partition_table[i] = i * N / flock + 1

        remainder = N - partition_table[flock]

        for i in range(remainder):
            partition_table[flock - i % flock : flock + 1] += 1

        return partition_table

    def csr_local_slice(self, Hc, partition_table, MPI_communicator):

        rank = MPI_communicator.Get_rank()

        lb = partition_table[rank] - 1
        ub = partition_table[rank + 1] - 1

        Hc_row_starts = Hc.indptr[lb:ub + 1]
        Hc_col_indexes = Hc.indices[Hc_row_starts[0]:Hc_row_starts[-1]] + 1
        Hc_values = Hc.data[Hc_row_starts[0]:Hc_row_starts[-1]]
        Hc_row_starts += 1

        return Hc_row_starts, Hc_col_indexes, Hc_values

    def evolve_state(self, gammas, ts):

        self.final_state = self.initial_state

        for gamma, t in zip(gammas, ts):

            self.final_state = np.multiply(np.exp(-I * gamma * self.qualities), self.final_state)

            self.final_state = fMPI.step(
                    self.size,
                    self.local_i,
                    self.Hc_row_starts,
                    self.Hc_col_indexes,
                    -I * t * self.Hc_values,
                    self.Hc_num_rec_inds,
                    self.Hc_rec_disps,
                    self.Hc_num_send_inds,
                    self.Hc_send_disps,
                    self.Hc_local_col_inds,
                    self.Hc_rhs_send_inds,
                    t,
                    self.final_state,
                    self.partition_table,
                    self.p,
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

    def graph(self, graph_array):
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

