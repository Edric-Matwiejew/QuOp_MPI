from mpi4py import MPI
import h5py
import json
import numpy as np
from scipy.optimize import minimize
import fqwao_mpi

class qwao:
    """
    Handles the creation of a :class:`qwao` system distributed over an MPI communicator, \
    the evolution of this system and execution of the QWAO algorithm in parallel. \
    Evolution of the :class:`qwao` state occurs via calls to the compiled Fortran library \
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

        """
        When performing a parallel 1D-FFT using FFTW it may be the case that \
        the transformed array is distributed on the MPI communicator differently \
        from the input. fqwao.mpi_local_size determines the size needed at each \
        MPI node to accomodate for this. Along with the number of actual array
        elements stored at each node and their offset relative to the 0-index \
        of the distributed array.
        """

        local_sizes = fqwao_mpi.mpi_local_size(self.size, comm.py2f())

        self.alloc_local = local_sizes[0]
        self.local_i = local_sizes[1]
        self.local_i_offset = local_sizes[2]
        self.local_o = local_sizes[3]
        self.local_o_offset = local_sizes[4]

        self.initial_state = np.ones(self.alloc_local, np.complex128)/np.sqrt(np.float64(self.size))
        self.final_state = np.empty(self.alloc_local, np.complex128)

        self.dummy_betas = np.empty(1, dtype = np.float64)
        self.dummy_gammas = np.empty(1, dtype = np.float64)
        self.dummy_qualities = np.empty(1, dtype = np.float64)
        self.dummy_lambdas = np.empty(1, dtype = np.float64)


    def qualities(self, method, *args):
        """
        Sets the qualities in the QWAO algorithm. As the array of qualities is \
        equal to the size of the QWAO state, ideally the qualities should be \
        generated in parallel. As such :method:`qwao.qualities` accepts a function \
        whose first three arguments are the size of the distributed qwao state, \
        the number of locally stored stored input elements and the offset of these \
        elements relative to the 0-index of the distributed array.

        :param method: Function with which to generate the local qualities.
        :method type: callable

        :param args: Extra arguments to pass to the quality function.
        :type args: array, optional
        """
        self.qualities = method(self.size, self.local_i, self.local_i_offset, *args)

    def graph(self, graph_array):
        """
        Given a 1D array representing the first row of a circulant matrix, \
        this returns a 1D array of matrix eigenvalues corresponding to a \
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
        Calls FFTW subroutines which set up the ancillary data structures needed to \
        efficiently perform 1D parallel Fourier and inverse Fourier transforms.
        """
        fqwao_mpi.qwao_state(
                self.size,
                self.dummy_betas,
                self.dummy_gammas,
                self.dummy_qualities,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.comm.py2f(),
                1)


    def evolve_state(self, betas, gammas):
        """
        Evolves the qwao initial state to the final state.

        :param betas:
        :type betas: float, array

        :param gamma:
        :type gamma: float, array
        """
        fqwao_mpi.qwao_state(
                self.size,
                betas,
                gammas,
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
                self.dummy_betas,
                self.dummy_gammas,
                self.dummy_qualities,
                self.dummy_lambdas,
                self.initial_state,
                self.final_state,
                self.comm.py2f(),
                -1)

    def expectation(self):
        """
        Returns the expectation of the qwao final state to all MPI processes.
        """
        probs = np.abs(self.final_state[:self.local_i])**2
        local_expectation = np.dot(probs, self.qualities)
        return self.comm.allreduce(local_expectation, op = MPI.SUM)

    def objective(self, betas_gammas):
        """
        Objection funtion to minimise as part of the QWAO algorithm.

        :param betas_gammas: Starting angles, an array of size :math:`2 \\times p`.
        :type betas_gammas: float, array
        """
        betas, gammas = np.split(betas_gammas, 2)
        self.evolve_state(betas, gammas)
        return self.expectation()

    def execute(self, betas_gammas, *args):
        """
        Execute the QWAO algorithm.

        :param betas_gammas: Starting angles, an array of size :math:`2 \\times p`.
        :type betas_gammas: float, array

        :param args: Extra arguments to pass to the SciPy minimise function.
        :type args: tuple, optional
        """
        self.betas_gammas = betas_gammas
        self.result = minimize(self.objective, betas_gammas, *args)
        return self.result

    def save(self, file_name, config_name, action = "a"):

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

        if self.comm.Get_rank() == 0:
            File = h5py.File(file_name + ".h5", "a")
            config = File[config_name]
            minimize_result = config.create_group("minimize_result")
            for key in self.result.keys():
                minimize_result.create_dataset(key, data = self.result.get(key))
            File.create_dataset(config_name + "/initial_phases", data = self.betas_gammas)
            File.create_dataset(config_name + "/graph_array", data = self.graph_array)
            File.close()
