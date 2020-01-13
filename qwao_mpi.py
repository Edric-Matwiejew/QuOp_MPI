from mpi4py import MPI
import fqwao_mpi
import numpy as np
from scipy.optimize import minimize

"""
To do:
    Write functions to save state data using parallel hdf5.
    Test on magnus.
    Make non-editable variables private and write functions to retrive their values.
    Produce Fortran makefile.
    Create docs using sphinx
    write README
"""

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
        self.size = n_qubits**2
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
        return minimize(self.objective, betas_gammas, *args)

class qualities:
    """
    Functions for parallel distributed memory generation of the QWAO quality array.
    """
    def integer(N, local_i, local_i_offset):
        """
        Produces the array [1, ..., N - 1] distributed of the active MPI communicator.

        :param N: Size of the distrubted system.
        :type N: integer

        :param local_i: Number of local input QWAO state values, given by qwao.local_i.
        :type local_i: integer

        :param local_i_offset: Offset of the local QWAO state values relative to the \ 
        zero index of the distributed array. Given by qwao.local_i_offset.
        :type local_i_offset: integer.
        """
        return np.asarray(range(local_i_offset, local_i_offset + local_i), dtype = np.float64)

class graph_array:
    """
    Fuctions to generate arrays corresponding to the first row of a circulat graph \
    adjacency matrix..
    """

    def complete(N):
        """
        Returns an array corresponding to a complete graph of size N.

        :param N: Number of graph nodes.
        :type N: integer
        """
        graph_array = np.ones(N, dtype = np.float64)
        graph_array[0] = 0
        return graph_array

if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    p = 10
    n_qubits = 4

    # random beta and gamma start anglels
    np.random.seed(1)
    x0 = np.random.rand(2*p)

    qwao = qwao(n_qubits, comm)
    qwao.graph(graph_array.complete(qwao.size))
    qwao.qualities(qualities.integer)
    qwao.plan()

    result = qwao.execute(x0)

    qwao.destroy_plan()

    if comm.Get_rank() == 0:
        print(result)

