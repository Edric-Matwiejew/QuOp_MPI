from quop_mpi.unitaries import diagonal
import numpy as np
from mpi4py import MPI
import unittest

class TestEvolutionMethods(unittest.TestCase):

    def setUp(self):

        self.complex_array = np.empty(2, dtype = np.complex128)

        self.COMM = MPI.COMM_WORLD
        self.qubits = 5
        self.system_size = 2**self.qubits

    def test_evolution(self):

        def test_operator(local_i):
            return np.ones(local_i, dtype = np.complex128)

        def test_param():
            return np.pi/5

        self.U = diagonal(
                test_operator,
                parameter_function = test_param)

        self.U.plan(self.system_size, self.COMM)


        self.initial_state = np.full(self.U.local_i, 1/np.sqrt(self.system_size), dtype = np.complex128)
        self.final_state = self.U.initial_state

        self.U.gen_operator()

        self.U.initial_state = self.initial_state
        self.U.final_state = self.final_state

        x = self.U.get_initial_params()
        self.U.propagate(x)

        self.assertIsInstance(self.U.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.U.final_state)**2)

        self.assertTrue(np.isclose((1/self.system_size) * self.U.local_i, local_probability))

    def test_evolution_array(self):

        n_operators = 5

        def test_operator(local_i):
            return [np.ones(local_i, dtype = np.complex128) for _ in range(n_operators)]

        def test_param(local_i, local_i_offset):
            return [np.pi/5 for _ in range(n_operators)]

        self.U = diagonal(
                test_operator,
                parameter_function = test_param,
                unitary_n_params = n_operators)

        self.assertTrue(self.U.unitary_n_params > 1)

        self.U.plan(self.system_size, self.COMM)

        self.initial_state = np.full(self.U.local_i, 1/np.sqrt(self.system_size), dtype = np.complex128)
        self.final_state = self.U.initial_state

        self.U.gen_operator()

        self.U.initial_state = self.initial_state
        self.U.final_state = self.final_state

        x = self.U.get_initial_params()
        self.U.propagate(x)

        self.assertIsInstance(self.U.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.U.final_state)**2)

        self.assertTrue(np.isclose((1/self.system_size) * self.U.local_i, local_probability))

if __name__ == '__main__':
    unittest.main()
