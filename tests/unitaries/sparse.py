from quop_mpi.unitaries import sparse
from quop_mpi.operators.sparse import hypercube
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

        def test_param():
            return np.pi/5

        self.U = sparse(
                hypercube,
                parameter_function = test_param)

        self.U.plan(self.system_size, self.COMM)


        self.initial_state = np.zeros(self.U.local_i, dtype = np.complex128)
        self.initial_state[0] = 1
        self.final_state = np.zeros(self.U.local_i, dtype = np.complex128)

        self.U.gen_operator()

        self.U.initial_state = self.initial_state
        self.U.final_state = self.final_state

        x = self.U.get_initial_params()
        self.U.propagate(x)

        self.assertIsInstance(self.U.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.U.final_state)**2)

        total_prob = self.COMM.allreduce(local_probability, op = MPI.SUM)

        self.assertTrue(np.isclose(1, total_prob))


    def test_evolution_array(self):

        n_operators = 5

        def test_operator(system_size, local_i, local_i_offset):

            lb = local_i_offset
            ub = local_i_offset + local_i - 1
            mixer = hypercube(system_size, lb, ub)
            row_starts, col_indexes, values = ([],[],[])
            for _ in range(n_operators):
                row_starts.append(mixer[0][0])
                col_indexes.append(mixer[1][0])
                values.append(mixer[2][0])
            return row_starts, col_indexes, values

        def test_param(local_i, local_i_offset):
            return [np.pi/(i+1) for i in range(n_operators)]

        self.U = sparse(
                test_operator,
                parameter_function = test_param,
                unitary_n_params = n_operators)

        self.assertTrue(self.U.unitary_n_params > 1)

        self.U.plan(self.system_size, self.COMM)

        self.initial_state = np.zeros(self.U.local_i, dtype = np.complex128)
        self.initial_state[0] = 1

        self.final_state = np.empty(self.U.local_i, dtype = np.complex128)

        self.U.gen_operator()

        self.U.initial_state = self.initial_state
        self.U.final_state = self.final_state

        x = self.U.get_initial_params()
        self.U.propagate(x)

        self.assertIsInstance(self.U.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.U.final_state)**2)

        total_prob = self.COMM.allreduce(local_probability, op = MPI.SUM)
        self.assertTrue(np.isclose(1, total_prob))


if __name__ == '__main__':
    unittest.main()
