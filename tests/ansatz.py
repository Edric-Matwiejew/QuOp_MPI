from quop_mpi import ansatz as anz
from quop_mpi.unitaries import diagonal, sparse
from quop_mpi.operators.sparse import  hypercube
import numpy as np
from mpi4py import MPI
import unittest

class TestEvolveState(unittest.TestCase):

    def setUp(self):

        self.complex_array = np.empty(2, dtype = np.complex128)

        self.COMM = MPI.COMM_WORLD
        self.qubits = 5
        self.system_size = 2**self.qubits

        self.n_operators = 5

    def operator(self, local_i, local_i_offset):
        return np.arange(local_i_offset, local_i + local_i_offset, dtype = np.complex128)

    def operator_array(self, local_i, local_i_offset):
            return [self.operator(local_i, local_i_offset) for _ in range(self.n_operators)]

    def param_array(self, local_i, local_i_offset):
            return [np.pi/(i + 1) for i in range(self.n_operators)]

    def test_evolve_array(self):

        U = diagonal(self.operator_array,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        self.alg = anz(self.system_size, self.COMM)
        self.alg.set_unitaries([U])
        self.alg.set_observables(self.operator)
        x = self.alg.get_initial_params()

        self.alg.evolve_state(x)

        self.assertIsInstance(self.alg.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.alg.final_state)**2)

        total_prob = self.COMM.allreduce(local_probability, op = MPI.SUM)

        self.assertTrue(np.isclose(1, total_prob))

    def test_evolve_composite(self):

        U1 = diagonal(self.operator_array,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        U2 = sparse(hypercube,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        self.alg = anz(self.system_size, self.COMM)
        self.alg.set_unitaries([U1, U2])
        self.alg.set_observables(self.operator)
        x = self.alg.get_initial_params()
        self.alg.evolve_state(x)

        self.assertIsInstance(self.alg.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.alg.final_state)**2)

        total_prob = self.COMM.allreduce(local_probability, op = MPI.SUM)

        self.assertTrue(np.isclose(1, total_prob))

    def test_evolve_composite_array(self):

        def sparse_operator_array(system_size, local_i, local_i_offset):

            lb = local_i_offset
            ub = local_i_offset + local_i - 1
            mixer = hypercube(system_size, lb, ub)
            row_starts, col_indexes, values = ([],[],[])
            for _ in range(self.n_operators):
                row_starts.append(mixer[0][0])
                col_indexes.append(mixer[1][0])
                values.append(mixer[2][0])
            return row_starts, col_indexes, values

        U1 = diagonal(self.operator_array,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        U2 = sparse(sparse_operator_array,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        self.alg = anz(self.system_size, self.COMM)
        self.alg.set_unitaries([U1, U2])
        self.alg.set_observables(self.operator)
        x = self.alg.get_initial_params()
        self.alg.evolve_state(x)

        local_probability = np.sum(np.abs(self.alg.final_state)**2)


        self.assertIsInstance(self.alg.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.alg.final_state)**2)

        total_prob = self.COMM.allreduce(local_probability, op = MPI.SUM)
        
        self.assertTrue(np.isclose(1, total_prob))

    def test_execute_composite_array(self):

        def sparse_operator_array(system_size, local_i, local_i_offset):

            lb = local_i_offset
            ub = local_i_offset + local_i - 1
            mixer = hypercube(system_size, lb, ub)
            row_starts, col_indexes, values = ([],[],[])
            for _ in range(self.n_operators):
                row_starts.append(mixer[0][0])
                col_indexes.append(mixer[1][0])
                values.append(mixer[2][0])
            return row_starts, col_indexes, values

        U1 = diagonal(self.operator_array,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        U2 = sparse(sparse_operator_array,
                    parameter_function = self.param_array,
                    unitary_n_params = self.n_operators)

        self.alg = anz(self.system_size, self.COMM)
        self.alg.set_unitaries([U1, U2])
        self.alg.set_observables(self.operator)
        self.alg.execute()
        self.alg.print_optimiser_result()

        local_probability = np.sum(np.abs(self.alg.final_state)**2)


        self.assertIsInstance(self.alg.final_state, type(self.complex_array))

        local_probability = np.sum(np.abs(self.alg.final_state)**2)

        total_prob = self.COMM.allreduce(local_probability, op = MPI.SUM)
        
        self.assertTrue(np.isclose(1, total_prob))


if __name__ == '__main__':
    unittest.main()



