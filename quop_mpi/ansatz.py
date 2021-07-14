from mpi4py import MPI
import quop_mpi as quop
import myqualfunc as my

comm = MPI.COMM_WORLD

n_qubits = 12
p = 4

alg = quop.MPI.ansatz(n_qubits,comm)

alg.set_unitaries(
        [op1, "sparse"],
        [op2, "circulant"],
        [op3, "diagonal"],
        [op4, "sparse", "function", 2]
        [op5, "diagonal", "function", 3]
        [op6, "circulant", "function", 4])

alg.set_observables(myobsfunc)

alg.execute(p)
alg.print_result()

class ansatz():

    def __init__(self, MPI_communicator, n_qubits):
        self.COMM = MPI_communicator
        self.n_qubits = n_qubits

    def set_unitaries(self, *args):
        self.unitaries = []
        for arg in args:
            self.unitaries.append(arg)

    def _parse_unitaries(self):

        if self.colours[self.COMM_OPT] != -1:

            self.evolution_methods = []
            self.operator_types = []
            self.parameter_groups = []
            self.evolution_args = []

            for unitary in self.unitaries:

                match unitary[1]:
                    case "sparse":
                        self.evolution_method.append(self.evolve_sparse)
                    case "circulant":
                        self.evolution_method.append(self.evolve_circulant)
                    case "diagonal":
                        self.evolution_method.append(self.evolve_diagonal)
                    case _:
                        print(ERROR_METHOD)

                match len(unitary):
                    case 2:
                        self.operator_types.append("constant")
                        self.parameter_groups.append(1)
                    case 3
                        self.operator_types.append("function")
                        self.parameter_groups.append(1)
                    case 4:
                        self.operator_types.append("variational function")
                        self.parameter_groups.append(unitary[3])
                    case _:
                        print(ERROR_LENGTH)

                operator_type = "{} {}".format(unitary[1], self.operator_types[-1]):

                evolution_args.append([])
                match
                    case "sparse constant":
                        DO SPARSE THINGS
                        evolution_args[-1].append([])
                    case "sparse function":
                        CALL SPARSE THINGS
                        evolution_args[-1].append([])
                    case "sparse variational function":
                        SET UP FOR RUNTIME CALLS
                        evolution_args[-1].append([])
                    case "circulant constant":
                        DO CIRCULANT THINGS
                        evolution_args[-1].append([])
                    case "circulant function":
                        DO GRAPH ARRAY THINGS
                        evolution_args[-1].append([])
                    case "circulant variational function":
                        SET UP FOR RUNTIME CALLS
                        evolution_args[-1].append([])
                    case "diagonal constant":
                        DO DIAGONAL THINGS
                        evolution_args[-1].append([])
                    case "diagonal function":
                        GEN_DIAGONAL_THINGS
                        evolution_args[-1].append([])
                    case "diagonal variational function":
                        SET UP FOR RUNTIME CALLS
                        evolution_args[-1].append([])
                    case _:
                        print(ERROR_OPERATOR_TYPE)

    def evolve_state(self, x):

        if self.colours[self.COMM.Get_rank()] != -1:

            param_groups = np.array_split(x, self.parameter_groups)
            evolve = zip(self.evolution_methods, self.evolution_args, param_groups)

            for method, evol_args, param_group in evolve:

                if evol_args[0] is "variational function":
                    # *_setup(variational_function(args))
                    args = evol_args[3](evol_args[1](evol_args[2]))
                else:
                    args = evol_args

                self.final_state = method(args, param_group)

    def _sparse_setup(self, csr_partition):
        pass
    def _circulant_setup(self, graph_array):
        pass
    def _diagonal_setup(self, diag_elements):

    def _evolve_sparse(self, evol_args, param_group):
        pass
    def _evolve_circulant(self, evol_args, param_group):
        # plan() and destroy()
        pass
    def _evolve_diagonal(self, evol_args, param_group):
        pass


