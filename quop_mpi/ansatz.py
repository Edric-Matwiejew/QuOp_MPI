from mpi4py import MPI
import quop_mpi as quop
import myqualfunc as my

comm = MPI.COMM_WORLD

n_qubits = 12
p = 4

alg = quop.MPI.ansatz(n_qubits,comm)

alg.set_unitaries(
        [op1, "sparse", 1.0, "mpi function", None],
        [op2, "circulant", True, ],
        [op3, "diagonal"],
        [op4, "sparse", True, "mpi function", 2]
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

            self.using_fftw = False
            self.partition_table_generated = False

            # parse operator type, evolution type and number of variational parameters.
            for i, unitary in enumerate(self.unitaries):
            
                [op1, "sparse", "mpi function", None],

                if unitary[1] == "sparse":

                    self.evolution_method.append(
                            _sparse._evolve_sparse(self.system_size, self.COMM_OPT))

                elif unitary[1] == "circulant":

                    self.evolution_method.append(
                            _circulant._evolve_circulant(self.system_size, sef.COMM_OPT)

                    if not self.using_fftw:
                        self.using_fftw = i

                elif unitary[1] == "diagonal":

                    self.evolution_method.append(
                            _diagonal._evolve_diagonal(self.system_size, self.COMM_OPT))

                else:
                    print(ERROR_METHOD)

                if isinstance(unitary[4], int):
                    self.parameter_groups.append(unitary[4] + 1)
                else:
                    self.parameter_groups.append(1)

                # TODO check input types
                # "constant" "function" "mpi function" "variational function", "mpi variational function"
                self.operator_types.append(unitary[3])

            # derive partitioning schemes from FFTW if "circulant" evolution is required.
            if isinstance(self.using_fftw, int):

                self.evolution_methods[self.using_fftw]._plan()

                for i, operator_type in enumerate(self.operator_types)
                    if i != self.using_fftw:
                        self.evolution_methods[i]._copy_plan(self.evolution_method[self.using_fftw])

            else:

                for method in self.evolution_methods:
                    method._generate_partition_table()


            # complete set-up of evolution methods.

            # "constant" "function" "mpi function" "variational function", "mpi variational function"
            for i, unitary in enumerate(self.unitaries):

                if unitary[1] == "sparse":

                    if unitary[3] == "constant":

                        self.evolution_methods[i]._csr_local_slice(unitary[0])
                        self.evolution_methods[i]._plan()

                    elif unitary[3] == "function":

                        self.evolution_methods[i]._call_local_csr_function(unitary[0])
                        self.evolution_methods[i]._plan()

                    elif unitary[3] == "mpi function":

                        self.evolution_methods[i]._call_mpi_csr_function(unitary[0])
                        self.evolution_methods[i]._plan()

                    elif unitary[3] == "variational function":

                        self.evolution_methods[i].variational_operator_function = unitary[0]
                        self.evolution_methods[i].variational_function_call = self.evolution_method[-1]._call_local_csr_function

                    elif unitary[3] == "mpi variational function":

                        self.evolution_methods[i].variational_operator_function = unitary[0]
                        self.evolution_methods[i].variational_function_call = self.evolution_method[-1]._call_mpi_csr_function

                    else:
                        print("ERROR")

                elif unitary[1] == "circulant":

                    if unitary[3] == "constant":

                    elif unitary[3] == "function":

                    elif unitary[3] == "variational":

                    else:
                        print("ERROR")

                elif unitary[1] == "diagonal":

                    if unitary[3] == "constant":
                        
                        if len(unitary[0]) == self.evolution_methods[i].local_i:
                            self.evolution_methods[i]._local_diag = unitary[0]

                        elif len(unitary[0]) == self.system_size:

                            local_offset_i = self.evolution_methods[i].local_offset_i
                            local_i = self.evolution_methods[i].local_i
                            self.evolution_methods[i]._local_diag = unitary[0][local_offset_i:local_offset_i + local_i]

                    elif unitary[3] == "function":

                        self.evolution_methods[i]._call_local_diag(unitary[0])

                    elif unitary[3] == "mpi function":

                        self.evolution_methods[i]._call_mpi_diag(unitary[0])

                    elif unitary[3] == "variational function":

                        self.evolution_methods[i].variational_operator_function = unitary[0]
                        self.evolution_methods[i].variational_function_call = self.evolution_methods[i]._call_local_diag

                    elif unitary[3] == "mpi variational function":

                        self.evolution_methods[i].variational_operator_function = unitary[0]
                        self.evolution_methods[i].variational_function_call = self.evolution_methods[i]._call_mpi_diag

                    else:
                        print("ERROR")


    def evolve_state(self, x):

        if self.colours[self.COMM.Get_rank()] != -1:

            self.final_state = self.initial_state

            param_groups = np.array_split(x, self.parameter_groups)
            evolve = zip(self.evolution_methods, param_groups, self.operator_types)

            for method, param_group, operator_type in evolve:

                if operator_type is "variational function":

                    n_operator_param = len(param_group) - 1

                    operator_parameters, evolution_parameter = np.array_split(
                            param_group,
                            [n_operator_param, 1])

                    operator_function = method.variational_operator_function

                    operator_call = method.variational_operator_function_call

                    operator_call(
                            operator_function,
                            variational_parameters = operator_parameters)

                    method._plan()

                else:

                    evolution_parameter = param_group

                method.initial_state = self.final_state
                method._evolve(args, evolution_parameter)
                self.final_state = method.final_state

                if operator_type is "variational function":
                    method._destroy()
