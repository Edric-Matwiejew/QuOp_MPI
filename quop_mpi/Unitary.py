from importlib import import_module
import numpy as np
from mpi4py import MPI
from .__utils.__interface import interface


class Unitary:
    """Used to define a phase-shift or mixing unitary that can be passed to the
    :math:`Ansatz` class :meth:`~Ansatz.set_unitaries` method.

    :param operator_function: Function returning the local partition of the unitary's matrix operator exponent, :math:`\hat{O}` or :math:`\hat{W}`.
    :type operator_function: callable

    :param operator_n_params: Number of variational parameters :math:`|\\theta|` associated with :math:`\hat{O}` or :math:`\hat{W}`.
    :type operator_n_params: optional, integer, default = 0

    :param operator_kwargs: Keyword arguments associated with `operator_function`.
    :type operator_kwargs: optional, dictionary, default = None

    :param parameter_function: Function responsible for generation of the :math:`\\theta` associated with the unitary and its matrix operator exponent.
    :type parameter_function: callable

    :param parameter_kwargs: Keyword arguments associated with `parameter_function`.
    :type parameter_kwargs: optional, dictionary, default = None

    :param unitary_n_params: Number of variational parameters :math:`|\\theta|` associated with the unitary time-evolution operator.
    :type unitary_n_params: optional, integer, default = 0
    """

    def __init__(
        self,
        operator_function,
        operator_n_params=0,
        operator_kwargs=None,
        parameter_function=None,
        parameter_kwargs=None,
        unitary_n_params=1,
    ):

        if operator_kwargs is None:
            operator_kwargs = {}
        if parameter_kwargs is None:
            parameter_kwargs = {}

        self.operator_function = operator_function
        self.operator_n_params = operator_n_params
        self.operator_kwargs = operator_kwargs
        self.parameter_function = parameter_function
        self.parameter_kwargs = parameter_kwargs
        self.unitary_n_params = unitary_n_params

        self.unitary_type = None
        self.planner = False

        #self.operator_parameters = [
        #    "operator_function",
        #    "system_size",
        #    "alloc_local",
        #    "local_i",
        #    "local_i_offset",
        #    "local_o",
        #    "local_o_offset",
        #    "partition_table",
        #    "lb",
        #    "ub",
        #    "variational_parameters",
        #    "seed",
        #    "MPI_COMM",
        #]

        #self.parameter_function_parameters = [
        #    "system_size",
        #    "operator",
        #    "n_params",
        #    "seed",
        #    "MPI_COMM",
        #]

        self.system_size = None
        self.operator = None
        self.n_params = 0
        self.seed = 0
        self.initial_parameters = None
        self.initial_state = None
        self.final_state = None
        self.alloc_local = None
        self.local_i = None
        self.local_i_offset = None
        self.partition_table = None
        self.lb = None
        self.ub = None
        self.variational_parameters = None
        self.planned = False

        self.n_params += operator_n_params + unitary_n_params

    def parse_operator_function(self):

        self.parsed_operator_function = interface(
            [self],
            self.operator_function,
            "operator",
            self.MPI_COMM,
        )

    def parse_parameter_function(self):

        if self.parameter_function is None:
            from quop_mpi.param.rand import uniform

            self.parameter_function = uniform

        self.parsed_parameter_function = interface(
            [self],
            self.parameter_function,
            "initial parameters",
            self.MPI_COMM,
        )

    def gen_initial_params(self):
        self.parsed_parameter_function.update_parameters()
        return self.parsed_parameter_function.call(**self.parameter_kwargs)

    def gen_operator(self):
        if self.variational_parameters is not None:
            self.parsed_operator_function.update_parameters()
        self.operator = self.parsed_operator_function.call(**self.operator_kwargs)

    def parse_plan(self, plan):

        self.local_i = plan[0]
        self.alloc_local = plan[1]

        self.rank = self.MPI_COMM.Get_rank()
        self.size = self.MPI_COMM.Get_size()

        self.partition_table = np.zeros(self.size + 1, dtype=np.int32)
        self.partition_table[self.rank + 1] = self.local_i

        self.partition_table = self.MPI_COMM.allreduce(
            self.partition_table,
            op = MPI.SUM
        )

        self.partition_table = np.cumsum(self.partition_table)
        self.partition_table += 1

        self.lb = self.partition_table[self.rank] - 1
        self.ub = self.partition_table[self.rank + 1] - 1

        self.local_i_offset = self.lb

        self.parse_operator_function()
        self.parse_parameter_function()

    def __plan(self, system_size, MPI_COMM):

        plan = self.plan(system_size, MPI_COMM)

        self.system_size = system_size
        self.MPI_COMM = MPI_COMM

        self.parse_plan(plan)

        self.final_state = np.empty(self.alloc_local, dtype=np.complex128)
        self.initial_state = np.empty(self.alloc_local, dtype=np.complex128)

    def __copy_plan(self, ex_unitary):

        self.copy_plan(ex_unitary)

        self.system_size = ex_unitary.system_size
        self.MPI_COMM = ex_unitary.MPI_COMM

        plan = [ex_unitary.local_i, ex_unitary.alloc_local]

        self.parse_plan(plan)

        self.final_state = ex_unitary.final_state
        self.initial_state = ex_unitary.initial_state

    def propagate(self, x):
        """A method implementing computation of the action of the unitary on
        the quantum state vector :math:`| \\boldsymbol{\\theta} \\rangle`. Must
        be consistent with the QuOp_MPI :class:`Ansatz` parallelisation scheme.

        :param x: An array of :math:`|\\theta|` variational parameters.
        :type x: array, float
        """
        raise NotImplementedError("Method 'propagate' not implemented by child class")

    def plan(self, system_size, MPI_COMM):
        """A method that plans the partitioning scheme used by :class:`Ansatz`
        and performs any other tasks required by :meth:`~Unitary.propagate`.

        :param system_size: Size of the quantum system :math:`N`.
        :type system_size: integer

        :param MPI_COMM: MPI communicator over which the :class:`Unitary` and :class:`Ansatz` operators and quantum state are partitioned.
        :type MPI_COMM: MPI4py communicator object.

        :return: The array ['local_i', 'alloc_local'] where 'local_i' is the number of elements in a row-wise partitioning of :math:`\\hat{O}`, :math:`\\hat{W}`, :math:`|\psi_0\\rangle_\\text{ANZ}` and :math:`| \\boldsymbol{\\theta} \\rangle`, and `alloc_local` is the size of the allocated array containing the quantum state vectors.
        :rtype: array, integer
        """
        raise NotImplementedError("Method 'plan' not implemented by child class")

    def copy_plan(self, ex_unitary):
        """Method to perform any setup required by :meth:`~Unitary.propagate`
        based off the partitioning plan of another :class:`Unitary`."""
        raise NotImplementedError("Method 'copy_plan' not implemented by child class")

    def destroy(self):
        """Method to free any memory allocated by :meth:`~Unitary.plan` or
        :meth:`~Unitary.copy_plan` that is not managed by the python garbage
        collector."""
        pass
