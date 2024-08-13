from __future__ import annotations
from importlib import import_module
import numpy as np
from mpi4py import MPI
from .__utils.__interface import interface

####################################
# imports and classes for type hints
####################################

from typing import Callable, Union, Iterable, Any

Intracomm = MPI.Intracomm
iterable = Iterable

class Unitary:
    """Base class for a ``unitary``.

    A ``unitary`` is derived from the ``Unitary`` class and implements
    simulation of a specfic :term:`unitary` through definition of the following
    methods:

    * :meth:`~Unitary.propagate`
    * :meth:`~Unitary.plan`
    * :meth:`~Unitary.copy_plan`
    * :meth:`~Unitary.destroy`

    A list of ``unitary`` instances passed to
    :meth:`quop_mpi.Ansatz.set_unitaries` defines the :term:`ansatz unitary` of
    a :term:`QVA`. After initialisation, ``unitary`` instances are managed by the
    :class:`quop_mpi.Ansatz` class and calls to ``unitary`` methods are not made
    explicitly.

    See :mod:`quop_mpi.propagator` for predefined ``unitary`` subclasses.

    Associated QuOp Functions:

    * :term:`Operator Function`
    * :term:`Parameter Function`

    The following attributes are common to all ``unitary`` instances.

    Attributes
    ----------

    final_state
        The :term:`system state` after the action of the unitary.
    initial_parameters
        Initial :term:`variational parameters` returned from the user-defined
        :term:`Parameter Function`.
    initial_state
        The :term:`initial state` of the quantum system.
    n_params
        The total number of :term:`unitary <unitary parameter>` and
        :term:`operator <operator parameter>` parameterising the
        :term:`unitary`.
    operator_function
        The user-defined :term:`Operator Function`.
    operator_dict
        A :term:`FunctionDict` of additional position and keyword arguments for
        the :term:`Operator Function`.
    operator
        The :term:`operator` object returned by the :term:`Operator Function`.
    operator_n_params
        Number of variational :term:`operator parameters <operator parameter>`.
    parameter_function
        The user-defined :term:`Parameter Function`.
    param_dict
        A :term:`FunctionDict` of additional position and keyword arguments for
        the :term:`Parameter Function`.
    planner
        If ``True``, the parallel partitioning scheme returned by
        :meth:`~Unitary.plan` takes precedence over non-planner ``unitaries``
        and ``unitaries`` that appear later in the :term:`ansatz unitary` list
        supplied to :meth:`quop_mpi.Ansatz.set_unitaries`.
    seed
        Integer for seeding random number generation, shared with
        :class:`quop_mpi.Ansatz`.
    system_size
        The size of the :term:`simulated quantum system<QVA>`, shared with
        :class:`quop_mpi.Ansatz`.
    unitary_n_params
        The number of :term:`unitary parameters <unitary parameter>`.
    unitary_type
        A string labeling the ``unitary`` type (e.g. "diagonal" or "sparse").
    variational_parameters
        :term:`Operator variational parameters <operator parameter>`. If present
        as an argument of the :term:`Operator Function`, a real array of size
        ``operator_n_params`` is passed to the :term:`Operator Function`.
    MPI_COMM
        MPI Intracommunicator, shared with :class:`quop_mpi.Ansatz`.
    alloc_local
        The size of the array storing the :term`operator` if the operator is an
        array (equal to ``local_i`` otherwise). The second return value of
        :meth:`~Unitary.plan`.
    lb
        The lower global index of the local :term:`system state` partition.
    ub
        The upper global index of the local :term:`system state` partition.
    local_i
        The size of the local :term:`system state` partition. The first return
        value of :meth:`~Unitary.plan`
    local_i_offset
        The global index offset of the local :term:`system state` partition.
    partition_table
        1-D integer array describing the global partitioning scheme such that
        for a given MPI rank ``partition_table[rank + 1] - partition_table[rank]
        = local_i``
    """

    def __init__(
        self,
        operator_function: Callable,
        operator_n_params: int = 0,
        operator_dict: dict = None,
        parameter_function: Callable = None,
        param_dict: dict = None,
        unitary_n_params: int = 1,
    ):
        """

        Parameters
        ----------
        operator_function : callable
            :term:`Operator Function`
        operator_n_params : int, optional
            number of :term:`operator parameters <operator parameter>` associated with ``operator_function``, assumed to be 0 by default
        operator_dict : dict, optional
            :term:`FunctionDict` for ``operator_function``
        parameter_function : callable, optional
            :term:`Parameters Function`
        param_dict : dict, optional
            FunctionDict for ``parameter_function``
        unitary_n_params : int, optional
            number of :term:`unitary parameters <unitary parameter>`, assumed to be 1 by default
        """

        self.operator_function = operator_function
        self.operator_n_params = operator_n_params
        self.operator_dict = operator_dict

        self.parameter_function = parameter_function
        self.param_dict = param_dict
        self.unitary_n_params = unitary_n_params

        self.unitary_type = None
        self.planner = False

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
        self.variational_parameters = []
        self.planned = False # modified by the Ansatz class

        self.n_params += operator_n_params + unitary_n_params

        #TODO document
        self.comm_size_constraints = [np.array([1], dtype = int)]

    def __parse_function_dict__(self, function_dict, attribute_name):

        parsed_dict = getattr(self, attribute_name)

        function_dict = {} if function_dict is None else function_dict
        parsed_dict = {"args": [], "kwargs": {}}

        for key in function_dict:
            if function_dict[key] is not None:
                parsed_dict[key] = function_dict[key]

        setattr(self, attribute_name, parsed_dict)

    def parse_operator_function(self):

        self.parsed_operator_function = interface(
            [self],
            self.operator_function,
            "operator",
            self.MPI_COMM,
        )

    def parse_parameter_function(self):

        self.__parse_function_dict__(self.param_dict, "param_dict")

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

        return self.parsed_parameter_function.call(
            *self.param_dict["args"], **self.param_dict["kwargs"]
        )

    def gen_operator(self) -> Any:

        self.__parse_function_dict__(self.operator_dict, "operator_dict")

        if len(self.variational_parameters) > 0:
            self.parsed_operator_function.update_parameters()

        self.operator = self.parsed_operator_function.call(
            *self.operator_dict["args"], **self.operator_dict["kwargs"]
        )

    def parse_plan(self, plan: list[int]):

        self.local_i = plan[0]
        self.alloc_local = plan[1]

        self.rank = self.MPI_COMM.Get_rank()
        self.size = self.MPI_COMM.Get_size()

        self.partition_table = np.zeros(self.size + 1, dtype=np.int32)
        self.partition_table[self.rank + 1] = self.local_i

        self.partition_table = self.MPI_COMM.allreduce(self.partition_table, op=MPI.SUM)

        self.partition_table = np.cumsum(self.partition_table)
        self.partition_table += 1

        self.lb = self.partition_table[self.rank] - 1
        self.ub = self.partition_table[self.rank + 1] - 1

        self.local_i_offset = self.lb

        self.parse_operator_function()
        self.parse_parameter_function()

    def __plan(self, system_size: int, MPI_COMM: Intracomm):

        plan = self.plan(system_size, MPI_COMM)

        self.system_size = system_size
        self.MPI_COMM = MPI_COMM

        self.parse_plan(plan)

        self.final_state = np.empty(self.alloc_local, dtype=np.complex128)
        self.initial_state = np.empty(self.alloc_local, dtype=np.complex128)

    def __copy_plan(self, ex_unitary: 'Unitary'):

        self.copy_plan(ex_unitary)

        self.system_size = ex_unitary.system_size
        self.MPI_COMM = ex_unitary.MPI_COMM

        plan = [ex_unitary.local_i, ex_unitary.alloc_local]

        self.parse_plan(plan)

        self.final_state = ex_unitary.final_state
        self.initial_state = ex_unitary.initial_state

    def propagate(self, x: np.ndarray[np.float64]):
        """Simulation of the action of a :term`unitary`.

        When implemented, ``propagate`` contains a call to a method (typically a
        contained in a complied Python extension module) that takes the class
        attributes ``initial_state``, ``final_state`` and ``MPI_COMM``, together
        with attributes describing the parallel partitioning scheme and
        :term:`variational parameters` ``x``, as input. The action of the unitary
        is computed in MPI parallel, with the computed result written to
        ``final_state``.

        .. warning::

            Not implemented by the base ``Unitary`` class.

        Examples
        --------

        .. code-block:: python

            def propagate(self, x):

                external_propagator(
                    x, self.partition_table, self.initial_state,
                    self.final_state, self.MPI_COMM )

        Parameters
        ----------
        x : ndarray[float64]
            a 1-D real array of ``n_params`` :term:`variational parameters`
        """
        raise NotImplementedError("Method 'propagate' not implemented by child class")

    def plan(self, system_size: int, MPI_COMM: Intracomm):
        """Plan the partitioning scheme used by an :class:`quop_mpi.Ansatz`
        instance and performs any other tasks required by
        :meth:`~Unitary.propagate`.

        An implemented ``plan`` returns (``local_i``, ``alloc_local``). Data
        structures and allocation required by the propagation method called in
        :meth:`~Unitary.propagate` are assigned to attributes of the ``Unitary``
        instance.

        The ``alloc_local`` return value specifies the size of the local
        :term:`system state` arrays ``initial_state`` and ``final_state``. In
        most cases ``alloc_local == local_i``, however ``alloc_local > local_i``
        may be required by particular external propagation methods (e.g. the
        parallel FFTW).

        .. warning::

            Not implemented by the base ``Unitary`` class.

        Examples
        --------

        .. code-block:: python

            def plan(self, system_size, MPI_COMM):

                local_i = system_size  // MPI_COMM.size

                local_i = (
                    system_size - local_i * MPI_COMM.rank if MPI_COMM.rank == 0
                    else local_i
                )

                alloc_local = local_i

                return local_i, alloc_local

        Parameters
        ----------
        system_size : int
            size of the simulated quantum :term:`system<system state>`.
        MPI_COMM : Intracomm
            MPI communicator over which the :class:`Ansatz` :term:`observables`,
            :term:`initial state` and :term:`final state` are partitioned.

        Returns
        -------
        (int, int)
            number of elements in a row-wise partitioning of the system state
            and size to allocate for the ``initial_state`` and ``final_state``
            arrays
        """
        raise NotImplementedError("Method 'plan' not implemented by child class")

    def copy_plan(self, ex_unitary : Unitary):
        """Perform any setup required by the propagation method called in
        :meth:`~Unitary.propagate`.

        When implemented, ``copy_plan`` performs the same internal operations as
        :meth:`~Unitary.plan` using the ``local_i`` and ``alloc_local``
        attributes of ``ex_unitary``. Does **not** return ``[local_i,
        alloc_local]``.
 
        .. warning::

            Not implemented by the base ``Unitary`` class.       

        Parameters
        ----------
        ex_unitary : unitary 
            a ``unitary`` instance with computed ``local_i`` and ``alloc_local``
            attributes
        """
        raise NotImplementedError("Method 'copy_plan' not implemented by child class")

    def destroy(self):
        """Free memory allocated by Python extension modules in
        :meth:`~Unitary.plan` or :meth:`~Unitary.copy_plan`. collector.

        Memory allocated by compiled Python extension modules is typically not
        managed by the Python garbage collector. These allocations must be freed
        via relevant methods in the extension module to prevent the occurrence
        of `memory leaks <https://en.wikipedia.org/wiki/Memory_leak>`_.

        .. warning::

            Not implemented by the base ``Unitary`` class.
        """
        pass
