"""Base unitary class for quantum variational algorithm propagators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._utils._bindable import Bindable
from ._utils._interface import Interface

if TYPE_CHECKING:
    from typing import Any, Callable

    from mpi4py import MPI

    from ._utils._comm_size import QuopMpiLayout

    Intracomm = MPI.Intracomm


class UnitaryBase(Bindable):
    """Base class for a ``unitary``.

    A ``unitary`` is derived from the ``UnitaryBase`` class and implements
    simulation of a specfic :term:`unitary` through definition of the following
    methods:

    * :meth:`~Unitary.propagate`
    * internal planning step ``_UnitaryBase__plan``
    * :meth:`~Unitary.destroy`

    A list of ``unitary`` instances passed to
    :meth:`quop_mpi.ansatz.set_unitaries` defines the :term:`ansatz unitary` of
    a :term:`QVA`. After initialisation, ``unitary`` instances are managed by the
    :class:`quop_mpi.ansatz` class and calls to ``unitary`` methods are not made
    explicitly.

    See :mod:`quop_mpi.propagator` for predefined ``unitary`` subclasses.

    Associated QuOp Functions:

    * :term:`Operator Function`
    * :term:`Parameter Function`

    The following attributes are common to all ``unitary`` instances.

    Attributes
    ----------

    final_state
        Legacy Python output buffer for custom ``unitary`` subclasses.
        Built-in propagators evolve through backend-owned native context
        buffers instead.
    initial_parameters
        Initial :term:`variational parameters` returned from the user-defined
        :term:`Parameter Function`.
    initial_state
        Legacy Python input buffer for custom ``unitary`` subclasses.
        Built-in propagators evolve through backend-owned native context
        buffers instead.
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
        If ``True``, the parallel partitioning scheme from the internal
        planning step ``_UnitaryBase__plan`` takes precedence over non-planner ``unitaries``
        and ``unitaries`` that appear later in the :term:`ansatz unitary` list
        supplied to :meth:`quop_mpi.ansatz.set_unitaries`.
    seed
        Integer for seeding random number generation, shared with
        :class:`quop_mpi.ansatz`.
    system_size
        The size of the :term:`simulated quantum system<QVA>`, shared with
        :class:`quop_mpi.ansatz`.
    unitary_n_params
        The number of :term:`unitary parameters <unitary parameter>`.
    unitary_type
        A string labeling the ``unitary`` type (e.g. "diagonal" or "sparse").
    variational_parameters
        :term:`Operator variational parameters <operator parameter>`. If present
        as an argument of the :term:`Operator Function`, a real array of size
        ``operator_n_params`` is passed to the :term:`Operator Function`.
    MPI_COMM
        MPI Intracommunicator, shared with :class:`quop_mpi.ansatz`.
    alloc_local
        The size of the array storing the :term:`operator` if the operator is an
        array (equal to ``local_i`` otherwise).
    lb
        The lower global index of the local :term:`system state` partition.
    ub
        The upper global index of the local :term:`system state` partition.
    local_i
        The size of the local :term:`system state` partition.
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
    ) -> None:
        """

        Parameters
        ----------
        operator_function : callable
            :term:`Operator Function`
        operator_n_params : int, optional
            number of :term:`operator parameters <operator parameter>`
            associated with ``operator_function``, assumed to be 0
            by default
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
        self._layout = None  # QuopMpiLayout (set by __plan)
        self._mpi_comm = None  # fallback when _layout is absent
        self._partition_table = None  # fallback when _layout is absent
        self.lb = None
        self.ub = None
        self.variational_parameters = []
        self.planned = False  # modified by the Ansatz class

        self.n_params += operator_n_params + unitary_n_params

        #: Constraints on valid MPI communicator sizes for this unitary.
        #: A list of 1-D integer arrays specifying divisibility requirements.
        #: Used by :meth:`quop_mpi.ansatz` to determine compatible parallelization.
        self.comm_size_constraints = [np.array([1], dtype=int)]

    # Bindable attributes for QuOp Functions bound to Unitary instances.
    # Subclasses (propagators) can extend this by defining their own BINDABLE_ATTRIBUTES dict.
    BINDABLE_ATTRIBUTES = {
        # Core partitioning (shared with Ansatz)
        "system_size": "Total number of quantum basis states",
        "local_i": "Number of elements in this rank's partition",
        "local_i_offset": "Global index offset for this rank's partition",
        "partition_table": "Array describing global partitioning scheme",
        # MPI (shared with Ansatz)
        "MPI_COMM": "MPI subcommunicator",
        "seed": "Random seed for parameter generation",
        # Unitary-specific partitioning
        "alloc_local": "Size of operator array (equals local_i for non-array operators)",
        "lb": "Lower global index of the local partition",
        "ub": "Upper global index of the local partition",
        # Parameter counts
        "n_params": "Total parameters for this Unitary (operator + unitary)",
        "operator_n_params": "Number of operator variational parameters",
        "unitary_n_params": "Number of unitary variational parameters",
        # State and operator
        "variational_parameters": "Operator variational parameters (for parameterised operators)",
        "initial_state": "Legacy explicit Python input buffer for custom Unitary subclasses",
        "final_state": "Legacy explicit Python output buffer for custom Unitary subclasses",
        "operator": "The operator object (after gen_operator called)",
    }

    # -- Properties backed by QuopMpiLayout ---------------------------

    @property
    def partition_table(self) -> np.ndarray | None:
        """Partition table from the layout, or the local fallback."""
        if self._layout is not None:
            pt = self._layout.partition_table
            if pt is not None:
                return pt
        return self._partition_table

    @partition_table.setter
    def partition_table(self, val: np.ndarray | None) -> None:
        self._partition_table = val

    @property
    def MPI_COMM(self) -> Intracomm | None:  # noqa: N802
        """MPI subcommunicator from the layout, or the local fallback."""
        if self._layout is not None:
            sc = self._layout.SUBCOMM
            if sc is not None:
                return sc
        return self._mpi_comm

    @MPI_COMM.setter
    def MPI_COMM(self, val: Intracomm | None) -> None:  # noqa: N802
        self._mpi_comm = val

    @property
    def local_i_offset(self) -> int | None:
        """Global index offset -- always equal to ``lb``."""
        return self.lb

    def __parse_function_dict__(self, function_dict: dict | None, attribute_name: str) -> None:

        parsed_dict = getattr(self, attribute_name)

        function_dict = {} if function_dict is None else function_dict
        parsed_dict = {"args": [], "kwargs": {}}

        for key in function_dict:
            if function_dict[key] is not None:
                parsed_dict[key] = function_dict[key]

        setattr(self, attribute_name, parsed_dict)

    def parse_operator_function(self) -> None:
        """Wrap the operator function in an :class:`Interface` for calling."""
        self.__parse_function_dict__(self.operator_dict, "operator_dict")

        self.parsed_operator_function = Interface(
            [self],
            self.operator_function,
            "operator",
            self.MPI_COMM,
            call_args=self.operator_dict["args"],
            call_kwargs=self.operator_dict["kwargs"],
        )

    def parse_parameter_function(self) -> None:
        """Wrap the parameter function in an :class:`Interface` for calling."""
        self.__parse_function_dict__(self.param_dict, "param_dict")

        if self.parameter_function is None:
            from quop_mpi.param.rand import uniform

            self.parameter_function = uniform

        self.parsed_parameter_function = Interface(
            [self],
            self.parameter_function,
            "initial parameters",
            self.MPI_COMM,
            call_args=self.param_dict["args"],
            call_kwargs=self.param_dict["kwargs"],
        )

    def gen_initial_params(self) -> np.ndarray:
        """Generate initial variational parameters from the parameter function."""
        self.parsed_parameter_function.update_parameters()

        return self.parsed_parameter_function.call(
            *self.param_dict["args"], **self.param_dict["kwargs"]
        )

    def gen_operator(self) -> Any:  # noqa: ANN401
        """Generate the unitary operator from the operator function."""
        self.__parse_function_dict__(self.operator_dict, "operator_dict")

        self.parsed_operator_function.update_parameters()

        self.operator = self.parsed_operator_function.call(
            *self.operator_dict["args"], **self.operator_dict["kwargs"]
        )

    def __plan(self, system_size: int, layout: QuopMpiLayout) -> None:
        """Set up this unitary using the negotiated *layout*.

        Reads ``local_i`` and ``alloc_local`` directly from the layout
        (which was populated by the Fortran negotiate step).  The real
        propagator planning happens later in ``gen_operator()`` via
        ``propagator.plan(context)``.
        """
        self.system_size = system_size
        self._layout = layout

        MPI_COMM = layout.SUBCOMM  # noqa: N806
        self.local_i = layout.local_i
        self.alloc_local = layout.alloc_local

        rank = MPI_COMM.Get_rank()

        pt = self.partition_table  # reads from _layout
        self.lb = pt[rank] - 1
        self.ub = pt[rank + 1] - 1

        self.parse_operator_function()
        self.parse_parameter_function()
        self.initial_state = None
        self.final_state = None

    def propagate(self, x: np.ndarray[np.float64]) -> None:
        """Simulation of the action of a :term:`unitary`.

        When implemented, ``propagate`` typically calls into a compiled Python
        extension module using the class attributes that describe the
        parallel partitioning scheme and
        :term:`variational parameters` ``x`` as input. Built-in propagators
        evolve through backend-owned native context buffers; custom subclasses
        that still rely on Python ``initial_state`` / ``final_state`` arrays
        must install those buffers explicitly.

        .. warning::

            Not implemented by the base ``Unitary`` class.

        Examples
        --------

        .. code-block:: python

            # Legacy custom-subclass pattern only.
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

    def destroy(self) -> None:
        """Free memory allocated by Python extension modules in
        the internal planning step ``_UnitaryBase__plan``.

        Memory allocated by compiled Python extension modules is typically not
        managed by the Python garbage collector. These allocations must be freed
        via relevant methods in the extension module to prevent the occurrence
        of `memory leaks <https://en.wikipedia.org/wiki/Memory_leak>`_.

        .. warning::

            Not implemented by the base ``Unitary`` class.
        """
        pass
