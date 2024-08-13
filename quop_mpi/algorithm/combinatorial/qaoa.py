from ... import Ansatz
from ...propagator import diagonal, sparse

####################################
# imports and classes for type hints
####################################

from mpi4py import MPI
from typing import Callable, Union, Iterable

Intracomm = MPI.Intracomm

####################################


class qaoa(Ansatz):
    """Simulate the :ref:`QAOA <QAOA>`.

    See :class:`quop_mpi.Ansatz`.

    Parameters
    ----------
    system_size : int
        :term:`system size` of the simulated :term:`QVA`
    MPI_COMM : Intracomm, optional
        MPI communicator, default :literal:`mpi4py.MPI.COMM_WORLD` 
    """

    def __init__(self, system_size: int, MPI_communicator: Intracomm = MPI.COMM_WORLD):

        super().__init__(system_size, MPI_communicator)

        self.operator_function = None
        self.param_function = None

    #TODO Update docstring

    def set_qualities(self, function: Callable, observables_dict: dict = None):
        """Define the :term:`observables` and :term:`phase-shift unitary` :term:`operator`

        Parameters
        ----------
        function : Callable
            an :term:`Operator Function`
        observables_dict : FunctionDict, optional
            :term:`FunctionDict` for :literal:`function`
        """
        self.set_observables(function, observables_dict)
        #self.set_observables(0)

    def set_params(self, param_function: Callable, param_dict: dict = None):
        """Define the :term:`Parameter Function` for the 
        :term:`phase-shift <phase-shift unitary>` and 
        :term:`mixing <mixing unitary>` unitaries.

        Parameters
        ----------
        param_function : Callable
            a :term:`Parameter Function`
        param_dict : FunctionDict
            :term:`FunctionDict` for :literal:`param_function`
        """
        self.param_function = param_function
        self.param_dict = param_dict

    def setup(self):

        if not self.setup_called:

            if self.observable_function is None:
                raise RuntimeError(
                    "Rank {}: Solution qualities not defined.".format(self.rank)
                )

            if self.param_function is None:

                from ...param.rand import uniform

                self.set_params(uniform)

                #operator_dict= self.operator_dict,
            UQ = diagonal.unitary(
                diagonal.operator.observables,
                parameter_function=self.param_function,
                param_dict=self.param_dict,
            )

            UW = sparse.unitary(
                sparse.operator.hypercube,
                parameter_function=self.param_function,
                param_dict=self.param_dict,
            )

            self.set_unitaries([UQ, UW])

        super().setup()
