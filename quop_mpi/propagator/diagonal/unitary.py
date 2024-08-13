from importlib import import_module
import numpy as np
from quop_mpi import config
from quop_mpi.Unitary import Unitary
from quop_mpi.__lib.propagator import propagator

class unitary(Unitary):
    """Compute the action of a :term:`mixing unitary` with a phase_shift
    :term:`operator` or a sequence of mixing-unitaries with phase_shift
    operators (see the :literal:`unitary_n_params` attribute below).

    **Inheritance Diagram:**

        .. graphviz::

            digraph "sphinx-ext-graphviz" {
                rankdir="LR";
                node [fontsize="10"];
                Unitary[label="quop_mpi.Unitary", shape="rectangle"];
                unitary[label="quop_mpi.propagator.phase_shift.unitary", shape="rectangle"];
    
                Unitary -> unitary;
            }

    See :class:`quop_mpi.Unitary`.

    Attributes
    ----------
    unitary_type
        :literal:`'phase_shift'`
    planner
        :literal:`false`
    unitary_n_params
        Set on initialisation to :literal:`1` or more. If :literal:`unitary_n_parameters > 1`,
        the :term:`Operator Function` must return a :literal:`list[csr_matrix]` of
        length :literal:`unitary_n_parameters` containing :literal:`csr_matrix` partitions of
        of :literal:`local_i` rows.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.unitary_type = "phase_shift"

        self.context = None

    def assign_backend(self, backend):

        self.propagator_module = backend.diagonal_propagator

        self.propagators = []
        for i in range(self.unitary_n_params):
            self.propagators.append(propagator(self.propagator_module.propagator_wrapper))

    def plan(self, system_size, MPI_COMM):

        size = MPI_COMM.Get_size()
        rank = MPI_COMM.Get_rank()

        local_i = int(system_size // size + np.ceil((system_size % size) // (rank + 1) / size))

        return local_i, local_i


    def copy_plan(self, ex_unitary):
        pass

    def gen_operator(self, *args):

        for propagator in self.propagators:
            propagator.plan(self.context)

        super().gen_operator(*args)

        diagonals = self.operator

        for i, propagator in enumerate(self.propagators):
            if self.unitary_n_params > 1:
                operator_args = [diagonals[i]]
            else:
                operator_args = [diagonals] 

            propagator.gen_operator(operator_args)

    def propagate(self, gammas):

        for i, (gamma, propagator) in enumerate(zip(gammas, self.propagators)):
            propagator.propagate(gamma)

    def destroy(self):
        for propagator in self.propagators:
            propagator.destroy()
