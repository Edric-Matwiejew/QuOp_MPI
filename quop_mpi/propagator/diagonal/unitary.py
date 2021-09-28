from ...Unitary import Unitary
import numpy as np

class unitary(Unitary):
    """Implements a phase-shift unitary with a diagonal matrix operator
    exponent.

    See :class:`Unitary` for more information.
    """
 
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.unitary_type="diagonal"

        if self.unitary_n_params > 1:
            self.propagate = self.evolve_group
        else:
            self.propagate = self.evolve_single

    def plan(self, system_size, MPI_COMM):

        size = MPI_COMM.Get_size()
        rank = MPI_COMM.Get_rank()

        local_i = int(system_size // size + np.ceil((system_size % size) // (rank + 1) / size))

        return local_i, local_i

    def copy_plan(self, ex_unitary):
        pass

    def evolve_single(self, x):

        self.final_state[:self.local_i] = np.exp(-1j * x * self.operator[:self.local_i]) * self.initial_state[:self.local_i]

    def evolve_group(self, x):

        self.final_state = self.initial_state

        for operator, param in zip(self.operator, x):
            self.final_state[:self.local_i] = np.exp(-1j * param * operator[:self.local_i]) * self.final_state[:self.local_i]


