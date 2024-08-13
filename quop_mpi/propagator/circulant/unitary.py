from importlib import import_module
import numpy as np
from ... import config
from ...Unitary import Unitary
from ...__lib.propagator import propagator

class unitary(Unitary):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.unitary_type = "circulant"

        self.context = None

        #TODO check number of unitary params

    def assign_backend(self, backend):

        self.propagator_module = backend.circulant_propagator
        self.propagators = [propagator(self.propagator_module.propagator_wrapper)]

    def plan(self, system_size, MPI_COMM):

        size = MPI_COMM.Get_size()
        rank = MPI_COMM.Get_rank()

        local_i = int(system_size // size + np.ceil((system_size % size) // (rank + 1) / size))

        return local_i, local_i


    def copy_plan(self, ex_unitary):
        pass

    def gen_operator(self, *args):

        self.propagators[0].plan(self.context)
        super().gen_operator(*args)
        self.propagators[0].gen_operator([np.real(self.operator).astype(np.float64)])

    def propagate(self, t):
        self.propagators[0].propagate(t[0])

    def destroy(self):
        self.propagators[0].destroy()
