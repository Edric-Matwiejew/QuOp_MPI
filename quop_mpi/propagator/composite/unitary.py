from importlib import import_module
import numpy as np
from ... import config
from ...Unitary import Unitary
from ...__lib.propagator import propagator

class unitary(Unitary):

    def __init__(self, Ns, *args, **kwargs):

        self.Ns = np.array(Ns, dtype = np.int32)

        super().__init__(*args, **kwargs)

        self.unitary_type = "composite"

        self.context = None

        self.comm_size_constraints = [np.array(Ns, dtype = np.int32)]

        self.planner = True

    def assign_backend(self, backend):

        self.propagator_module = backend.composite_propagator
        self.propagators = [propagator(self.propagator_module.composite_propagator_wrapper)]

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
        g_shape = self.operator.shape
        self.propagators[0].gen_operator([self.Ns, self.operator.flatten()])

    def propagate(self, t):
        self.propagators[0].propagate(t)

    def destroy(self):
        self.propagators[0].destroy()
