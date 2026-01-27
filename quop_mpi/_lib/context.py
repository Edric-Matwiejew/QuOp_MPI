import sys
from importlib import import_module
from logging import warn
from .. import config


#def select_context():
#    module_name = None
#    if config.backend == "wavefront":
#        backend_path = "wavefront"
#    elif config.backend != "mpi":
#        warn("Invalid backend selected, defaulting to 'mpi'.")
#
#    if module_name is None:
#        module_name = "mpi"
#
#    context = import_module(module_name, package="quop_mpi").context_wrapper
#
#    return context

class context:
    def __init__(self, backend, system_size, alloc_local, local_i, local_i_offset, SUBCOMM):

        self.system_size = system_size
        self.host_alloc_local = alloc_local
        self.host_local_i = local_i
        self.host_local_i_offset = local_i_offset
        self.SUBCOMM = SUBCOMM
        self.initialised = False

        self.context_wrapper = backend.context.context_wrapper

        self.ptr = self.context_wrapper.setup(
            self.system_size,
            self.host_alloc_local,
            self.host_local_i,
            self.host_local_i_offset,
            self.SUBCOMM.py2f(),
        )

        self.SUBCOMM.barrier()
        self.initialised = True

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.initialised:
            self.context_wrapper.destroy(self.ptr)
            self.initialised = False

    @property
    def observables(self):
        if self.initialised:
            return self.context_wrapper.get_observables(self.ptr, self.host_local_i)
        return None

    @observables.setter
    def observables(self, obs):
        self.context_wrapper.set_observables(self.ptr, obs)

    @property
    def state(self):
        if self.initialised:
            return self.context_wrapper.get_state(self.ptr, self.host_alloc_local)
        return None

    @state.setter
    def state(self, state):
        self.context_wrapper.set_state(self.ptr, state)

    #@property
    #def final_state(self):
    #    if self.initialised:
    #        return self.context_wrapper.get_final_state(self.ptr, self.host_alloc_local)
    #    return None

    #@final_state.setter
    #def final_state(self, state):
    #    self.context_wrapper.set_final_state(self.ptr, state)

    def get_expectation_value(self):
        if self.initialised:
            expectation_value = self.context_wrapper.get_expectation_value(self.ptr)
            return expectation_value if self.SUBCOMM.rank == 0 else None
        return None

    def get_state_norm(self):
        if self.initialised:
            norm = self.context_wrapper.get_state_norm(self.ptr)
            return norm if self.SUBCOMM.rank == 0 else None
        return None
