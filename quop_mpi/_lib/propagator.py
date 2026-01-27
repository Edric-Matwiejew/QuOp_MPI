import numpy as np

def array_list_to_pointers(arrays):
    """
    Convert a list of NumPy arrays into two Numpy arrays:
    one containing the pointer addresses (as np.intp) and one with their sizes.
    """
    ptrs = []
    array_sizes = []
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        ptrs.append(contiguous.ctypes.data)
        array_sizes.append(contiguous.size)
    ptrs = np.array(ptrs, dtype=np.intp)
    array_sizes = np.array(array_sizes, dtype=np.int64)
    return ptrs, array_sizes

class propagator:

    def __init__(self, propagator):
        self.propagator = propagator
        self.ptr = self.propagator.setup()
        self.initialised = True


    def destroy(self):
        if self.initialised:
            self.propagator.destroy(self.ptr)
            self.initialised = False

    def max_comm_size(self, system_size, available_ranks, constraints, COMM):
        ptrs, array_sizes = array_list_to_pointers(constraints)
        return self.propagator.max_comm_size(self.ptr, system_size, available_ranks,
                                          ptrs, array_sizes, COMM)

    def plan(self, context):
        self.propagator.plan(self.ptr, context.ptr)

    def gen_operator(self, operator_args):
        ptrs, array_sizes = array_list_to_pointers(operator_args)
        self.propagator.gen_operator(self.ptr, ptrs, array_sizes)

    def propagate(self, ts):
        ts_arr = np.ascontiguousarray(ts, dtype=np.float64)
        self.propagator.propagate(self.ptr, ts_arr)
