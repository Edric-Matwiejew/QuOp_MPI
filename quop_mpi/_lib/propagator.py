import numpy as np


def array_list_to_pointers(arrays):
    """
    Convert a list of arrays into contiguous native-call buffers.

    Returns the owned contiguous arrays alongside pointer and size arrays. The
    caller must keep the returned array list alive for the full duration of the
    native call so the raw addresses remain valid.
    """
    owned_arrays = []
    ptrs = []
    array_sizes = []
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        owned_arrays.append(contiguous)
        ptrs.append(contiguous.ctypes.data)
        array_sizes.append(contiguous.size)
    ptrs = np.array(ptrs, dtype=np.intp)
    array_sizes = np.array(array_sizes, dtype=np.int64)
    return owned_arrays, ptrs, array_sizes


class Propagator:

    def __init__(self, propagator):
        self.propagator = propagator
        self.initialised = False
        self.ptr = 0
        self._negotiate_callback = None
        self.ptr, error_code = self.propagator.setup()
        self._raise_propagator_status(
            "initialize propagator",
            error_code,
            {
                100: (
                    RuntimeError,
                    "Cannot initialize propagator: failed to allocate native propagator wrapper",
                ),
            },
        )
        self.ptr = int(self.ptr)
        self.initialised = True

    def _require_initialised(self, action):
        """Raise if the native propagator handle is unavailable."""
        if not self.initialised or self.ptr in (None, 0):
            raise RuntimeError(
                f"Cannot {action}: propagator is not initialised or has been destroyed"
            )

    def destroy(self):
        if self.initialised and self.ptr not in (None, 0):
            self.propagator.destroy(self.ptr)
        self.ptr = 0
        self.initialised = False
        self._negotiate_callback = None

    @staticmethod
    def _raise_propagator_status(action, error_code, status_map=None):
        """Raise a Python exception for a non-zero native status code."""
        if error_code == 0:
            return

        if status_map is None:
            status_map = {}

        exc_type, message = status_map.get(
            int(error_code),
            (
                RuntimeError,
                f"Cannot {action}: backend returned status {int(error_code)}",
            ),
        )
        raise exc_type(message)

    def store_constraints(self, constraints):
        """Store constraint data on the Fortran propagator for negotiate.

        Parameters:
            constraints: List of NumPy arrays (e.g. [np.array(Ns, dtype=np.int32)])
        """
        self._require_initialised("store propagator constraints")
        _owned_arrays, ptrs, array_sizes = array_list_to_pointers(constraints)
        self.propagator.store_constraints(self.ptr, ptrs, array_sizes)

    @property
    def negotiate_callback(self):
        """Return the int64 address of the bind(C) negotiate trampoline."""
        self._require_initialised("fetch propagator negotiate callback")
        if self._negotiate_callback is None:
            self._negotiate_callback = self.propagator.get_negotiate_callback()
        return self._negotiate_callback

    def max_comm_size(self, ci_ptr):
        """Call the propagator's max_comm_size with a quop_mpi_layout_t pointer.

        Parameters:
            ci_ptr: int64 opaque pointer to quop_mpi_layout_t
        """
        self._require_initialised("negotiate propagator layout")
        error_code = self.propagator.max_comm_size(self.ptr, ci_ptr)
        self._raise_propagator_status(
            "negotiate propagator layout",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot negotiate propagator layout: backend configuration failed",
                ),
            },
        )

    def plan(self, context):
        self._require_initialised("plan propagator")
        error_code = self.propagator.plan(self.ptr, context.ptr)
        self._raise_propagator_status(
            "plan propagator",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot plan propagator: backend initialization failed",
                ),
                2: (
                    RuntimeError,
                    "Cannot plan propagator: backend kernel dimension limit exceeded",
                ),
            },
        )

    def gen_operator(self, operator_args):
        self._require_initialised("generate propagator operator")
        _owned_arrays, ptrs, array_sizes = array_list_to_pointers(operator_args)
        error_code = self.propagator.gen_operator(self.ptr, ptrs, array_sizes)
        self._raise_propagator_status(
            "generate propagator operator",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot generate propagator operator: current layout is incompatible",
                ),
                2: (
                    RuntimeError,
                    "Cannot generate propagator operator: backend kernel dimension limit exceeded",
                ),
            },
        )

    def propagate(self, ts):
        self._require_initialised("propagate")
        ts_arr = np.ascontiguousarray(ts, dtype=np.float64)
        error_code = self.propagator.propagate(self.ptr, ts_arr)
        self._raise_propagator_status(
            "propagate",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot propagate: backend precondition"
                    " failed (e.g. kernel dimension limit"
                    " exceeded)",
                ),
            },
        )
