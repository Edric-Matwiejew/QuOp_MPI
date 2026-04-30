import warnings

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


def _is_compliant_sparse_csr(row_starts, col_indexes, values):
    """Cheap structural check for canonical sparse CSR buffers.

    Compliant inputs are 1-D, C-contiguous, ``int64``/``complex128``, and
    expressed in 0-based form with ``row_starts[0] == 0``. Sortedness of
    columns within each row is *not* checked here -- the native backend
    validates that precondition via ``check_csr_sorted``.
    """
    if row_starts.dtype != np.int64 or col_indexes.dtype != np.int64:
        return False
    if not row_starts.flags.c_contiguous or not col_indexes.flags.c_contiguous:
        return False
    if row_starts.ndim != 1 or col_indexes.ndim != 1:
        return False
    if row_starts.size == 0 or int(row_starts[0]) != 0:
        return False
    if values is not None:
        if values.dtype != np.complex128 or not values.flags.c_contiguous:
            return False
        if values.ndim != 1 or values.size != col_indexes.size:
            return False
    return True


def _normalize_sparse_csr_operator_args(operator_args):
    """Return canonical 0-based, sorted CSR buffers and a compliance flag.

    When the inputs are already compliant the original arrays are returned
    unchanged (zero-copy fast path) and the native backend will validate
    sortedness. Legacy 1-based or unsorted inputs are converted in-place to
    fresh canonical arrays and ``compliant`` is reported as ``False`` so the
    caller can emit a deprecation warning.
    """
    if len(operator_args) not in (2, 3):
        raise ValueError(
            "Sparse CSR operator arguments must contain row_starts, "
            "col_indexes, and optional values"
        )

    raw_row_starts = np.asarray(operator_args[0])
    raw_col_indexes = np.asarray(operator_args[1])
    raw_values = None if len(operator_args) == 2 else np.asarray(operator_args[2])

    if _is_compliant_sparse_csr(raw_row_starts, raw_col_indexes, raw_values):
        # Zero-copy fast path. Sortedness is validated in the native backend.
        return list(operator_args), True

    # Legacy / non-compliant inputs: build canonical copies. In-tree generators
    # emit 1-based CSR with a per-rank global nnz offset on row_starts; we
    # detect that via row_starts[0] != 0.
    legacy_one_based = raw_row_starts.size > 0 and int(raw_row_starts[0]) != 0

    row_starts = np.ascontiguousarray(raw_row_starts, dtype=np.int64)
    col_indexes = np.ascontiguousarray(raw_col_indexes, dtype=np.int64)
    values = None if raw_values is None else np.ascontiguousarray(
        raw_values, dtype=np.complex128
    )

    # Ensure copies (ascontiguousarray returns the input when types/layout
    # already match) so we never mutate caller buffers.
    if row_starts is raw_row_starts:
        row_starts = row_starts.copy()
    if col_indexes is raw_col_indexes:
        col_indexes = col_indexes.copy()
    if values is not None and values is raw_values:
        values = values.copy()

    if row_starts.size > 0 and int(row_starts[0]) != 0:
        row_starts -= row_starts[0]
    if legacy_one_based:
        col_indexes -= 1

    monotone = bool(np.all(np.diff(row_starts) >= 0)) if row_starts.size > 0 else True
    lengths_match = (
        int(row_starts[-1]) == col_indexes.size
        if row_starts.size > 0
        else col_indexes.size == 0
    )
    if not (monotone and lengths_match):
        raise ValueError(
            "Sparse CSR row_starts must be monotone and terminate at the "
            "local nnz count"
        )

    # Sort columns within each row to satisfy the kernel precondition.
    for row in range(row_starts.size - 1):
        lo = int(row_starts[row])
        hi = int(row_starts[row + 1])
        if hi - lo <= 1:
            continue
        order = np.argsort(col_indexes[lo:hi], kind="stable")
        col_indexes[lo:hi] = col_indexes[lo:hi][order]
        if values is not None:
            values[lo:hi] = values[lo:hi][order]

    normalized = (
        [row_starts, col_indexes]
        if values is None
        else [row_starts, col_indexes, values]
    )
    return normalized, False


class Propagator:

    def __init__(self, propagator):
        self.propagator = propagator
        self.initialised = False
        self.ptr = 0
        self._negotiate_callback = None
        # Strong references to Python buffers borrowed by the native backend
        # via raw pointers (e.g. sparse CSR partitions). Cleared on destroy().
        self._native_pinned_buffers = None
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
        self._native_pinned_buffers = None
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

    def gen_operator(self, operator_args, *, prepare_sparse_csr=False):
        """Generate the native operator from ``operator_args``.

        When ``prepare_sparse_csr`` is true the inputs are normalized to the
        canonical 0-based, sorted CSR contract and the resulting buffers are
        retained on this instance for the lifetime of the native propagator
        (the Fortran backend borrows them as raw pointers).
        """
        self._require_initialised("generate propagator operator")
        prepared_args = operator_args
        if prepare_sparse_csr:
            prepared_args, inputs_were_compliant = _normalize_sparse_csr_operator_args(
                operator_args
            )
            if not inputs_were_compliant:
                warnings.warn(
                    "Legacy sparse CSR inputs are deprecated; return 0-based, "
                    "contiguous, sorted int64/complex128 arrays instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        owned_arrays, ptrs, array_sizes = array_list_to_pointers(prepared_args)
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
        if prepare_sparse_csr:
            # The native backend holds raw pointers into these buffers until
            # destroy(); keep strong Python references alive here.
            if self._native_pinned_buffers is None:
                self._native_pinned_buffers = []
            self._native_pinned_buffers.append(owned_arrays)

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
