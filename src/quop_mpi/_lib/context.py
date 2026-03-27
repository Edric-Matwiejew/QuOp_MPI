class Context:
    """Immediate Python adapter for the native context wrapper.

    The state and observables accessors expose host-resident buffers using the
    negotiated layout sizes:

    - ``state`` uses ``alloc_local``
    - ``observables`` uses ``local_i``

    These accessors are collective over ``SUBCOMM``. Backends may satisfy that
    contract trivially (MPI host-only case) or via internal MPI/device transfer
    steps (wavefront).
    """

    @staticmethod
    def _raise_context_status(action, error_code, status_map=None):
        """Raise a Python exception for a non-zero native context status."""
        if error_code == 0:
            return

        if status_map is None:
            status_map = {}

        exc_type, message = status_map.get(
            int(error_code),
            (RuntimeError, f"Cannot {action}: backend returned status {int(error_code)}"),
        )
        raise exc_type(message)

    def __init__(self, backend, comm_info):
        """
        Initialize a context for quantum state management.

        Parameters
        ----------
        backend :
            The backend module (mpi or wavefront).
        comm_info : QuopMpiLayout
            A locked ``QuopMpiLayout`` wrapping the Fortran
            ``quop_mpi_layout_t`` handle.  All partitioning, communicators
            and sizes are read from this single object.
        """
        self.initialised = False
        self.ptr = 0

        self.context_wrapper = backend.context.context_wrapper

        # Store reference to the layout so it stays alive while the context exists
        self._comm_info = comm_info

        # Read the negotiated host partitioning/allocation contract from layout.
        self.system_size = comm_info.system_size
        self.host_local_i = comm_info.local_i
        self.host_local_i_offset = comm_info.local_i_offset
        self.host_alloc_local = comm_info.alloc_local
        self.SUBCOMM = comm_info.subcomm

        # f2py signature: setup(ci_ptr) -> (context_ptr, error_code)
        self.ptr, error_code = self.context_wrapper.setup(comm_info.handle)
        self._raise_context_status(
            "initialize context",
            error_code,
            {
                100: (
                    RuntimeError,
                    "Cannot initialize context: failed to allocate native context wrapper",
                ),
            },
        )
        self.ptr = int(self.ptr)

        self.SUBCOMM.barrier()
        self.initialised = True

    def _require_initialised(self, action):
        """Raise if the native context handle is unavailable."""
        if not self.initialised or self.ptr in (None, 0):
            raise RuntimeError(f"Cannot {action}: context is not initialised or has been destroyed")

    def __enter__(self):
        """Return ``self`` so the native context can be scoped with ``with``."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Deterministically release native context resources at scope exit."""
        self.destroy()
        return False

    def destroy(self):
        if self.initialised and self.ptr not in (None, 0):
            self.context_wrapper.destroy(self.ptr)
        # Mirror the Ansatz destroy path: once native cleanup completes,
        # release borrowed Python-side references to the negotiated layout
        # and communicator so a still-live Context object cannot outlive
        # the layout teardown with stale MPI handles attached.
        self.SUBCOMM = None
        self._comm_info = None
        self.ptr = 0
        self.initialised = False

    @property
    def observables(self):
        """Collectively fetch the host observable buffer over ``SUBCOMM``."""
        if self.initialised:
            observables, error_code = self.context_wrapper.get_observables(
                self.ptr, self.host_local_i
            )
            self._raise_context_status(
                "fetch context observables",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context observables:"
                    " host buffer is smaller than local_i",
                    ),
                    2: (
                        RuntimeError,
                        "Cannot fetch context observables:"
                    " native buffers or transfer metadata"
                    " are unavailable",
                    ),
                    3: (
                        RuntimeError,
                        "Cannot fetch context observables:"
                    " host buffer exceeds internal allocation",
                    ),
                },
            )
            return observables
        return None

    @observables.setter
    def observables(self, obs):
        """Collectively update the host observable buffer over ``SUBCOMM``."""
        self._require_initialised("update context observables")
        error_code = self.context_wrapper.set_observables(self.ptr, obs)
        self._raise_context_status(
            "update context observables",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot update context observables:"
                    " host buffer is smaller than local_i",
                ),
                2: (
                    RuntimeError,
                    "Cannot update context observables:"
                    " native buffers or transfer metadata"
                    " are unavailable",
                ),
                3: (
                    RuntimeError,
                    "Cannot update context observables:"
                    " host buffer exceeds internal allocation",
                ),
            },
        )

    @property
    def state(self):
        """Collectively fetch the host state buffer over ``SUBCOMM``."""
        if self.initialised:
            state, error_code = self.context_wrapper.get_state(self.ptr, self.host_alloc_local)
            self._raise_context_status(
                "fetch context state",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context state: host buffer is smaller than alloc_local",
                    ),
                    2: (
                        RuntimeError,
                        "Cannot fetch context state:"
                        " native buffers or transfer metadata"
                        " are unavailable",
                    ),
                    3: (
                        RuntimeError,
                        "Cannot fetch context state: host buffer exceeds internal allocation",
                    ),
                },
            )
            return state
        return None

    @state.setter
    def state(self, state):
        """Collectively update the host state buffer over ``SUBCOMM``."""
        self._require_initialised("update context state")
        error_code = self.context_wrapper.set_state(self.ptr, state)
        self._raise_context_status(
            "update context state",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot update context state: host buffer is smaller than alloc_local",
                ),
                2: (
                    RuntimeError,
                    "Cannot update context state:"
                    " native buffers or transfer metadata"
                    " are unavailable",
                ),
                3: (
                    RuntimeError,
                    "Cannot update context state: host buffer exceeds internal allocation",
                ),
            },
        )

    def get_expectation_value(self):
        if self.initialised:
            expectation_value, error_code = self.context_wrapper.get_expectation_value(self.ptr)
            self._raise_context_status(
                "fetch context expectation value",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context expectation value:"
                        " native buffers or reduction state"
                        " are unavailable",
                    ),
                },
            )
            return expectation_value
        return None

    def get_state_norm(self):
        if self.initialised:
            state_norm, error_code = self.context_wrapper.get_state_norm(self.ptr)
            self._raise_context_status(
                "fetch context state norm",
                error_code,
                {
                    1: (
                        RuntimeError,
                        "Cannot fetch context state norm:"
                        " native buffers or reduction state"
                        " are unavailable",
                    ),
                },
            )
            return state_norm
        return None
